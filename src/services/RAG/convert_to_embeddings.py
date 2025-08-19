#!/usr/bin/env python3
r"""
PDF → OCR (when needed) → paragraph chunks → per-course JSONL export (and optional Chroma), with MULTI-PROCESS workers.

Enhanced with:
- Beautiful descriptive logging with emojis and colors
- Real-time caching for improved performance
- Gemini embeddings integration
- ChromaDB storage optimization

Install (Python deps):
    pip install chromadb==0.5.5 pymupdf pillow pytesseract duckdb

Windows Tesseract:
    C:\Program Files\Tesseract-OCR\tesseract.exe
    Use --tesseract-cmd if it's not on PATH.

Layout assumption (tail-based parsing exactly as requested):
    ...\\\\DEPARTMENT\\\\LEVEL\\\\SEMESTER\\\\COURSE_FOLDER\\\\FILENAME.pdf
Example:
    C:\\\\Users\\\\...\\\\COMPILATION\\\\EEE\\\\300\\\\1\\\\EEE 313\\\\EEE 313.pdf
Parsed as:
    DEPARTMENT=EEE, LEVEL=300, SEMESTER=1, COURSE_CODE=EEE, COURSE_NUMBER=313
If COURSE_FOLDER doesn't contain a code+number, we try the filename. If both fail, code/number stay empty.
"""

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import fitz  # PyMuPDF
from PIL import Image
import pytesseract

# Optional Chroma (only used if --with-chroma)
import chromadb
from chromadb.config import Settings

# Enhanced utilities
try:
    from src.utils.logging_utils import get_rag_logger, log_section_header, log_step, log_success, log_processing, log_cache_hit, log_cache_miss, log_embedding, log_database, log_file_operation, log_progress
    from src.utils.Caching.enhanced_cache import get_enhanced_cache, EnhancedCache
    from src.utils.progress_tracker import ProgressTracker, ProcessingStatus, resume_or_create_session
    from src.utils.metadata_extractor import MetadataExtractor, extract_document_metadata
except ImportError:
    # Fallback for when running from different directory
    import sys
    from pathlib import Path
    # Add the project root to the path
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "src"))
    from src.utils.logging_utils import get_rag_logger, log_section_header, log_step, log_success, log_processing, log_cache_hit, log_cache_miss, log_embedding, log_database, log_file_operation, log_progress
    from src.utils.Caching.enhanced_cache import get_enhanced_cache, EnhancedCache
    from src.utils.progress_tracker import ProgressTracker, ProcessingStatus, resume_or_create_session
    from src.utils.metadata_extractor import MetadataExtractor, extract_document_metadata

# ------------------ Optional Gemini loader ------------------
def _try_get_gemini_service():
    try:
        from src.services.Gemini.gemini_service import GeminiService  # type: ignore
        return GeminiService
    except Exception:
        return None

# ------------------ Hash helpers ------------------
def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "MISC"

# ------------------ Path → metadata (tail-based, robust) ------------------
SEM_ALIASES = {
    "1": "1", "2": "2",
    "FIRST": "1", "SECOND": "2",
    "SEM1": "1", "SEM2": "2",
    "SEMESTER1": "1", "SEMESTER2": "2",
}

_COURSE_RE = re.compile(r"([A-Za-z]{2,})\s*[-_ ]*\s*(\d{2,3})")

def _norm_semester(tok: str) -> str:
    t = (tok or "").strip().upper().replace(" ", "")
    return SEM_ALIASES.get(t, "")

def _is_level_core(t: str) -> bool:
    return t.isdigit() and len(t) == 3 and t.endswith("00")

def _clean_level(tok: str) -> str:
    # Accept "300", "300L", "300 LEVEL", "300-Level", etc.
    t = (tok or "").strip().upper()
    t = t.replace("LEVEL", "").replace("-", "").replace(" ", "").replace("L", "")
    return t if _is_level_core(t) else ""

def _extract_course_from_text(text: str) -> tuple[str, str]:
    """
    Pull COURSE_CODE and COURSE_NUMBER from folder or filename.
    Examples:
      "EEE 313" -> ("EEE","313")
      "EEE-405" -> ("EEE","405")
      "cpe103_notes.pdf" -> ("CPE","103")
    """
    if not text:
        return "", ""
    m = _COURSE_RE.search(text)
    if not m:
        return "", ""
    code = m.group(1).upper()
    num = m.group(2)
    return code, num

def parse_metadata_from_path(path: str | Path) -> Dict[str, str]:
    """
    Tail-based parsing per user rule:
      ...\DEPARTMENT\LEVEL\SEMESTER\COURSE_FOLDER\FILENAME
    """
    p = Path(path)
    parts = list(p.parts)

    filename = parts[-1] if len(parts) >= 1 else ""
    course_folder = parts[-2] if len(parts) >= 2 else ""
    semester_raw = parts[-3] if len(parts) >= 3 else ""
    level_raw = parts[-4] if len(parts) >= 4 else ""
    department = parts[-5] if len(parts) >= 5 else ""

    semester = _norm_semester(semester_raw)
    level = _clean_level(level_raw)

    # Extract from course-folder, fallback to filename stem
    course_code, course_number = _extract_course_from_text(course_folder)
    if not course_code or not course_number:
        cc2, cn2 = _extract_course_from_text(Path(filename).stem)
        if cc2 and cn2:
            course_code, course_number = cc2, cn2
        else:
            course_code, course_number = "", ""

    # Derive level if missing
    if not level and course_number and course_number.isdigit():
        if len(course_number) >= 3:
            level = course_number[0] + "00"

    # Category quick guess
    cf_up = (course_folder or "").strip().upper()
    fn_up = (filename or "").strip().upper()
    category = ""
    if cf_up == "GENERAL":
        category = "GENERAL"
    elif cf_up in {"PQ", "PQS", "PASTQUESTIONS"} or "PQ" in fn_up or "PAST QUESTION" in fn_up or "PAST QUESTIONS" in fn_up:
        category = "PQ"

    return {
        "DEPARTMENT": department,
        "LEVEL": level,
        "SEMESTER": semester,
        "CATEGORY": category,
        "COURSE_FOLDER": course_folder,
        "COURSE_CODE": course_code,
        "COURSE_NUMBER": course_number,
        "SUBCATEGORY": "",  # not used with strict tail parsing
        "FILENAME": filename,
        "EXT": p.suffix.lower(),
        "STEM": Path(filename).stem,
        "BASENAME": filename,
    }

def build_group_key(tags: Dict[str, str], mode: str = "dept_code_num") -> str:
    """
    Grouping key for per-course export or collection routing.

    Modes:
      - "dept_code_num":  DEPARTMENT-COURSE_CODE-COURSE_NUMBER (e.g. "EEE-EEE-313")
      - "code_num":       COURSE_CODE-COURSE_NUMBER             (e.g. "EEE-313")
      - "dept_code":      DEPARTMENT-COURSE_CODE                (e.g. "EEE-EEE")
      - "dept":           DEPARTMENT                            (e.g. "EEE")
    """
    dept = (tags.get("DEPARTMENT") or "").upper()
    code = (tags.get("COURSE_CODE") or "").upper()
    num  = tags.get("COURSE_NUMBER") or ""

    if mode == "code_num":
        return (f"{code}-{num}".strip("-")) or (dept or "MISC")
    if mode == "dept_code":
        return (f"{dept}-{code}".strip("-")) or "MISC"
    if mode == "dept":
        return dept or "MISC"
    # default
    if dept and code and num:
        return f"{dept}-{code}-{num}"
    if code and num:
        return f"{code}-{num}"
    if dept and code:
        return f"{dept}-{code}"
    return dept or code or "MISC"

# ------------------ Paragraph chunking ------------------
def split_into_paragraphs(text: str) -> List[str]:
    normalized = re.sub(r"\r\n?", "\n", text)
    return [p.strip() for p in re.split(r"\n\s*\n", normalized) if p.strip()]

def merge_small_paragraphs(paras: List[str], min_chars: int = 200, max_chars: int = 1800) -> List[str]:
    merged: List[str] = []
    buf = ""
    for p in paras:
        if not buf:
            buf = p
            continue
        if len(buf) < min_chars or (len(buf) + 2 + len(p) <= max_chars):
            buf = f"{buf}\n\n{p}"
        else:
            merged.append(buf)
            buf = p
    if buf:
        merged.append(buf)
    final: List[str] = []
    for chunk in merged:
        if len(chunk) <= max_chars:
            final.append(chunk)
            continue
        # sentence-aware split
        parts = re.split(r"(?<=[.!?])\s+", chunk)
        cur = ""
        for s in parts:
            s = s.strip()
            if not s:
                continue
            if not cur:
                cur = s
            elif len(cur) + 1 + len(s) <= max_chars:
                cur = f"{cur} {s}"
            else:
                final.append(cur)
                cur = s
        if cur:
            final.append(cur)
    return final

def paragraph_chunk(text: str, min_chars: int = 200, max_chars: int = 1800) -> List[str]:
    return merge_small_paragraphs(split_into_paragraphs(text), min_chars, max_chars)

# ------------------ OCR cache ------------------
def _cache_key(path: Path, extra: str = "") -> str:
    st = path.stat()
    base = f"{str(path.resolve())}|{int(st.st_mtime)}|{extra}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()

def cache_read(cache_dir: str, key: str) -> Optional[str]:
    if not cache_dir:
        return None
    p = Path(cache_dir) / f"{key}.txt"
    if p.exists():
        try:
            return p.read_text(encoding="utf-8")
        except Exception:
            return None
    return None

def cache_write(cache_dir: str, key: str, text: str) -> None:
    if not cache_dir:
        return
    p = Path(cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    (p / f"{key}.txt").write_text(text, encoding="utf-8")

# ------------------ PDF extraction & OCR ------------------
@dataclass
class PDFAnalysis:
    pages: int
    avg_image_area_ratio: float
    total_chars_text_layer: int

def analyze_pdf_sample(doc: fitz.Document, sample_pages: int = 5) -> PDFAnalysis:
    total_ratio = 0.0
    total_chars = 0
    n = min(sample_pages, len(doc))
    if n <= 0:
        return PDFAnalysis(0, 0.0, 0)
    for i in range(n):
        page = doc[i]
        raw = page.get_text("rawdict")
        blocks = raw.get("blocks", [])
        page_area = max(1.0, page.rect.width * page.rect.height)
        img_area = 0.0
        for b in blocks:
            t = b.get("type")
            if t == 1:
                x0, y0, x1, y1 = b.get("bbox", [0, 0, 0, 0])
                img_area += max(0.0, (x1 - x0) * (y1 - y0))
            elif t == 0:
                for line in b.get("lines", []):
                    for span in line.get("spans", []):
                        total_chars += len(span.get("text", ""))
        total_ratio += min(1.0, img_area / page_area)
    return PDFAnalysis(pages=n, avg_image_area_ratio=total_ratio / n, total_chars_text_layer=total_chars)

def extract_pdf_text(
    path: Path,
    force_ocr: bool = False,
    image_threshold: float = 0.75,
    ocr_dpi: int = 300,
    ocr_lang: str = "eng",
    max_pages: int = 0,
    cache_dir: str = "",
    enhanced_cache: Optional[EnhancedCache] = None,
) -> str:
    logger = get_rag_logger("PDFExtractor")
    cache = enhanced_cache or get_enhanced_cache("pdf_extraction_cache.json")
    
    # Generate cache key using enhanced cache
    cache_key = cache.cache_pdf_processing(
        str(path), 
        f"extract_f={force_ocr}_t={image_threshold}_dpi={ocr_dpi}_lang={ocr_lang}_mp={max_pages}"
    )
    
    # Check enhanced cache first
    cached = cache.get(cache_key)
    if cached is not None:
        log_cache_hit(f"PDF text extraction for {path.name}")
        return cached
    
    # Fallback to legacy cache
    legacy_key = _cache_key(path, extra=f"f={force_ocr}|t={image_threshold}|dpi={ocr_dpi}|lang={ocr_lang}|mp={max_pages}")
    legacy_cached = cache_read(cache_dir, legacy_key)
    if legacy_cached is not None:
        log_cache_hit(f"PDF text extraction (legacy cache) for {path.name}")
        # Migrate to enhanced cache
        cache.set(cache_key, legacy_cached, tags=["pdf_extraction", "migrated"])
        return legacy_cached
    
    log_cache_miss(f"PDF text extraction for {path.name}")
    log_processing(f"Extracting text from PDF", str(path.name))

    with fitz.open(path) as doc:
        analysis = analyze_pdf_sample(doc, sample_pages=5)
        need_ocr = force_ocr or analysis.avg_image_area_ratio >= image_threshold or analysis.total_chars_text_layer < 200
        
        logger.info(f"ANALYSIS: {analysis.pages} pages, {analysis.avg_image_area_ratio:.2%} image ratio, {analysis.total_chars_text_layer} text chars")
        
        if need_ocr:
            logger.info(f"OCR required (threshold: {image_threshold:.2%}, actual: {analysis.avg_image_area_ratio:.2%})")
        else:
            logger.info(f"Direct text extraction (sufficient text layer detected)")

        texts: List[str] = []
        page_limit = len(doc) if not max_pages or max_pages <= 0 else min(max_pages, len(doc))
        
        if page_limit < len(doc):
            logger.info(f"Processing {page_limit}/{len(doc)} pages (limited by max_pages)")

        if need_ocr:
            zoom = ocr_dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            logger.info(f"Starting OCR processing at {ocr_dpi} DPI...")
            
            for i in range(page_limit):
                try:
                    if i % 10 == 0 and i > 0:
                        log_progress(i, page_limit, f"OCR processing page {i}")
                    
                    page = doc[i]
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    page_text = pytesseract.image_to_string(img, lang=ocr_lang)
                    texts.append(page_text)
                except Exception as e:
                    logger.warning(f"OCR failed for page {i+1}: {e}")
                    texts.append("")
            
            log_success(f"OCR completed for {page_limit} pages")
        else:
            logger.info(f"Extracting text directly from {page_limit} pages...")
            for i in range(page_limit):
                if i % 50 == 0 and i > 0:
                    log_progress(i, page_limit, f"Text extraction page {i}")
                texts.append(doc[i].get_text("text"))
            
            log_success(f"Direct text extraction completed for {page_limit} pages")

    out = "\n".join(texts)
    
    # Cache in both systems
    cache.set(cache_key, out, tags=["pdf_extraction", "ocr" if need_ocr else "direct"])
    cache_write(cache_dir, legacy_key, out)  # Keep legacy cache for compatibility
    
    logger.info(f"Extracted {len(out)} characters from {path.name}")
    return out

# ------------------ Embedding backend ------------------
class Embedder:
    def __init__(self, prefer_gemini: bool = True, fallback_dim: int = 768, cache: Optional[EnhancedCache] = None) -> None:
        self._backend = "hash"
        self._gemini_service = None
        self._dim: Optional[int] = None
        self._fallback_dim = fallback_dim
        
        # Set cache to gemini_cache directory
        if cache is None:
            from pathlib import Path
            cache_dir = Path(__file__).parent.parent.parent.parent / "data" / "gemini_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache = get_enhanced_cache(str(cache_dir / "embeddings_cache.json"), default_ttl=86400)
        
        self.cache = cache
        self.logger = get_rag_logger("Embedder")
        
        if prefer_gemini:
            GeminiService = _try_get_gemini_service()
            if GeminiService:
                try:
                    # Use default API keys from gemini_api_keys.py
                    self._gemini_service = GeminiService()
                    self._backend = "gemini"
                    self.logger.success(f"Initialized Gemini embeddings with API keys")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize Gemini service: {e}")
                    self._gemini_service = None
                    self._backend = "hash"
            else:
                self.logger.warning("GeminiService not available, falling back to hash embeddings")
        
        self.logger.info(f"Embedder initialized with backend: {self._backend}, dimensions: {self._fallback_dim}")

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def dim(self) -> int:
        if self._dim is not None:
            return self._dim
        return self._fallback_dim if self._backend == "hash" else -1

    def _ensure_dimension(self, vecs: List[List[float]]) -> None:
        if not vecs:
            return
        
        # Handle mixed dimensions from different embedding sources
        dimensions = [len(v) for v in vecs]
        unique_dims = set(dimensions)
        
        if len(unique_dims) > 1:
            # If we have mixed dimensions, normalize to the most common one
            from collections import Counter
            most_common_dim = Counter(dimensions).most_common(1)[0][0]
            
            # Pad or truncate vectors to match the most common dimension
            for i, vec in enumerate(vecs):
                if len(vec) != most_common_dim:
                    if len(vec) < most_common_dim:
                        # Pad with zeros
                        vecs[i] = vec + [0.0] * (most_common_dim - len(vec))
                    else:
                        # Truncate
                        vecs[i] = vec[:most_common_dim]
            
            self.logger.warning(f"Mixed embedding dimensions detected. Normalized to {most_common_dim} dimensions.")
        
        if vecs and self._dim is None:
            self._dim = len(vecs[0])

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        
        # Generate cache keys for each text
        text_hashes = [sha1_text(text) for text in texts]
        cache_keys = [self.cache.cache_embedding(text_hash, self._backend) for text_hash in text_hashes]
        
        # Check cache for existing embeddings
        cached_embeddings = {}
        uncached_texts = []
        uncached_indices = []
        
        for i, (text, cache_key) in enumerate(zip(texts, cache_keys)):
            cached_embedding = self.cache.get(cache_key)
            if cached_embedding is not None:
                cached_embeddings[i] = cached_embedding
                log_cache_hit(f"Embedding for text hash {text_hashes[i][:8]}...")
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
                log_cache_miss(f"Embedding for text hash {text_hashes[i][:8]}...")
        
        # Generate embeddings for uncached texts
        new_embeddings = []
        if uncached_texts:
            log_embedding(f"Generating embeddings for {len(uncached_texts)} texts using {self._backend}", len(uncached_texts))
            
            if self._backend == "gemini" and self._gemini_service is not None:
                try:
                    # Pass target dimension to ensure consistency
                    new_embeddings = self._gemini_service.embed(uncached_texts, target_dim=self._fallback_dim)
                    log_success(f"Generated {len(new_embeddings)} Gemini embeddings")
                except Exception as e:
                    self.logger.error(f"Gemini embedding failed: {e}, falling back to hash embeddings")
                    new_embeddings = [self._hash_embed(t, dim=self._fallback_dim) for t in uncached_texts]
            else:
                new_embeddings = [self._hash_embed(t, dim=self._fallback_dim) for t in uncached_texts]
                log_processing(f"Generated {len(new_embeddings)} hash embeddings")
            
            # Cache the new embeddings
            for i, embedding in enumerate(new_embeddings):
                cache_key = cache_keys[uncached_indices[i]]
                self.cache.set(cache_key, embedding, tags=["embedding", self._backend])
        
        # Combine cached and new embeddings in correct order
        result_embeddings = [None] * len(texts)
        
        # Fill in cached embeddings
        for i, embedding in cached_embeddings.items():
            result_embeddings[i] = embedding
        
        # Fill in new embeddings
        for i, embedding in enumerate(new_embeddings):
            result_embeddings[uncached_indices[i]] = embedding
        
        # Ensure all embeddings are present
        final_embeddings = [emb for emb in result_embeddings if emb is not None]
        self._ensure_dimension(final_embeddings)
        
        log_success(f"Returned {len(final_embeddings)} embeddings ({len(cached_embeddings)} from cache, {len(new_embeddings)} newly generated)")
        return final_embeddings

    @staticmethod
    def _hash_embed(text: str, dim: int = 384) -> List[float]:
        if not text:
            return [0.0] * dim
        vec = [0.0] * dim
        t = text.strip()
        for i in range(max(1, len(t) - 2)):
            tri = t[i:i+3]
            h1 = int(hashlib.md5(tri.encode("utf-8")).hexdigest(), 16)
            h2 = int(hashlib.sha1(tri.encode("utf-8")).hexdigest(), 16)
            vec[h1 % dim] += 1.0
            vec[h2 % dim] += 0.5
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

# ------------------ Chroma helpers ------------------
def build_client(persist_directory: str) -> chromadb.Client:
    persist_directory = str(Path(persist_directory))
    try:
        return chromadb.PersistentClient(path=persist_directory)  # type: ignore[attr-defined]
    except Exception:
        return chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory))

def stable_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)

# ------------------ Discovery ------------------
DEFAULT_IGNORES = {".git", "node_modules", "__pycache__", ".venv", "build", "dist", ".idea", ".vscode"}

def discover_pdfs(root: Path, ignore_dirs: Optional[Sequence[str]] = None) -> List[Path]:
    ignores = set(ignore_dirs or []) | DEFAULT_IGNORES
    out: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ignores and not d.startswith(".")]
        for fname in filenames:
            p = Path(dirpath) / fname
            if p.suffix.lower() == ".pdf":
                out.append(p)
    return out

# ------------------ Per-course exporter (final files) ------------------
class PerCourseExporter:
    def __init__(self, export_dir: Path):
        self.export_dir = export_dir
        self.export_dir.mkdir(parents=True, exist_ok=True)
    def append_file(self, group_key: str, tmp_file: Path) -> None:
        fname = safe_filename(group_key) + ".jsonl"
        final = self.export_dir / fname
        final.parent.mkdir(parents=True, exist_ok=True)
        with final.open("a", encoding="utf-8") as out, tmp_file.open("r", encoding="utf-8") as inp:
            for line in inp:
                out.write(line)

# ------------------ Worker init & task ------------------
def _init_worker(tesseract_cmd: str, omp_threads: int):
    if omp_threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(omp_threads)
        os.environ["OMP_THREAD_LIMIT"] = str(omp_threads)
    if tesseract_cmd:
        try:
            import pytesseract as _pt
            _pt.pytesseract.tesseract_cmd = tesseract_cmd
        except Exception:
            pass

def _process_one_pdf(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker: OCR/chunk one PDF and write to per-PDF temp file grouped by GROUP_KEY.
    Returns stats and list of temp files created.
    """
    import time
    
    file_path = Path(args["file_path"])
    root = Path(args["root"])
    export_tmp = Path(args["export_tmp"])
    export_include = args["export_include"]  # "text" or "text+hash-embedding"
    group_key_mode = args["group_key_mode"]
    force_ocr = args["force_ocr"]
    image_threshold = args["image_threshold"]
    ocr_dpi = args["ocr_dpi"]
    ocr_lang = args["ocr_lang"]
    max_pages = args["max_pages"]
    cache_dir = args["cache_dir"]
    min_par = args["min_par"]
    max_par = args["max_par"]
    prefer_gemini = args["prefer_gemini"]

    stats = {"file": str(file_path), "chunks": 0, "tmp_files": [], "group_keys": []}
    start_time = time.time()

    try:
        text = extract_pdf_text(
            file_path,
            force_ocr=force_ocr,
            image_threshold=image_threshold,
            ocr_dpi=ocr_dpi,
            ocr_lang=ocr_lang,
            max_pages=max_pages,
            cache_dir=cache_dir,
        )
    except Exception as e:
        processing_time = time.time() - start_time
        return {**stats, "error": f"extract_failed: {e}", "processing_time": processing_time}

    if not text.strip():
        processing_time = time.time() - start_time
        return {**stats, "info": "empty_text", "processing_time": processing_time}

    rel_path = stable_relpath(file_path, root)
    stat = file_path.stat()
    tags = parse_metadata_from_path(file_path)
    group_key = build_group_key(tags, mode=group_key_mode)

    file_hash = sha1_text(text)
    chunks = paragraph_chunk(text, min_chars=min_par, max_chars=max_par)
    if not chunks:
        processing_time = time.time() - start_time
        return {**stats, "info": "no_chunks", "processing_time": processing_time}

    # Optional embeddings inside export
    pre_embs: List[List[float]] = []
    if export_include in ["text+hash-embedding", "text+gemini-embedding"]:
        prefer_gemini_emb = export_include == "text+gemini-embedding" or prefer_gemini
        embedder = Embedder(prefer_gemini=prefer_gemini_emb, fallback_dim=768)
        pre_embs = embedder.embed(chunks)

    # Extract comprehensive metadata
    try:
        document_metadata = extract_document_metadata(
            file_path, 
            content=text,
            custom_tags={
                "processing_method": "ocr" if force_ocr or (image_threshold and len(text) < 200) else "direct_text",
                "chunks_generated": len(chunks),
                "embeddings_generated": len(pre_embs),
                "group_key": group_key
            }
        )
        metadata_dict = {
            "document_metadata": {
                "file_hash": document_metadata.file_hash,
                "page_count": document_metadata.page_count,
                "word_count": document_metadata.word_count,
                "department": document_metadata.department,
                "course_code": document_metadata.course_code,
                "course_number": document_metadata.course_number,
                "level": document_metadata.level,
                "document_type": document_metadata.document_type,
                "topics": document_metadata.topics,
                "keywords": document_metadata.keywords,
                "processing_method": document_metadata.processing_method,
                "completeness_score": document_metadata.completeness_score,
                "tags": document_metadata.tags
            },
            "extraction_status": "success"
        }
    except Exception as e:
        metadata_dict = {"extraction_status": "metadata_failed", "metadata_error": str(e)}

    # Temp file unique per PDF per group
    export_tmp.mkdir(parents=True, exist_ok=True)
    tmp_name = f"{safe_filename(group_key)}__{sha1_text(str(file_path))}.jsonl"
    tmp_path = export_tmp / tmp_name
    with tmp_path.open("w", encoding="utf-8") as f:
        total_in_doc = len(chunks)
        for idx, chunk in enumerate(chunks):
            chunk_hash = sha1_text(chunk)
            doc_id = sha1_text(f"{file_hash}:{idx}:{chunk_hash}")
            meta = {
                "path": rel_path,
                "abs_path": str(file_path.resolve()),
                "chunk_index": idx,
                "total_chunks_in_doc": total_in_doc,
                "file_size": stat.st_size,
                "file_mtime": int(stat.st_mtime),
                "ext": file_path.suffix.lower(),
                "file_hash": file_hash,
                "chunk_hash": chunk_hash,
                "GROUP_KEY": group_key,
                # parsed metadata
                "DEPARTMENT": tags.get("DEPARTMENT", ""),
                "LEVEL": tags.get("LEVEL", ""),
                "SEMESTER": tags.get("SEMESTER", ""),
                "CATEGORY": tags.get("CATEGORY", ""),
                "COURSE_FOLDER": tags.get("COURSE_FOLDER", ""),
                "COURSE_CODE": tags.get("COURSE_CODE", ""),
                "COURSE_NUMBER": tags.get("COURSE_NUMBER", ""),
                "SUBCATEGORY": tags.get("SUBCATEGORY", ""),
                "FILENAME": tags.get("FILENAME", ""),
                "STEM": tags.get("STEM", ""),
                # enhanced metadata
                **metadata_dict
            }
            record: Dict[str, Any] = {"id": doc_id, "text": chunk, "metadata": meta}
            if export_include in ["text+hash-embedding", "text+gemini-embedding"]:
                record["embedding"] = pre_embs[idx]
                record["embedding_type"] = "gemini" if export_include == "text+gemini-embedding" else "hash"
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["chunks"] += 1

    processing_time = time.time() - start_time
    stats["tmp_files"] = [str(tmp_path)]
    stats["group_keys"] = [group_key]
    stats["processing_time"] = processing_time
    stats["metadata"] = metadata_dict
    return stats

# ------------------ Post-merge Chroma ingestion ------------------
def _ingest_export_to_chroma(
    export_dir: Path,
    client: chromadb.Client,
    collection_name: str,
    prefer_gemini: bool,
    include_embeddings: bool,
    batch_size: int = 64,
) -> Tuple[int, int]:
    """Read per-course JSONL files and upsert into a single Chroma collection."""
    logger = get_rag_logger("ChromaIngestion")
    
    log_database(f"Creating/accessing collection: {collection_name}")
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    
    embedder = Embedder(prefer_gemini=prefer_gemini, fallback_dim=768)
    logger.info(f"Embedder backend: {embedder.backend}")

    total = 0
    files = sorted(export_dir.glob("*.jsonl"))
    logger.info(f"Found {len(files)} course JSONL files to ingest")
    
    docs: List[str] = []; ids: List[str] = []; metas: List[Dict[str, Any]] = []; embs: List[List[float]] = []
    
    for file_idx, jf in enumerate(files, 1):
        log_file_operation("Processing", str(jf))
        
        with jf.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                ids.append(rec["id"])
                docs.append(rec["text"])
                metas.append(rec["metadata"])
                
                if include_embeddings and "embedding" in rec:
                    embs.append(rec["embedding"])
                
                # Process batch when full
                if len(docs) >= batch_size:
                    if not include_embeddings:
                        log_embedding(f"Generating embeddings for batch", len(docs))
                        embs = embedder.embed(docs)
                    
                    log_database(f"Upserting batch of {len(docs)} documents")
                    collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
                    total += len(docs)
                    
                    # Clear batch
                    docs.clear(); ids.clear(); metas.clear(); embs.clear()
                    
                    if total % (batch_size * 10) == 0:
                        logger.info(f"Processed {total:,} documents so far...")
        
        log_progress(file_idx, len(files), f"Completed {jf.name}")
    
    # Process final batch
    if docs:
        if not include_embeddings:
            log_embedding(f"Generating embeddings for final batch", len(docs))
            embs = embedder.embed(docs)
        
        log_database(f"Upserting final batch of {len(docs)} documents")
        collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
        total += len(docs)
    
    # Persist the collection
    log_database("Persisting ChromaDB collection")
    try:
        client.persist()  # type: ignore[attr-defined]
        log_success("ChromaDB collection persisted successfully")
    except Exception as e:
        logger.warning(f"Could not persist ChromaDB: {e}")
    
    return len(files), total

# ------------------ CLI & orchestration ------------------
def main():
    # Initialize beautiful logging
    logger = get_rag_logger("RAG-Pipeline", "rag_pipeline.log")
    log_section_header("RAG PIPELINE INITIALIZATION")
    
    ap = argparse.ArgumentParser(description="PDF → OCR → paragraph chunks → per-course JSONL export (optional Chroma), with multi-process workers.")
    ap.add_argument("-i", "--input-dir", required=True, help="Root directory to scan recursively (PDFs only).")
    ap.add_argument("--export-dir", type=str, default="vectorize_export", help="Directory to write per-course JSONL files.")
    ap.add_argument("--export-include", choices=["text", "text+hash-embedding", "text+gemini-embedding"], default="text+gemini-embedding",
                    help="Export text+metadata, with hash embeddings, or with Gemini embeddings.")
    ap.add_argument("--group-key-mode", choices=["dept_code_num", "code_num", "dept_code", "dept"], default="dept_code_num",
                    help="How to group per-course files.")

    # Workers
    ap.add_argument("--workers", type=int, default=0, help="Number of worker processes (0 = auto = CPU count - 1, min 1).")
    ap.add_argument("--omp-threads", type=int, default=1, help="Threads per Tesseract inside each worker (1 avoids oversubscription).")

    # OCR / extraction config
    ap.add_argument("--image-threshold", type=float, default=0.75, help="If avg image area ≥ threshold, OCR doc.")
    ap.add_argument("--force-ocr", action="store_true", help="Force OCR for all PDFs.")
    ap.add_argument("--ocr-dpi", type=int, default=300, help="DPI used to render pages for OCR.")
    ap.add_argument("--ocr-lang", type=str, default="eng", help="Tesseract languages, e.g., 'eng+equ' or 'eng+fra'.")
    ap.add_argument("--max-pages-per-pdf", type=int, default=0, help="Limit pages per PDF (0 = all).")
    ap.add_argument("--cache-dir", type=str, default="", help="Directory to cache OCR outputs.")

    # Chunking
    ap.add_argument("--min-par-chars", type=int, default=200, help="Min chars after merging for a paragraph chunk.")
    ap.add_argument("--max-par-chars", type=int, default=1800, help="Max chars per paragraph chunk.")

    # Discovery / control
    ap.add_argument("--ignore-dirs", type=str, default="", help="Comma-separated directories to ignore.")
    ap.add_argument("--max-pdfs", type=int, default=0, help="Limit number of PDFs (0 = no limit).")
    ap.add_argument("--dry-run", action="store_true", help="Extract and export, skip only Chroma ingestion.")

    # Tesseract EXE path (Windows convenience)
    ap.add_argument("--tesseract-cmd", type=str, default="", help=r'Path to tesseract.exe (e.g., "C:\Program Files\Tesseract-OCR\tesseract.exe").')

    # Embedding backend (local vs gemini)
    ap.add_argument("--prefer-gemini", action="store_true", help="Use Gemini embeddings if configured (not recommended for cost).")

    # Optional Chroma (post-merge ingestion)
    ap.add_argument("--with-chroma", action="store_true", help="After export, ingest JSONL into local Chroma.")
    ap.add_argument("-c", "--collection", default="pdfs", help="ChromaDB collection name.")
    ap.add_argument("-p", "--persist-dir", default="chroma_db", help="ChromaDB persist directory.")

    # Progress tracking arguments
    ap.add_argument("--resume", action="store_true", help="Resume from previous session if available")
    ap.add_argument("--progress-file", help="Custom progress file path")
    ap.add_argument("--session-id", help="Custom session ID")
    ap.add_argument("--cleanup-session", action="store_true", help="Clean up progress file after completion")

    args = ap.parse_args()

    # Configure Tesseract path on Windows if provided or use common default
    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd
    else:
        default_win = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.name == "nt" and Path(default_win).exists():
            pytesseract.pytesseract.tesseract_cmd = default_win

    log_step("Validating input directory", 1, 8)
    root = Path(args.input_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        logger.error(f"❌ Input directory does not exist or is not a directory: {root}")
        sys.exit(2)
    log_success(f"Input directory validated: {root}")

    log_step("Discovering PDF files", 2, 8)
    ignores = [d.strip() for d in args.ignore_dirs.split(",") if d.strip()] if args.ignore_dirs else None
    if ignores:
        logger.info(f"🚫 Ignoring directories: {', '.join(ignores)}")
    
    all_pdfs = discover_pdfs(root, ignore_dirs=ignores)
    all_pdfs.sort()
    log_success(f"Discovered {len(all_pdfs)} PDF files")
    
    # Initialize progress tracking
    log_step("Initializing progress tracking", 3, 8)
    if args.resume:
        progress_tracker = resume_or_create_session(args.progress_file)
    else:
        progress_tracker = ProgressTracker(args.progress_file, args.session_id)
    
    # Check for resumable session
    existing_progress = progress_tracker.load_progress()
    if existing_progress and args.resume:
        logger.info(f"Resuming session {existing_progress.session_id}")
        logger.info(f"Previous progress: {existing_progress.completed_files}/{existing_progress.total_files} files completed")
        pdfs = [Path(p) for p in progress_tracker.get_pending_files([str(p) for p in all_pdfs])]
        logger.info(f"Resuming with {len(pdfs)} remaining files")
    else:
        pdfs = all_pdfs
        # Initialize new session
        processing_params = {
            'input_dir': args.input_dir,
            'export_include': args.export_include,
            'workers': args.workers,
            'force_ocr': args.force_ocr,
            'image_threshold': args.image_threshold,
            'max_pages': getattr(args, 'max_pages', 0),  # ✅ Safe access with default
            'ocr_dpi': args.ocr_dpi
        }
        progress_tracker.initialize_session(len(all_pdfs), processing_params)
        logger.info(f"Started new session {progress_tracker.session_id}")
    
    if args.max_pdfs and args.max_pdfs > 0:
        pdfs = pdfs[:args.max_pdfs]
        logger.info(f"Limited to {len(pdfs)} PDFs (max_pdfs setting)")
    
    # Initialize metadata extractor
    metadata_extractor = MetadataExtractor()
    log_success("Progress tracking and metadata extraction initialized")

    log_step("Setting up export directories", 4, 8)
    export_dir = Path(args.export_dir).expanduser().resolve()
    export_tmp = export_dir / "_tmp"
    export_tmp.mkdir(parents=True, exist_ok=True)
    log_success(f"Export directory ready: {export_dir}")

    log_step("Configuring worker processes", 5, 8)
    cpu = os.cpu_count() or 2
    workers = (cpu - 1) if args.workers == 0 else max(1, args.workers)
    logger.info(f"System CPUs: {cpu}, Using {workers} worker process(es)")
    
    # Initialize enhanced cache (will use data/gemini_cache by default)
    with get_enhanced_cache(default_ttl=86400) as cache:
        log_step("Cache system initialized", 6, 8)
        cache_stats = cache.get_stats()
        logger.info(f"Cache: {cache_stats['active_entries']} active entries, {cache_stats['expired_entries']} expired")
        
        log_section_header(f"PROCESSING {len(pdfs)} PDF FILES")
        logger.info(f"Export mode: {args.export_include}")
        logger.info(f"Grouping mode: {args.group_key_mode}")
        logger.info(f"Workers: {workers}")
        logger.info(f"Export directory: {export_dir}")

    tasks: List[Dict[str, Any]] = []
    for fp in pdfs:
        tasks.append({
            "file_path": str(fp),
            "root": str(root),
            "export_tmp": str(export_tmp),
            "export_include": args.export_include,
            "group_key_mode": args.group_key_mode,
            "force_ocr": args.force_ocr,
            "image_threshold": args.image_threshold,
            "ocr_dpi": args.ocr_dpi,
            "ocr_lang": args.ocr_lang,
            "max_pages": args.max_pages_per_pdf,
            "cache_dir": args.cache_dir,
            "min_par": args.min_par_chars,
            "max_par": args.max_par_chars,
            "prefer_gemini": args.prefer_gemini,
        })

        log_step("Processing PDF files", 7, 8)
        stats: List[Dict[str, Any]] = []
        
        if workers == 1:
            logger.info("🔄 Single-threaded processing mode")
            _init_worker(pytesseract.pytesseract.tesseract_cmd, args.omp_threads)
            for i, t in enumerate(tasks, 1):
                file_name = Path(t['file_path']).name
                log_processing(f"Processing PDF {i}/{len(tasks)}", file_name)
                result = _process_one_pdf(t)
                
                if result.get('error'):
                    logger.error(f"ERROR {file_name}: {result['error']}")
                elif result.get('info'):
                    logger.info(f"INFO {file_name}: {result['info']}")
                else:
                    log_success(f"{file_name}: {result.get('chunks', 0)} chunks extracted")
                
                stats.append(result)
                log_progress(i, len(tasks), f"Completed {file_name}")
        else:
            logger.info(f"Multi-threaded processing with {workers} workers")
            with cf.ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_worker,
                initargs=(pytesseract.pytesseract.tesseract_cmd, args.omp_threads),
            ) as ex:
                futures = [ex.submit(_process_one_pdf, t) for t in tasks]
                completed = 0
                
                for fut in cf.as_completed(futures):
                    completed += 1
                    res = fut.result()
                    file_name = Path(res.get('file', '?')).name
                    file_path = res.get('file', '')
                    
                    # Update progress tracking
                    if res.get('error'):
                        logger.error(f"ERROR [{completed}/{len(tasks)}] {file_name}: {res['error']}")
                        progress_tracker.update_file_status(
                            file_path, ProcessingStatus.FAILED,
                            chunks=res.get('chunks', 0),
                            embeddings=res.get('embeddings', 0),
                            processing_time=res.get('processing_time', 0.0),
                            error=res.get('error'),
                            metadata=res.get('metadata', {})
                        )
                    elif res.get('info'):
                        logger.info(f"INFO [{completed}/{len(tasks)}] {file_name}: {res['info']}")
                        progress_tracker.update_file_status(
                            file_path, ProcessingStatus.SKIPPED,
                            chunks=res.get('chunks', 0),
                            embeddings=res.get('embeddings', 0),
                            processing_time=res.get('processing_time', 0.0),
                            metadata=res.get('metadata', {})
                        )
                    else:
                        log_success(f"[{completed}/{len(tasks)}] {file_name}: {res.get('chunks', 0)} chunks")
                        progress_tracker.update_file_status(
                            file_path, ProcessingStatus.COMPLETED,
                            chunks=res.get('chunks', 0),
                            embeddings=res.get('embeddings', 0),
                            processing_time=res.get('processing_time', 0.0),
                            metadata=res.get('metadata', {})
                        )
                    
                    stats.append(res)
                    
                    if completed % 10 == 0 or completed == len(tasks):
                        log_progress(completed, len(tasks), f"Processed {completed} files")
                        # Save progress periodically
                        progress_tracker.save_progress()

        log_step("Merging temporary files", 8, 8)
        # Merge temp files into final per-course JSONL
        exporter = PerCourseExporter(export_dir)
        merged = 0
        for s in stats:
            for tmp in s.get("tmp_files", []):
                tmp_path = Path(tmp)
                group_key_part = tmp_path.name.split("__", 1)[0]
                exporter.append_file(group_key_part, tmp_path)
                merged += 1
        
        log_success(f"Merged {merged} temporary files into per-course JSONL files")
        
        # Cleanup temp files
        try:
            temp_files = list(export_tmp.glob("*.jsonl"))
            for f in temp_files:
                f.unlink(missing_ok=True)
            export_tmp.rmdir()
            logger.info(f"Cleaned up {len(temp_files)} temporary files")
        except Exception as e:
            logger.warning(f"Could not clean up temp files: {e}")

        # Calculate final statistics
        total_files = sum(1 for s in stats if s.get("chunks", 0) > 0)
        total_chunks = sum(s.get("chunks", 0) for s in stats)
        failed_files = sum(1 for s in stats if s.get("error"))
        skipped_files = sum(1 for s in stats if s.get("info") and not s.get("error"))
        
        log_section_header("PROCESSING COMPLETE")
        log_success(f"Successfully processed {total_files} PDF files")
        logger.info(f"Total chunks extracted: {total_chunks:,}")
        if failed_files > 0:
            logger.warning(f"Failed files: {failed_files}")
        if skipped_files > 0:
            logger.info(f"Skipped files: {skipped_files}")
        logger.info(f"Per-course JSONL files saved to: {export_dir}")
        
        # Cache statistics
        final_cache_stats = cache.get_stats()
        logger.info(f"Final cache: {final_cache_stats['active_entries']} entries, {final_cache_stats['total_entries']} total")

        # Optional Chroma ingestion post-merge
        if args.with_chroma:
            log_step("ChromaDB ingestion", 9, 9)
            log_database(f"Initializing ChromaDB client at {args.persist_dir}")
            
            try:
                client = build_client(args.persist_dir)
                include_embeddings = args.export_include in ["text+hash-embedding", "text+gemini-embedding"]
                
                logger.info(f"Collection: {args.collection}")
                logger.info(f"Include embeddings: {include_embeddings}")
                logger.info(f"Prefer Gemini: {args.prefer_gemini}")
                
                files_ingested, vectors = _ingest_export_to_chroma(
                    export_dir=export_dir,
                    client=client,
                    collection_name=args.collection,
                    prefer_gemini=args.prefer_gemini,
                    include_embeddings=include_embeddings,
                    batch_size=64,
                )
                
                log_success(f"ChromaDB ingestion complete!")
                logger.info(f"Course files ingested: {files_ingested}")
                logger.info(f"Vector embeddings stored: {vectors:,}")
                logger.info(f"ChromaDB location: {Path(args.persist_dir).resolve()}")
                
            except Exception as e:
                logger.error(f"ChromaDB ingestion failed: {e}")
                raise
        
        log_section_header("RAG PIPELINE COMPLETE")
        log_success("All processing steps completed successfully!")
        
        # Final progress summary
        progress_summary = progress_tracker.get_progress_summary()
        logger.info(f"Session {progress_summary['session_id']} completed:")
        logger.info(f"  - Total files: {progress_summary['total_files']}")
        logger.info(f"  - Completed: {progress_summary['completed_files']}")
        logger.info(f"  - Failed: {progress_summary['failed_files']}")
        logger.info(f"  - Skipped: {progress_summary['skipped_files']}")
        logger.info(f"  - Total chunks: {progress_summary['total_chunks']:,}")
        logger.info(f"  - Total embeddings: {progress_summary['total_embeddings']:,}")
        logger.info(f"  - Completion: {progress_summary['completion_percentage']:.1f}%")
        
        # Export detailed results
        results_file = progress_tracker.export_results()
        logger.info(f"Detailed results exported to: {results_file}")
        
        # Cleanup session if requested
        if args.cleanup_session:
            progress_tracker.cleanup_session()
            logger.info("Progress session cleaned up")
        
        logger.info(f"Check the log file for detailed processing information")
        
    # Save cache before exit
    cache.save()

if __name__ == "__main__":
    main()
