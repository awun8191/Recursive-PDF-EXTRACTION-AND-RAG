#!/usr/bin/env python3
r"""
Simplified PDF → OCR → paragraph chunks → per-course JSONL export with ChromaDB, using OLLAMA embeddings only.

Hardcoded settings for efficiency:
- Ollama embeddings with bge-m3:latest model
- Advanced OCR detection enabled
- Optimized chunking parameters
- Predefined paths and settings

Layout assumption (tail-based parsing):
    ...\\\\DEPARTMENT\\\\LEVEL\\\\SEMESTER\\\\COURSE_FOLDER\\\\FILENAME.pdf
"""

# ==================== HARDCODED CONFIGURATION ====================
# Paths
DEFAULT_INPUT_DIR = r"C:\Users\awun8\Documents\SCHOOL\COMPILATION"
DEFAULT_EXPORT_DIR = r"C:\Users\awun8\Documents\Recursive-PDF-EXTRACTION-AND-RAG\data\exported_data"
DEFAULT_CACHE_DIR = r"C:\Users\awun8\Documents\Recursive-PDF-EXTRACTION-AND-RAG\data\ocr_cache"
DEFAULT_CHROMA_DIR = r"C:\Users\awun8\Documents\Recursive-PDF-EXTRACTION-AND-RAG\chroma_db_ollama"

# Processing settings
DEFAULT_WORKERS = 10
DEFAULT_OMP_THREADS = 3
DEFAULT_EXPORT_INCLUDE = "text+ollama-embedding"
DEFAULT_COLLECTION_NAME = "pdfs_ollama_1024"

# OCR settings
DEFAULT_OCR_DPI = 300
DEFAULT_OCR_LANG = "eng"
DEFAULT_MAX_PAGES_PER_PDF = 0  # All pages

# Advanced OCR detection
DEFAULT_TEXT_QUALITY_THRESHOLD = 0.6
DEFAULT_SCANNED_CONTENT_THRESHOLD = 0.7
DEFAULT_MIN_TEXT_DENSITY = 0.05
DEFAULT_SAMPLE_PERCENTAGE = 0.1

# Chunking
DEFAULT_MIN_PAR_CHARS = 200
DEFAULT_MAX_PAR_CHARS = 1800
DEFAULT_CHUNK_OVERLAP = 100  # Characters to overlap between chunks

# Ollama settings
DEFAULT_OLLAMA_MODEL = "bge-m3:latest"
DEFAULT_EMBEDDING_DIM = 1024

# Tesseract path
DEFAULT_TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# ==================================================================

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Set

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

# ------------------ Ollama loader ------------------
def _get_ollama_service():
    try:
        from src.services.Ollama.ollama_service import OllamaService  # type: ignore
        return OllamaService
    except Exception as e:
        raise RuntimeError(f"OllamaService not available: {e}") from e

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
      ...\\DEPARTMENT\\LEVEL\\SEMESTER\\COURSE_FOLDER\\FILENAME
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

def merge_small_paragraphs(paras: List[str], min_chars: int = 200, max_chars: int = 1800, overlap: int = 0) -> List[str]:
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
    
    # Apply sentence-aware splitting and overlap
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
    
    # Apply overlap between chunks
    if overlap > 0 and len(final) > 1:
        overlapped_chunks = []
        for i, chunk in enumerate(final):
            if i == 0:
                # First chunk - no overlap needed
                overlapped_chunks.append(chunk)
            else:
                # Get overlap from previous chunk
                prev_chunk = final[i-1]
                if len(prev_chunk) > overlap:
                    # Take the last 'overlap' characters from previous chunk
                    overlap_text = prev_chunk[-overlap:]
                    # Find a good break point (word boundary)
                    space_idx = overlap_text.find(' ')
                    if space_idx > 0:
                        overlap_text = overlap_text[space_idx+1:]
                    
                    # Combine overlap with current chunk
                    overlapped_chunk = f"{overlap_text} {chunk}"
                    overlapped_chunks.append(overlapped_chunk)
                else:
                    # Previous chunk too short for meaningful overlap
                    overlapped_chunks.append(chunk)
        return overlapped_chunks
    
    return final

def paragraph_chunk(text: str, min_chars: int = 200, max_chars: int = 1800, overlap: int = 0) -> List[str]:
    return merge_small_paragraphs(split_into_paragraphs(text), min_chars, max_chars, overlap)

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

@dataclass
class PDFAnalysisAdvanced:
    pages_analyzed: int
    total_pages: int
    text_quality_score: float
    avg_image_area_ratio: float
    scanned_content_score: float
    text_density_score: float
    total_chars_text_layer: int
    confidence_score: float
    needs_ocr: bool
    analysis_details: Dict[str, Any]

def analyze_text_quality(text: str) -> float:
    """Analyze the quality of extracted text to determine if OCR is needed.
    Returns a score from 0.0 (poor quality, needs OCR) to 1.0 (good quality)."""
    if not text or len(text.strip()) < 50:
        return 0.0
    
    text = text.strip()
    total_chars = len(text)
    
    # Check for readable words vs garbage
    words = re.findall(r'\b[a-zA-Z]{2,}\b', text)
    if not words:
        return 0.0
    
    word_chars = sum(len(w) for w in words)
    word_ratio = word_chars / max(total_chars, 1)
    
    # Check sentence structure
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if not sentences:
        return word_ratio * 0.5  # Some words but no sentences
    
    avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
    sentence_score = min(1.0, avg_sentence_length / 15)  # Optimal around 15 words
    
    # Check for repeated characters (OCR artifacts)
    repeated_chars = len(re.findall(r'(.)\1{4,}', text))  # 5+ repeated chars
    repetition_penalty = min(0.3, repeated_chars / 50)
    
    # Check for proper spacing and punctuation
    spaces = text.count(' ')
    space_ratio = spaces / max(total_chars, 1)
    spacing_score = min(1.0, space_ratio * 10)  # Good spacing around 10%
    
    # Check for mixed case (indicates real text vs all caps OCR)
    has_lower = any(c.islower() for c in text)
    has_upper = any(c.isupper() for c in text)
    case_score = 1.0 if (has_lower and has_upper) else 0.5
    
    # Combine scores
    quality_score = (
        word_ratio * 0.3 +
        sentence_score * 0.25 +
        spacing_score * 0.2 +
        case_score * 0.15 +
        (1 - repetition_penalty) * 0.1
    )
    
    return max(0.0, min(1.0, quality_score))

def smart_page_sampling(doc: fitz.Document, sample_percentage: float = 0.1, min_pages: int = 5, max_pages: int = 20) -> List[int]:
    """Intelligently sample pages for analysis."""
    total_pages = len(doc)
    if total_pages <= min_pages:
        return list(range(total_pages))
    
    # Calculate sample size
    sample_size = max(min_pages, min(max_pages, int(total_pages * sample_percentage)))
    
    if sample_size >= total_pages:
        return list(range(total_pages))
    
    # Always include first, last, and middle pages
    key_pages = {0, total_pages - 1, total_pages // 2}
    
    # Add evenly distributed pages
    step = total_pages / sample_size
    sampled_pages = set()
    for i in range(sample_size):
        page_idx = int(i * step)
        sampled_pages.add(min(page_idx, total_pages - 1))
    
    # Combine key pages with sampled pages
    all_pages = key_pages | sampled_pages
    
    # If we still need more pages, add random ones
    while len(all_pages) < sample_size and len(all_pages) < total_pages:
        import random
        remaining = set(range(total_pages)) - all_pages
        if remaining:
            all_pages.add(random.choice(list(remaining)))
    
    return sorted(list(all_pages))

def detect_image_type(block: Dict[str, Any], page_area: float) -> str:
    """Classify image blocks as scanned, embedded, or decorative."""
    if block.get("type") != 1:  # Not an image block
        return "none"
    
    bbox = block.get("bbox", [0, 0, 0, 0])
    img_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    area_ratio = img_area / max(page_area, 1)
    
    # Large images covering most of the page are likely scanned content
    if area_ratio > 0.8:
        return "scanned"
    elif area_ratio > 0.3:
        return "embedded"
    else:
        return "decorative"

def calculate_text_density(page: fitz.Page) -> float:
    """Calculate text density (text coverage) on a page."""
    page_area = page.rect.width * page.rect.height
    if page_area <= 0:
        return 0.0
    
    text_blocks = page.get_text("dict")["blocks"]
    text_area = 0.0
    
    for block in text_blocks:
        if block.get("type") == 0:  # Text block
            bbox = block.get("bbox", [0, 0, 0, 0])
            text_area += (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    
    return text_area / page_area

def analyze_pdf_advanced(
    doc: fitz.Document, 
    sample_percentage: float = 0.1,
    text_quality_threshold: float = 0.6,
    scanned_content_threshold: float = 0.7,
    min_text_density: float = 0.05
) -> PDFAnalysisAdvanced:
    """Advanced PDF analysis to determine if OCR is needed."""
    total_pages = len(doc)
    if total_pages == 0:
        return PDFAnalysisAdvanced(
            pages_analyzed=0, total_pages=0, text_quality_score=0.0,
            avg_image_area_ratio=0.0, scanned_content_score=0.0,
            text_density_score=0.0, total_chars_text_layer=0,
            confidence_score=0.0, needs_ocr=True,
            analysis_details={"reason": "empty_document"}
        )
    
    # Smart sampling
    sample_pages = smart_page_sampling(doc, sample_percentage)
    
    total_text_quality = 0.0
    total_image_ratio = 0.0
    total_scanned_score = 0.0
    total_text_density = 0.0
    total_chars = 0
    page_details = []
    
    for page_idx in sample_pages:
        page = doc[page_idx]
        page_area = max(1.0, page.rect.width * page.rect.height)
        
        # Extract text and analyze quality
        page_text = page.get_text("text")
        text_quality = analyze_text_quality(page_text)
        total_text_quality += text_quality
        total_chars += len(page_text)
        
        # Analyze images and content type
        raw_dict = page.get_text("rawdict")
        blocks = raw_dict.get("blocks", [])
        
        img_area = 0.0
        scanned_blocks = 0
        total_image_blocks = 0
        
        for block in blocks:
            if block.get("type") == 1:  # Image block
                total_image_blocks += 1
                bbox = block.get("bbox", [0, 0, 0, 0])
                block_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                img_area += block_area
                
                img_type = detect_image_type(block, page_area)
                if img_type == "scanned":
                    scanned_blocks += 1
        
        page_image_ratio = min(1.0, img_area / page_area)
        page_scanned_score = scanned_blocks / max(total_image_blocks, 1) if total_image_blocks > 0 else 0.0
        page_text_density = calculate_text_density(page)
        
        total_image_ratio += page_image_ratio
        total_scanned_score += page_scanned_score
        total_text_density += page_text_density
        
        page_details.append({
            "page": page_idx,
            "text_quality": text_quality,
            "image_ratio": page_image_ratio,
            "scanned_score": page_scanned_score,
            "text_density": page_text_density,
            "char_count": len(page_text)
        })
    
    # Calculate averages
    n_pages = len(sample_pages)
    avg_text_quality = total_text_quality / n_pages
    avg_image_ratio = total_image_ratio / n_pages
    avg_scanned_score = total_scanned_score / n_pages
    avg_text_density = total_text_density / n_pages
    
    # Decision logic with multiple factors
    needs_ocr_reasons = []
    
    # Check text quality
    if avg_text_quality < text_quality_threshold:
        needs_ocr_reasons.append(f"low_text_quality_{avg_text_quality:.2f}")
    
    # Check scanned content
    if avg_scanned_score > scanned_content_threshold:
        needs_ocr_reasons.append(f"high_scanned_content_{avg_scanned_score:.2f}")
    
    # Check text density
    if avg_text_density < min_text_density:
        needs_ocr_reasons.append(f"low_text_density_{avg_text_density:.3f}")
    
    # Check absolute character count
    if total_chars < 200:
        needs_ocr_reasons.append(f"low_char_count_{total_chars}")
    
    needs_ocr = len(needs_ocr_reasons) > 0
    
    # Calculate confidence score
    confidence_factors = [
        avg_text_quality,
        1.0 - avg_scanned_score,
        min(1.0, avg_text_density * 10),  # Scale density to 0-1
        min(1.0, total_chars / 1000)  # Scale char count to 0-1
    ]
    confidence_score = sum(confidence_factors) / len(confidence_factors)
    
    analysis_details = {
        "sample_pages": sample_pages,
        "page_details": page_details,
        "needs_ocr_reasons": needs_ocr_reasons,
        "thresholds": {
            "text_quality": text_quality_threshold,
            "scanned_content": scanned_content_threshold,
            "min_text_density": min_text_density
        }
    }
    
    return PDFAnalysisAdvanced(
        pages_analyzed=n_pages,
        total_pages=total_pages,
        text_quality_score=avg_text_quality,
        avg_image_area_ratio=avg_image_ratio,
        scanned_content_score=avg_scanned_score,
        text_density_score=avg_text_density,
        total_chars_text_layer=total_chars,
        confidence_score=confidence_score,
        needs_ocr=needs_ocr,
        analysis_details=analysis_details
    )

def analyze_pdf_sample(doc: fitz.Document, sample_pages: int = 15) -> PDFAnalysis:
    """Legacy function for backward compatibility."""
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
    # New advanced analysis parameters
    use_advanced_analysis: bool = True,
    text_quality_threshold: float = 0.6,
    scanned_content_threshold: float = 0.7,
    min_text_density: float = 0.05,
    sample_percentage: float = 0.1,
) -> str:
    logger = get_rag_logger("PDFExtractor")
    cache = enhanced_cache or get_enhanced_cache("pdf_extraction_cache.json")
    
    # Generate cache key using enhanced cache
    cache_key = cache.cache_pdf_processing(
        str(path), 
        f"extract_f={force_ocr}_t={image_threshold}_dpi={ocr_dpi}_lang={ocr_lang}_mp={max_pages}_adv={use_advanced_analysis}"
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
        # Use advanced analysis by default, fall back to legacy if disabled
        if use_advanced_analysis and not force_ocr:
            analysis = analyze_pdf_advanced(
                doc, 
                sample_percentage=sample_percentage,
                text_quality_threshold=text_quality_threshold,
                scanned_content_threshold=scanned_content_threshold,
                min_text_density=min_text_density
            )
            need_ocr = analysis.needs_ocr
            
            logger.info(f"ADVANCED ANALYSIS: {analysis.pages_analyzed}/{analysis.total_pages} pages analyzed")
            logger.info(f"Text quality: {analysis.text_quality_score:.3f}, Image ratio: {analysis.avg_image_area_ratio:.2%}")
            logger.info(f"Scanned content: {analysis.scanned_content_score:.3f}, Text density: {analysis.text_density_score:.3f}")
            logger.info(f"Confidence: {analysis.confidence_score:.3f}, Total chars: {analysis.total_chars_text_layer}")
            
            if need_ocr:
                reasons = ", ".join(analysis.analysis_details.get("needs_ocr_reasons", []))
                logger.info(f"OCR required - Reasons: {reasons}")
            else:
                logger.info(f"Direct text extraction - High quality text layer detected")
        else:
            # Legacy analysis for backward compatibility or when force_ocr is True
            legacy_analysis = analyze_pdf_sample(doc, sample_pages=10)
            need_ocr = force_ocr or legacy_analysis.avg_image_area_ratio >= image_threshold or legacy_analysis.total_chars_text_layer < 200
            
            logger.info(f"LEGACY ANALYSIS: {legacy_analysis.pages} pages, {legacy_analysis.avg_image_area_ratio:.2%} image ratio, {legacy_analysis.total_chars_text_layer} text chars")
            
            if need_ocr:
                if force_ocr:
                    logger.info(f"OCR forced by user")
                else:
                    logger.info(f"OCR required (threshold: {image_threshold:.2%}, actual: {legacy_analysis.avg_image_area_ratio:.2%})")
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
    def __init__(self, cache: Optional[EnhancedCache] = None) -> None:
        self._backend = "ollama"
        self._ollama_service = None
        self._dim: Optional[int] = None
        self._fallback_dim = DEFAULT_EMBEDDING_DIM
        self._ollama_model = DEFAULT_OLLAMA_MODEL
        
        # Set cache to ollama_cache directory
        if cache is None:
            from pathlib import Path
            cache_dir = Path(__file__).parent.parent.parent.parent / "data" / "ollama_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache = get_enhanced_cache(str(cache_dir / "embeddings_cache.json"), default_ttl=86400)
        
        self.cache = cache
        self.logger = get_rag_logger("Embedder")
        
        # Initialize Ollama service
        OllamaService = _get_ollama_service()
        try:
            self._ollama_service = OllamaService(model=self._ollama_model)
            self.logger.success(f"Initialized Ollama embeddings with model {self._ollama_model}")
            self.logger.info(f"Ollama embeddings dimension: {self._fallback_dim}")
        except Exception as e:
            self.logger.error(f"Ollama embedding failed: {e}")
            self.logger.error("Make sure Ollama is running and the model is available")
            self.logger.error(f"Try: ollama pull {self._ollama_model}")
            raise RuntimeError(f"Failed to initialize Ollama embeddings. Is Ollama running? Error: {e}") from e
        
        self.logger.info(f"Embedder initialized with backend: {self._backend}, dimensions: {self._fallback_dim}")

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def dim(self) -> int:
        if self._dim is not None:
            return self._dim
        return self._fallback_dim if self._backend == "hash" else -1



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
            log_embedding(f"Generating embeddings for {len(uncached_texts)} texts using Ollama", len(uncached_texts))
            
            try:
                new_embeddings = self._ollama_service.embed(list(uncached_texts), target_dim=self._fallback_dim)
                log_success(f"Generated {len(new_embeddings)} Ollama embeddings")
            except Exception as e:
                self.logger.error(f"Ollama embedding failed: {e}")
                raise RuntimeError(f"Failed to generate Ollama embeddings: {e}") from e
            
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

    def _ensure_dimension(self, embeddings: List[List[float]]) -> None:
        """
        Validate that all embeddings in the batch have the same length and match expected Ollama dimensions.
        """
        if not embeddings:
            return
        detected = len(embeddings[0])
        # Ensure batch consistency
        for e in embeddings:
            if len(e) != detected:
                self.logger.error("Inconsistent embedding dimensions within batch")
                raise RuntimeError("Inconsistent embedding dimensions within batch")
        # Record detected dimension
        self._dim = detected
        # Validate expected size for Ollama
        if detected != self._fallback_dim:
            self.logger.error(f"Unexpected embedding dimension {detected} from Ollama; expected {self._fallback_dim}")
            raise RuntimeError(f"Unexpected embedding dimension {detected} from Ollama; expected {self._fallback_dim}")
        return

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
    group_key_mode = args["group_key_mode"]
    force_ocr = args["force_ocr"]
    image_threshold = args["image_threshold"]

    stats = {"file": str(file_path), "chunks": 0, "tmp_files": [], "group_keys": []}
    start_time = time.time()

    try:
        text = extract_pdf_text(
            file_path,
            force_ocr=force_ocr,
            image_threshold=image_threshold,
            ocr_dpi=DEFAULT_OCR_DPI,
            ocr_lang=DEFAULT_OCR_LANG,
            max_pages=DEFAULT_MAX_PAGES_PER_PDF,
            cache_dir=DEFAULT_CACHE_DIR,
            use_advanced_analysis=True,
            text_quality_threshold=DEFAULT_TEXT_QUALITY_THRESHOLD,
            scanned_content_threshold=DEFAULT_SCANNED_CONTENT_THRESHOLD,
            min_text_density=DEFAULT_MIN_TEXT_DENSITY,
            sample_percentage=DEFAULT_SAMPLE_PERCENTAGE,
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
    chunks = paragraph_chunk(text, min_chars=DEFAULT_MIN_PAR_CHARS, max_chars=DEFAULT_MAX_PAR_CHARS, overlap=DEFAULT_CHUNK_OVERLAP)
    if not chunks:
        processing_time = time.time() - start_time
        return {**stats, "info": "no_chunks", "processing_time": processing_time}

    # Generate Ollama embeddings
    pre_embs: List[List[float]] = []
    embedder = Embedder()
    try:
        pre_embs = embedder.embed(chunks)
    except Exception as e:
        # Convert to simple string to avoid sending un-picklable objects
        raise RuntimeError(f"Embedding failed: {e}") from None

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
            record: Dict[str, Any] = {
                "id": doc_id, 
                "text": chunk, 
                "metadata": meta,
                "embedding": pre_embs[idx],
                "embedding_type": "ollama"
            }
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
    
    embedder = Embedder()
    logger.info(f"Embedder backend: {embedder.backend}")

    total = 0
    files = sorted(export_dir.glob("*.jsonl"))
    logger.info(f"Found {len(files)} course JSONL files to ingest")
    
    docs: List[str] = []; ids: List[str] = []; metas: List[Dict[str, Any]] = []; embs: List[List[float]] = []
    seen_ids: Set[str] = set()  # Track IDs to prevent duplicates
    
    for file_idx, jf in enumerate(files, 1):
        log_file_operation("Processing", str(jf))
        
        with jf.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                rec_id = rec["id"]
                if rec_id in seen_ids:
                    logger.warning(f"Duplicate ID detected, skipping: {rec_id}")
                    continue
                seen_ids.add(rec_id)
                ids.append(rec_id)
                docs.append(rec["text"])
                meta = rec["metadata"]
                # -------- sanitize --------
                def _sanitize(d):
                    for k, v in list(d.items()):
                        # allow scalars or lists/tuples of scalars
                        if isinstance(v, (str, int, float, bool)) or v is None:
                            continue
                        if isinstance(v, (list, tuple)):
                            # stringify any non-scalar element inside list
                            d[k] = [str(x) if not isinstance(x, (str, int, float, bool)) else x for x in v]
                        else:
                            # fallback: convert complex objects (dicts etc.) to JSON-string
                            d[k] = json.dumps(v, ensure_ascii=False)
                    return d
                metas.append(_sanitize(meta))
                
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
    log_section_header("SIMPLIFIED RAG PIPELINE INITIALIZATION")
    
    ap = argparse.ArgumentParser(description="Simplified PDF → OCR → Ollama embeddings → ChromaDB pipeline.")
    # Essential arguments only
    ap.add_argument("-i", "--input-dir", default=DEFAULT_INPUT_DIR, help="Root directory to scan recursively (PDFs only).")
    ap.add_argument("--export-dir", default=DEFAULT_EXPORT_DIR, help="Directory to write per-course JSONL files.")
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, help="Directory to cache OCR outputs.")
    ap.add_argument("-c", "--collection", default=DEFAULT_COLLECTION_NAME, help="ChromaDB collection name.")
    ap.add_argument("-p", "--persist-dir", default=DEFAULT_CHROMA_DIR, help="ChromaDB persist directory.")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Number of worker processes.")
    
    # User-configurable arguments
    ap.add_argument("--group-key-mode", choices=["dept_code_num", "code_num", "dept_code", "dept"], default="dept_code_num",
                    help="How to group per-course files.")
    ap.add_argument("--force-ocr", action="store_true", help="Force OCR for all PDFs.")
    ap.add_argument("--image-threshold", type=float, default=0.75, help="If avg image area ≥ threshold, OCR doc (legacy mode only).")
    
    # Control arguments
    ap.add_argument("--with-chroma", action="store_true", default=True, help="Ingest JSONL into local Chroma.")
    ap.add_argument("--resume", action="store_true", help="Resume from previous session if available")
    ap.add_argument("--max-pdfs", type=int, default=0, help="Limit number of PDFs (0 = no limit).")

    args = ap.parse_args()

    # Configure Tesseract path
    if Path(DEFAULT_TESSERACT_CMD).exists():
        pytesseract.pytesseract.tesseract_cmd = DEFAULT_TESSERACT_CMD
        logger.info(f"Using Tesseract at: {DEFAULT_TESSERACT_CMD}")
    else:
        logger.warning(f"Tesseract not found at {DEFAULT_TESSERACT_CMD}, using system PATH")

    log_step("Validating input directory", 1, 8)
    root = Path(args.input_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        logger.error(f"❌ Input directory does not exist or is not a directory: {root}")
        sys.exit(2)
    log_success(f"Input directory validated: {root}")

    log_step("Discovering PDF files", 2, 8)
    all_pdfs = discover_pdfs(root, ignore_dirs=None)
    all_pdfs.sort()
    log_success(f"Discovered {len(all_pdfs)} PDF files")
    
    # Initialize progress tracking
    log_step("Initializing progress tracking", 3, 8)
    if args.resume:
        progress_tracker = resume_or_create_session(None)
    else:
        progress_tracker = ProgressTracker(None, None)
    
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
            'export_include': DEFAULT_EXPORT_INCLUDE,
            'workers': args.workers,
            'force_ocr': args.force_ocr,
            'image_threshold': args.image_threshold,
            'ocr_dpi': DEFAULT_OCR_DPI
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
    workers = args.workers
    logger.info(f"Using {workers} worker process(es)")
    
    # Initialize enhanced cache (will use data/ollama_cache by default)
    with get_enhanced_cache(default_ttl=86400) as cache:
        log_step("Cache system initialized", 6, 8)
        cache_stats = cache.get_stats()
        logger.info(f"Cache: {cache_stats['active_entries']} active entries, {cache_stats['expired_entries']} expired")
        
        log_section_header(f"PROCESSING {len(pdfs)} PDF FILES")
        logger.info(f"Export mode: {DEFAULT_EXPORT_INCLUDE}")
        logger.info(f"Grouping mode: {args.group_key_mode}")
        logger.info(f"Workers: {workers}")
        logger.info(f"Export directory: {export_dir}")
        logger.info(f"Ollama model: {DEFAULT_OLLAMA_MODEL}")
        logger.info(f"Chunk size: {DEFAULT_MIN_PAR_CHARS}-{DEFAULT_MAX_PAR_CHARS} chars with {DEFAULT_CHUNK_OVERLAP} char overlap")
        
        # Log OCR analysis mode
        logger.info(f"OCR Detection: ADVANCED mode")
        logger.info(f"   Text quality threshold: {DEFAULT_TEXT_QUALITY_THRESHOLD:.2f}")
        logger.info(f"   Scanned content threshold: {DEFAULT_SCANNED_CONTENT_THRESHOLD:.2f}")
        logger.info(f"   Min text density: {DEFAULT_MIN_TEXT_DENSITY:.3f}")
        logger.info(f"   Sample percentage: {DEFAULT_SAMPLE_PERCENTAGE:.1%}")
        
        if args.force_ocr:
            logger.info(f"WARNING: OCR FORCED for all PDFs (analysis will be bypassed)")

    tasks: List[Dict[str, Any]] = []
    for fp in pdfs:
        tasks.append({
            "file_path": str(fp),
            "root": str(root),
            "export_tmp": str(export_tmp),
            "group_key_mode": args.group_key_mode,
            "force_ocr": args.force_ocr,
            "image_threshold": args.image_threshold,
        })

        log_step("Processing PDF files", 7, 8)
        stats: List[Dict[str, Any]] = []
        
        if workers == 1:
            logger.info("🔄 Single-threaded processing mode")
            _init_worker(DEFAULT_TESSERACT_CMD, DEFAULT_OMP_THREADS)
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
                initargs=(DEFAULT_TESSERACT_CMD, DEFAULT_OMP_THREADS),
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
                
                logger.info(f"Collection: {args.collection}")
                logger.info(f"Include embeddings: True (Ollama)")
                logger.info(f"Ollama model: {DEFAULT_OLLAMA_MODEL}")
                
                files_ingested, vectors = _ingest_export_to_chroma(
                    export_dir=export_dir,
                    client=client,
                    collection_name=args.collection,
                    include_embeddings=True,  # Always include embeddings since we always generate them
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
        
        # Note: Session cleanup removed in simplified version
        
        logger.info(f"Check the log file for detailed processing information")
        
    # Save cache before exit
    cache.save()

if __name__ == "__main__":
    main()



# EXECUTION

# python .\convert_to_embeddings.py -i "C:\Users\awun8\Documents\SCHOOL\COMPILATION" --export-dir "C:\Users\awun8\Documents\Recursive-PDF-EXTRACTION-AND-RAG\data\exported_data" --cache-dir "C:\Users\awun8\Documents\Recursive-PDF-EXTRACTION-AND-RAG\data\ocr_cache" --workers 8 --omp-threads 1 --force-ocr --resume --with-chroma --collection pdfs --persist-dir chromadb_storage


# python src/services/RAG/convert_to_embeddings.py -i "C:\Users\awun8\Documents\SCHOOL\COMPILATION" --export-dir "C:\Users\awun8\Documents\Recursive-PDF-EXTRACTION-AND-RAG\data\exported_data" --cache-dir "C:\Users\awun8\Documents\Recursive-PDF-EXTRACTION-AND-RAG\data\ocr_cache" --workers 2 --omp-threads 1 --force-ocr --resume --with-chroma -c pdfs_ollama_1024 -p chroma_db_ollama --prefer-ollama