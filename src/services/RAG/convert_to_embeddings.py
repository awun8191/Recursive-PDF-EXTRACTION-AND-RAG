#!/usr/bin/env python3
"""
Streamlined: PDF -> (auto OCR w/ PaddleOCR) -> chunk -> dedupe -> BGE-M3 (Cloudflare)
-> per-file JSONL -> immediate Chroma upsert -> real-time billing -> resume.

Major fixes:
- Fixed PaddleOCR configuration conflict (use_angle_cls vs use_textline_orientation)
- Fixed Windows file permission issues with progress saving
- No hardcoded Cloudflare credentials; strictly require env vars.
- Proper EasyOCR fallback wired into OCR flow.
- Retry/backoff for Cloudflare API (429/5xx/network).
- Saner OCR image-size check using nbytes; DPI ladder and env-tunable cap.
- Correct cache keying and no caching of near-empty OCR/text.
- Fixed multiprocessing status merge order so "pending" isn't resurrected.
- Safer Chroma upsert and metadata sanitization.

Env:
  CLOUDFLARE_ACCOUNT_ID          (required)
  CLOUDFLARE_API_TOKEN           (required)
  CF_PRICE_PER_M_TOKENS=0.012    # USD per 1M input tokens (BGE-M3 input price)
  CF_EMBED_MAX_BATCH=96
  OMP_NUM_THREADS=4
  BILLING_ENABLED=1
  PADDLE_LANG=en                  # optional; e.g., en, fr, de, ar, hi
  EASYOCR_GPU=0                   # optional; set 1 to enable GPU for EasyOCR
  OCR_MAX_IMAGE_BYTES=67108864    # 64MB default; adjust as needed
"""

from __future__ import annotations

import os, re, sys, json, time, argparse, signal
from dataclasses import dataclass
from pathlib import Path

# Make repository root and repo 'src' directory importable when this script
# is executed directly (so "from src.services..." works). This keeps
# package-style imports intact while allowing execution from the file tree.
try:
    # convert_to_embeddings.py is at: <repo>/src/services/RAG/convert_to_embeddings.py
    # parents[3] -> <repo>
    repo_root = Path(__file__).resolve().parents[3]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    src_dir = repo_root / "src"
    src_dir_str = str(src_dir)
    if src_dir.exists() and src_dir_str not in sys.path:
        sys.path.insert(0, src_dir_str)
except Exception:
    # Best-effort only; do not fail if path manipulation isn't possible
    pass
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError

# Optional: your own tracker (kept unused to avoid conflicts)
# from src.utils.progress_tracker import ProgressTracker, ProcessingStatus

import fitz  # PyMuPDF
from PIL import Image
import requests
# ChromaDB handled via src.services.RAG.chroma_store

# New modular imports
from src.services.RAG.log_utils import setup_logging, snapshot
from src.services.RAG.billing import Billing
from src.services.RAG.ocr_engine import extract_text as ocr_extract_text
from src.services.RAG.chroma_store import chroma_client, chroma_upsert_jsonl
from src.services.RAG.progress_store import (
    load_progress,
    save_progress,
    safe_file_replace,
    should_skip,
)
from src.services.RAG.path_meta import parse_path_meta
from src.services.RAG.cache_utils import sha256_file
from src.services.RAG.chunking import chunk, dedupe, sha1_text

# Optional OpenCV for image handling (Paddle likes numpy arrays)
try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except Exception:
    OPENCV_AVAILABLE = False

# PaddleOCR (optional)
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except Exception:
    PADDLE_AVAILABLE = False

# ====== Small helpers ======

def log(msg: str) -> None:
    print(msg, flush=True)

# sha256_file and sha1_text are provided by shared utils

def now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()

# ====== Token counting (tiktoken; fallback ≈chars/4) ======

class TokenCounter:
    def __init__(self) -> None:
        self._enc = None
        try:
            import tiktoken
            self._enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._enc = None

    def count_batch(self, texts: List[str]) -> int:
        if self._enc:
            return sum(len(self._enc.encode(t)) for t in texts)
        return sum(max(1, len(t) // 4) for t in texts)

# ====== Billing ======
# Externalized: using Billing from billing module (imported above)

# ====== Cloudflare BGE-M3 client ======

@dataclass
class RetryCfg:
    tries: int = 5
    backoff: float = 1.5  # exponential factor
    max_sleep: float = 20.0

class CFEmbeddings:
    def __init__(self, account_id: str, api_token: str, batch_max: int, retry: RetryCfg | None = None):
        self.url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/baai/bge-m3"
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"})
        self.batch_max = max(1, min(100, batch_max))
        self.counter = TokenCounter()
        self.retry = retry or RetryCfg()

    def _post_embed(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        sleep = 0.5
        for attempt in range(1, self.retry.tries + 1):
            try:
                r = self.s.post(self.url, json=payload, timeout=90)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"{r.status_code} {r.text[:200]}")
                r.raise_for_status()
                return r.json()
            except Exception:
                if attempt == self.retry.tries:
                    raise
                time.sleep(min(self.retry.max_sleep, sleep))
                sleep *= self.retry.backoff
        raise RuntimeError("unhandled retry loop")

    def embed(self, texts: List[str], src_file: str, batch_size: int) -> Tuple[List[List[float]], int]:
        bsz = min(batch_size, self.batch_max)
        out: List[List[float]] = []
        total_tokens = 0
        for i in range(0, len(texts), bsz):
            sub = texts[i:i+bsz]
            payload = {"text": sub, "truncate_inputs": True}
            js = self._post_embed(payload)
            data = js.get("result", {}).get("data")
            if not isinstance(data, list):
                raise RuntimeError(f"Bad embedding response: {str(js)[:200]}")
            out.extend(data)

            tokens = self.counter.count_batch(sub)
            total_tokens += tokens
        return out, total_tokens

# ====== ChromaDB handled by src.services.RAG.chroma_store ======

# ====== Path metadata handled by src.services.RAG.path_meta.parse_path_meta ======

# ====== OCR decision ======

def need_ocr(doc: fitz.Document, sample_pages: int = 8, min_chars_per_page: int = 200) -> bool:
    n = min(sample_pages, len(doc))
    if n == 0:
        return True
    low = 0
    for i in range(n):
        txt = doc[i].get_text("text")
        if len(txt) < min_chars_per_page:
            low += 1
    return (low / max(1, n)) >= 0.6

# ====== PaddleOCR (per-process singleton) ======

_PADDLE: Optional['PaddleOCR'] = None

def get_paddle_ocr(lang: str = "en") -> Optional['PaddleOCR']:
    global _PADDLE
    if not PADDLE_AVAILABLE:
        return None
    if _PADDLE is None:
        try:
            log(f"[INFO] Initializing PaddleOCR with language: {lang}")
            # Fixed: Use modern PaddleOCR parameter names
            # use_textline_orientation replaces deprecated use_angle_cls
            local_model_dir = './paddle_models'  # Adjust this path as needed
            if os.path.exists(local_model_dir):
                log(f"[INFO] Using local models from {local_model_dir}")
                _PADDLE = PaddleOCR(
                    use_textline_orientation=True,  # Modern parameter for text orientation
                    lang=lang,
                    det_model_dir=f'{local_model_dir}/det',
                    rec_model_dir=f'{local_model_dir}/rec',
                    cls_model_dir=f'{local_model_dir}/cls'
                )
            else:
                log(f"[INFO] Using default PaddleOCR models")
                # Fallback to default models if local models not available
                _PADDLE = PaddleOCR(
                    use_textline_orientation=True,  # Modern parameter for text orientation
                    lang=lang,
                )
            log(f"[INFO] PaddleOCR initialized successfully")
        except Exception as e:
            log(f"[WARN] Failed to initialize PaddleOCR: {e}")
            _PADDLE = None
    return _PADDLE

def _pixmap_to_numpy(pix: fitz.Pixmap) -> 'np.ndarray':  # type: ignore[name-defined]
    if not OPENCV_AVAILABLE:
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return np.array(img) if 'np' in globals() else __import__('numpy').array(img)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    return arr

def ocr_page_with_paddle_or_tesseract(page: fitz.Page, dpi: int = 300, lang: str = "en") -> str:
    # Prefer Paddle
    ocr = get_paddle_ocr(lang=lang) if PADDLE_AVAILABLE else None

    last_img = None
    ladder = [dpi, 240, 200, 150, 100, 72] if dpi >= 240 else [dpi, 150, 100, 72]
    MAX_OCR_BYTES = int(os.getenv("OCR_MAX_IMAGE_BYTES", str(64 * 1024 * 1024)))  # 64MB

    for attempt_dpi in ladder:
        try:
            zoom = attempt_dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = _pixmap_to_numpy(pix)
            last_img = img

            nbytes = getattr(img, "nbytes", img.size)
            if nbytes > MAX_OCR_BYTES:
                if attempt_dpi == ladder[-1]:
                    log(f"[WARN] Page image {nbytes}B too large at {attempt_dpi} DPI, skipping OCR")
                    break
                log(f"[WARN] Page image {nbytes}B > {MAX_OCR_BYTES}B at {attempt_dpi} DPI; trying lower DPI")
                continue

            if ocr is not None:
                # PaddleOCR 3.x: call without det/rec/cls kwargs
                res = ocr.ocr(img)
                flat = []
                for blk in res:
                    if blk:  # Check if block is not None
                        flat.extend(blk if isinstance(blk, list) else [blk])
                lines = []
                for line in flat:
                    try:
                        if line and len(line) >= 2:  # Ensure line has expected structure
                            box, (text, conf) = line
                            xs = [p[0] for p in box]; ys = [p[1] for p in box]
                            cx = sum(xs)/4.0; cy = sum(ys)/4.0
                            lines.append((text, float(conf), (cx, cy)))
                    except Exception:
                        continue
                lines.sort(key=lambda t: (round(t[2][1]/16.0), round(t[2][0]/16.0)))
                texts = [t for (t, conf, _) in lines if t and conf >= 0.35]
                if texts:
                    if attempt_dpi != dpi:
                        log(f"[OCR] Paddle used {attempt_dpi} DPI instead of {dpi}")
                    return "\n".join(texts)
        except Exception as e:
            if any(s in str(e).lower() for s in ("memory", "alloc")) and attempt_dpi != ladder[-1]:
                log(f"[WARN] Paddle OOM at {attempt_dpi} DPI; trying lower DPI")
                continue
            log(f"[WARN] Paddle error at {attempt_dpi} DPI: {e}")
            break  # don't keep hammering if it's not memory

    # Last resort: Tesseract (only if sane)
    tess_cmd = os.getenv("TESSERACT_CMD")
    if tess_cmd and os.path.exists(tess_cmd):
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = tess_cmd

            # Tesseract language code mapping (so you can keep --ocr-lang en)
            tess_lang_map = {"en":"eng","fr":"fra","de":"deu","es":"spa","it":"ita","pt":"por","nl":"nld"}
            tess_lang = tess_lang_map.get(lang, lang)

            # Verify tessdata exists and language file is present
            tess_prefix = os.getenv("TESSDATA_PREFIX")
            if tess_prefix:
                lang_path = Path(tess_prefix) / "tessdata" / f"{tess_lang}.traineddata"
                if not lang_path.exists():
                    log(f"[WARN] Tesseract missing {tess_lang}.traineddata under {lang_path.parent}; skipping Tesseract")
                    return ""
            # Render minimal if needed
            if last_img is None:
                pix = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
                last_img = _pixmap_to_numpy(pix)

            img_pil = Image.fromarray(last_img)
            tesseract_text = pytesseract.image_to_string(img_pil, lang=tess_lang)
            if tesseract_text.strip():
                log("[OCR] EasyOCR fallback used")
                return tesseract_text.strip()
        except Exception as te:
            log(f"[WARN] Tesseract fallback failed: {te}")
    return ""

# ====== Text extract (auto OCR with Paddle, fallback to Tesseract) ======
# Externalized: using extract_text from ocr_engine module (imported above)
extract_text = ocr_extract_text

# ====== Chunking & dedupe ======

# Chunking and deduplication are provided by src.services.RAG.chunking

# ====== Worker ======

def _worker_init_for_ocr(lang: str) -> None:
    """Preload PaddleOCR inside worker processes so the first task doesn't pay init cost."""
    if PADDLE_AVAILABLE:
        try:
            from src.services.RAG.ocr_engine import get_paddle_ocr as _warm
            _ = _warm(lang)
            log(f"[WARMUP] Worker preloaded PaddleOCR (lang={lang})")
        except Exception as e:
            log(f"[WARN] Worker OCR warmup failed: {e}")

def process_one(pdf_path: str, root: str, export_tmp: str,
                cache_dir: str, cf_acct: str, cf_token: str,
                billing_file: str, embed_batch: int,
                force_ocr: bool, ocr_policy: str,
                ocr_dpi: int, ocr_lang: str) -> Dict[str, Any]:

    path = Path(pdf_path)
    rel = str(Path(pdf_path).resolve())
    log(f"[START] Processing {path.name}")


    log(f"[META] Parsing metadata for {path.name}")
    meta_path = parse_path_meta(path)
    
    log(f"[TEXT] Extracting text from {path.name}")
    text = extract_text(path, Path(cache_dir), force_ocr, ocr_policy, ocr_dpi=ocr_dpi, ocr_lang=ocr_lang)
    if not text.strip():
        return {"file": rel, "skip": True, "reason": "empty_text"}
    log(f"[EXTRACT] {path.name} chars={len(text)} snapshot='{snapshot(text)}'")

    log(f"[CHUNK] Chunking text for {path.name}")
    chunks_all = chunk(text)
    uniq, dup_map = dedupe(chunks_all)
    if not uniq:
        return {"file": rel, "skip": True, "reason": "no_chunks"}
    log(f"[CHUNK] {path.name} uniq_chunks={len(uniq)} first_chunk='{snapshot(uniq[0]) if uniq else ''}'")

    # Cloudflare embeddings
    log(f"[EMBED] Creating embeddings for {path.name}")
    cf = CFEmbeddings(cf_acct, cf_token, int(os.getenv("CF_EMBED_MAX_BATCH", "96")))
    vecs, total_tokens = cf.embed(uniq, src_file=rel, batch_size=embed_batch)
    if len(vecs) != len(uniq):
        return {"file": rel, "error": "embedding_mismatch"}

    # Write per-file JSONL (in tmp dir)
    tmp_dir = Path(export_tmp); tmp_dir.mkdir(parents=True, exist_ok=True)
    group = meta_path["GROUP_KEY"]
    jsonl_name = f"{re.sub(r'[^A-Za-z0-9._-]+','_',group)}__{sha1_text(rel)}.jsonl"
    jsonl_tmp = tmp_dir / jsonl_name

    file_hash = sha256_file(path)[:16]
    st = path.stat()
    doc_hash = sha1_text(text)
    with jsonl_tmp.open("w", encoding="utf-8") as out:
        total = len(chunks_all)
        k = 0
        for idx, ch in enumerate(chunks_all):
            if idx in dup_map:
                continue
            chash = sha1_text(ch)
            rid = sha1_text(f"{doc_hash}:{idx}:{chash}")
            md = {
                "path": str(path),
                "chunk_index": idx,
                "total_chunks_in_doc": total,
                "file_size": st.st_size,
                "file_mtime": int(st.st_mtime),
                "file_hash": file_hash,
                "chunk_hash": chash,
                **meta_path,
            }
            out.write(json.dumps({
                "id": rid,
                "text": ch,
                "metadata": md,
                "embedding": vecs[k],
                "embedding_type": "cloudflare-bge-m3"
            }) + "\n")
            k += 1
        # duplicates (metadata only)
        for idx, (orig_idx, orig_h) in dup_map.items():
            ch = chunks_all[idx]
            rid = sha1_text(f"{doc_hash}:{idx}:{orig_h}:dup")
            md = {
                "path": str(path),
                "chunk_index": idx,
                "total_chunks_in_doc": total,
                "file_hash": file_hash,
                "chunk_hash": sha1_text(ch),
                "is_duplicate": True,
                "duplicate_of_index": orig_idx,
                "duplicate_of_hash": orig_h,
                "skip_index": True,
                **meta_path
            }
            out.write(json.dumps({"id": rid, "text": ch, "metadata": md}) + "\n")

    return {"file": rel, "jsonl_tmp": str(jsonl_tmp),
            "chunks": len(uniq), "dups": len(dup_map), "jsonl_name": jsonl_name,
            "total_tokens": total_tokens}

# ====== Progress (resume) ======

"""Progress helpers provided by src.services.RAG.progress_store"""

# ====== Main ======

def signal_handler(signum, frame):
    log(f"[INTERRUPT] Received signal {signum}, gracefully shutting down...")
    sys.exit(0)

def main():
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)
    
    ap = argparse.ArgumentParser("Streamlined BGE-M3 pipeline (PaddleOCR)")
    ap.add_argument("-i", "--input-dir", required=True)
    ap.add_argument("--export-dir", default="OUTPUT_DATA/progress_report")
    ap.add_argument("--cache-dir", default="OUTPUT_DATA/cache")
    ap.add_argument("--workers", type=int, default=6)
    # If >0, sets OMP_NUM_THREADS; if 0, leave runtime defaults
    ap.add_argument("--omp-threads", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--timeout", type=int, default=1800, help="Timeout per file in seconds (default: 1800)")
    # Chroma toggles
    ap.add_argument("--with-chroma", dest="with_chroma", action="store_true", default=True)
    ap.add_argument("--no-chroma", dest="with_chroma", action="store_false")
    ap.add_argument("-c", "--collection", default="pdfs_bge_m3_cloudflare")
    ap.add_argument("-p", "--persist-dir", default="OUTPUT_DATA/chroma_db_data")
    ap.add_argument("--ocr-on-missing", choices=["fallback", "error", "skip"], default="fallback")
    ap.add_argument("--force-ocr", action="store_true")
    ap.add_argument("--max-pdfs", type=int, default=0)
    ap.add_argument("--embed-batch", type=int, default=int(os.getenv("CF_EMBED_MAX_BATCH", "96")))
    # Paddle controls
    ap.add_argument("--ocr-dpi", type=int, default=200)
    ap.add_argument("--ocr-lang", default=os.getenv("PADDLE_LANG", "en"),
                    help="OCR language (e.g., en, fr, de, ar, hi)")
    args = ap.parse_args()

    # Initialize descriptive logging
    try:
        setup_logging(level=os.getenv("LOG_LEVEL", "DEBUG"))
    except Exception:
        pass

    # Tame OpenMP noise/conflicts: only set when explicitly requested.
    if getattr(args, "omp_threads", 0) and args.omp_threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(args.omp_threads)
    else:
        # If limits are present from the shell, unset them for better defaults
        for _omp_var in ("OMP_THREAD_LIMIT", "KMP_DEVICE_THREAD_LIMIT", "KMP_TEAMS_THREAD_LIMIT"):
            if _omp_var in os.environ:
                os.environ.pop(_omp_var, None)

    # Cloudflare credentials (STRICT: no defaults, no hardcoding)
    acct = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
    tok = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
    if not acct or not tok:
        log("ERROR: Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN")
        sys.exit(2)

    root = Path(args.input_dir).resolve()
    export_dir = Path(args.export_dir).resolve(); export_dir.mkdir(parents=True, exist_ok=True)
    export_tmp = export_dir / "_tmp"; export_tmp.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir).resolve(); cache_dir.mkdir(parents=True, exist_ok=True)
    persist_dir = Path(args.persist_dir).resolve(); persist_dir.mkdir(parents=True, exist_ok=True)
    billing_file = persist_dir / "billing_state.json"
    seen_index_path = persist_dir / "seen_files.json"
    progress_path = export_dir / "progress_state.json"
    billing = Billing(Path(billing_file))

    # discover PDFs
    pdfs: List[Path] = []
    ignores = {".git", "node_modules", "__pycache__", ".venv", ".idea", ".vscode", "build", "dist"}
    for d, dirnames, files in os.walk(root):
        dirnames[:] = [x for x in dirnames if x not in ignores and not x.startswith(".")]
        for f in files:
            # Skip hidden and macOS resource-fork files (e.g., '._file.pdf')
            if f.startswith('.') or f.startswith('._'):
                continue
            if f.lower().endswith(".pdf"):
                pdfs.append(Path(d) / f)
    pdfs.sort()
    if args.max_pdfs > 0:
        pdfs = pdfs[:args.max_pdfs]
    log(f"Found {len(pdfs)} PDFs")

    # Chroma
    collection = None; client = None
    if args.with_chroma:
        client = chroma_client(str(persist_dir))
        collection = client.get_or_create_collection(name=args.collection, metadata={"hnsw:space": "cosine"})
        log(f"Chroma collection: {args.collection}")

    # progress & seen duplicates
    prog = load_progress(progress_path)
    files_state = prog.setdefault("files", {})
    try:
        seen = json.loads(seen_index_path.read_text(encoding="utf-8")) if seen_index_path.exists() else {}
    except Exception:
        seen = {}

    tasks: List[Path] = []

    for fp in pdfs:
        file_key = str(fp)
        st = fp.stat()
        fh = sha256_file(fp)[:16]
        # Initialize file in progress state
        if file_key not in files_state:
            files_state[file_key] = {
                "status": "pending",
                "file_size": st.st_size,
                "file_mtime": int(st.st_mtime),
                "discovered_at": now_iso(),
            }
        else:
            # Fast-skip previously completed files if unchanged and already upserted
            try:
                if should_skip(files_state[file_key], st.st_size, int(st.st_mtime)):
                    files_state[file_key]["status"] = "skipped"
                    files_state[file_key]["reason"] = "already_processed"
                    files_state[file_key]["finished_at"] = now_iso()
                    continue
            except Exception:
                pass
        # full-file duplicate skip
        if fh in seen and seen[fh] != file_key:
            files_state[file_key]["status"] = "skipped"
            files_state[file_key]["reason"] = "file_duplicate"
            files_state[file_key]["duplicate_of"] = seen[fh]
            files_state[file_key]["finished_at"] = now_iso()
            continue
        seen[fh] = file_key
        try:
            seen_index_path.write_text(json.dumps(seen, indent=2), encoding="utf-8")
        except Exception as e:
            log(f"[WARN] Failed to update seen files index: {e}")
        tasks.append(fp)

    save_progress(progress_path, prog)
    log(f"Queued {len(tasks)} files")

    # Pre-warm OCR model outside the per-file timeout window.
    # This triggers model download/initialization once to avoid first-file delays.
    if PADDLE_AVAILABLE:
        try:
            log(f"[WARMUP] Preloading PaddleOCR models (lang={args.ocr_lang})")
            from src.services.RAG.ocr_engine import get_paddle_ocr as _warmup_get  # local import to avoid name clash
            _ = _warmup_get(args.ocr_lang)
        except Exception as e:
            log(f"[WARN] OCR warmup failed: {e}")

    def archive_tmp(tmp: Path) -> Path:
        final = export_dir / tmp.name
        if final.exists():
            final.unlink()
        try:
            safe_file_replace(tmp, final)
        except Exception as e:
            log(f"[WARN] Failed to archive {tmp.name}, using fallback copy: {e}")
            final.write_bytes(tmp.read_bytes())
            try:
                tmp.unlink()
            except Exception:
                pass
        return final

    processed = 0
    if args.workers == 1:
        for fp in tasks:
            files_state[str(fp)]["status"] = "in_progress"
            files_state[str(fp)]["started_at"] = now_iso()
            save_progress(progress_path, prog)

            try:
                log(f"[PROCESS] Starting {fp.name} with {args.timeout}s timeout")
                start_time = time.time()
                # Run in a separate process to honor per-file timeout
                from concurrent.futures import ProcessPoolExecutor as _PPE
                ex1 = _PPE(max_workers=1)
                fut = ex1.submit(
                    process_one,
                    str(fp), str(root), str(export_tmp), str(cache_dir),
                    acct, tok, str(billing_file), args.embed_batch,
                    args.force_ocr, args.ocr_on_missing, args.ocr_dpi, args.ocr_lang,
                )
                try:
                    res = fut.result(timeout=max(1, int(args.timeout)))
                except TimeoutError:
                    # Best-effort cancel; on Windows the worker may continue in background
                    ex1.shutdown(wait=False, cancel_futures=True)
                    elapsed = time.time() - start_time
                    log(f"[ERROR] Timeout after {elapsed:.1f}s on {fp.name}")
                    res = {"error": "timeout", "file": str(fp)}
                else:
                    ex1.shutdown(wait=True, cancel_futures=False)
                    elapsed = time.time() - start_time
                    log(f"[PROCESS] Completed {fp.name} in {elapsed:.1f}s")
            except KeyboardInterrupt:
                log(f"[INTERRUPT] Processing interrupted for {fp.name}")
                sys.exit(0)
            except Exception as e:
                elapsed = time.time() - start_time if 'start_time' in locals() else 0
                log(f"[ERROR] Exception in {fp.name} after {elapsed:.1f}s: {e}")
                res = {"error": f"exception: {e}", "file": str(fp)}

            processed += 1

            if res.get("error"):
                files_state[str(fp)]["status"] = "failed"
                files_state[str(fp)]["error"] = res["error"]
                files_state[str(fp)]["finished_at"] = now_iso()
                save_progress(progress_path, prog)
                log(f"[FAIL] {fp.name}: {res['error']}")
                continue

            if res.get("skip"):
                files_state[str(fp)].update({"status": "skipped", "reason": res.get("reason", "unknown"),
                                             "finished_at": now_iso()})
                save_progress(progress_path, prog)
                log(f"[SKIP] {fp.name}: {res.get('reason')}")
                continue

            if "total_tokens" in res:
                ftoks, fcost = billing.add(res["file"], res["total_tokens"])
                log(f"[Billing] {Path(res['file']).name}: total tokens={ftoks:,} cost=${fcost:.6f}")

            jsonl_tmp = Path(res["jsonl_tmp"])  # temp path
            jsonl_final = archive_tmp(jsonl_tmp)

            chroma_done = False
            if args.with_chroma and collection:
                try:
                    # Use smaller batch size to reduce ChromaDB pressure
                    added = chroma_upsert_jsonl(jsonl_final, collection, client, batch=64)
                    chroma_done = added > 0
                    log(f"[Chroma] {fp.name}: +{added} vectors")
                except Exception as e:
                    log(f"[ERROR] ChromaDB failed for {fp.name}: {e}")
                    chroma_done = False

            files_state[str(fp)].update({
                "status": "completed",
                "jsonl_name": jsonl_final.name,
                "jsonl_archived": True,
                "chroma_upserted": chroma_done,
                "chunks": res.get("chunks", 0),
                "duplicates": res.get("dups", 0),
                "finished_at": now_iso(),
            })
            save_progress(progress_path, prog)
    else:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_worker_init_for_ocr,
                                 initargs=(args.ocr_lang,)) as ex:
            fut_map = {
                ex.submit(
                    process_one, str(fp), str(root), str(export_tmp), str(cache_dir),
                    acct, tok, str(billing_file), args.embed_batch,
                    args.force_ocr, args.ocr_on_missing, args.ocr_dpi, args.ocr_lang,
                ): fp for fp in tasks
            }
            for fut in as_completed(fut_map):
                fp = fut_map[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    res = {"error": f"exception: {e}", "file": str(fp)}
                processed += 1
                if res.get("error"):
                    files_state[str(fp)] = {
                        **files_state.get(str(fp), {}),
                        "status": "failed",
                        "error": res["error"],
                        "finished_at": now_iso(),
                    }
                    save_progress(progress_path, prog)
                    log(f"[FAIL] {fp.name}: {res['error']}")
                    continue
                if res.get("skip"):
                    files_state[str(fp)] = {
                        **files_state.get(str(fp), {}),
                        "status": "skipped",
                        "reason": res.get("reason", "unknown"),
                        "finished_at": now_iso(),
                    }
                    save_progress(progress_path, prog)
                    log(f"[SKIP] {fp.name}: {res.get('reason')}")
                    continue

                if "total_tokens" in res:
                    ftoks, fcost = billing.add(res["file"], res["total_tokens"])
                    log(f"[Billing] {Path(res['file']).name}: total tokens={ftoks:,} cost=${fcost:.6f}")

                jsonl_final = archive_tmp(Path(res["jsonl_tmp"]))
                chroma_done = False
                if args.with_chroma and collection:
                    try:
                        # Use smaller batch size to reduce ChromaDB pressure
                        added = chroma_upsert_jsonl(jsonl_final, collection, client, batch=64)
                        chroma_done = added > 0
                        log(f"[Chroma] {fp.name}: +{added} vectors")
                    except Exception as e:
                        log(f"[ERROR] ChromaDB failed for {fp.name}: {e}")
                        chroma_done = False

                base = files_state.get(str(fp), {})
                base.update({
                    "status": "completed", "jsonl_name": jsonl_final.name,
                    "jsonl_archived": True, "chroma_upserted": chroma_done,
                    "chunks": res.get("chunks", 0), "duplicates": res.get("dups", 0),
                    "finished_at": now_iso(),
                })
                files_state[str(fp)] = base
                save_progress(progress_path, prog)

    log(f"Done. Processed this run: {processed}")

if __name__ == "__main__":
    main()
