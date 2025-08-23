#!/usr/bin/env python3
"""
Streamlined: PDF -> (auto OCR w/ PaddleOCR) -> chunk -> dedupe -> BGE-M3 (Cloudflare)
-> per-file JSONL -> immediate Chroma upsert -> real-time billing -> resume.

Env:
  CLOUDFLARE_ACCOUNT_ID          (required)
  CLOUDFLARE_API_TOKEN           (required)
  CF_PRICE_PER_M_TOKENS=0.012    # USD per 1M input tokens (BGE-M3 input price)
  CF_EMBED_MAX_BATCH=96
  OMP_NUM_THREADS=4
  BILLING_ENABLED=1
"""

from __future__ import annotations
import os, re, sys, json, time, hashlib, argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

# progress tracker you already have
from src.utils.progress_tracker import ProgressTracker, ProcessingStatus

import fitz  # PyMuPDF
from PIL import Image
import requests
import chromadb
from chromadb.config import Settings

# Optional OpenCV for image handling (Paddle likes numpy arrays)
try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except Exception:
    OPENCV_AVAILABLE = False

# PaddleOCR
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except Exception:
    PADDLE_AVAILABLE = False

# ====== Small helpers ======

def log(msg: str): print(msg, flush=True)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

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

class Billing:
    def __init__(self, file: Path) -> None:
        self.file = file
        self.enabled = os.getenv("BILLING_ENABLED", "1") == "1"
        try:
            self.price_per_m = float(os.getenv("CF_PRICE_PER_M_TOKENS", "0.012"))
        except Exception:
            self.price_per_m = 0.012
        self.state = {
            "model": "@cf/baai/bge-m3",
            "price_per_m": self.price_per_m,
            "totals": {"tokens": 0, "cost": 0.0, "batches": 0, "files": 0},
            "files": {}
        }
        if self.file.exists():
            try:
                self.state = json.loads(self.file.read_text(encoding="utf-8"))
            except Exception:
                pass

    def _save(self):
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        tmp.replace(self.file)

    def add(self, src: str, new_tokens: int) -> Tuple[int, float]:
        if not self.enabled:
            return (0, 0.0)
        rec = self.state["files"].setdefault(
            src, {"tokens": 0, "cost": 0.0, "batches": 0, "last": now_iso()}
        )
        rec["tokens"] += new_tokens
        rec["batches"] += 1
        inc_cost = (new_tokens / 1_000_000.0) * self.price_per_m
        rec["cost"] += inc_cost
        rec["last"] = now_iso()

        t = self.state["totals"]
        t["tokens"] += new_tokens
        t["cost"] += inc_cost
        t["batches"] += 1
        t["files"] = len(self.state["files"])
        self._save()
        return rec["tokens"], rec["cost"]

# ====== Cloudflare BGE-M3 client ======

class CFEmbeddings:
    def __init__(self, account_id: str, api_token: str, batch_max: int, billing: Billing):
        self.url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/baai/bge-m3"
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {api_token}"})
        self.batch_max = max(1, min(100, batch_max))
        self.billing = billing
        self.counter = TokenCounter()

    def embed(self, texts: List[str], src_file: str, batch_size: int) -> List[List[float]]:
        bsz = min(batch_size, self.batch_max)
        out: List[List[float]] = []
        for i in range(0, len(texts), bsz):
            sub = texts[i:i+bsz]
            payload = {"text": sub, "truncate_inputs": True}
            r = self.s.post(self.url, json=payload, timeout=90)
            r.raise_for_status()
            # Workers AI often returns {"result":{"data":[[...], ...]}}
            data = r.json().get("result", {}).get("data")
            if not isinstance(data, list):
                raise RuntimeError(f"Bad embedding response: {r.text[:200]}")
            out.extend(data)

            tokens = self.counter.count_batch(sub)
            ftoks, fcost = self.billing.add(src_file, tokens)
            log(f"[Billing] {Path(src_file).name}: +{tokens} tokens | file tokens={ftoks:,} cost=${fcost:.6f}")
        return out

# ====== ChromaDB ======

def chroma_client(persist_dir: str):
    try:
        return chromadb.PersistentClient(path=str(Path(persist_dir)))
    except Exception:
        return chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=str(Path(persist_dir))))

def chroma_upsert_jsonl(jsonl_path: Path, collection, client, batch: int = 256) -> int:
    ids, docs, metas, embs = [], [], [], []
    n_added = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            md = rec.get("metadata", {})
            if md.get("is_duplicate") or md.get("skip_index"):
                continue
            ids.append(rec["id"])
            docs.append(rec["text"])
            metas.append(_sanitize_meta(md))
            embs.append(rec["embedding"])
            if len(ids) >= batch:
                collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
                n_added += len(ids)
                ids, docs, metas, embs = [], [], [], []
    if ids:
        collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
        n_added += len(ids)
    try:
        client.persist()
    except Exception:
        pass
    return n_added

def _sanitize_meta(d: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = [x if isinstance(x, (str, int, float, bool)) or x is None else str(x) for x in v]
        else:
            out[k] = json.dumps(v, ensure_ascii=False)
    return out

# ====== Path metadata (department/level/semester/group_key) ======

SEM = {
    "1": "1", "2": "2", "FIRST": "1", "SECOND": "2",
    "SEM1": "1", "SEM2": "2", "SEMESTER1": "1", "SEMESTER2": "2",
}

def parse_path_meta(path: Path) -> Dict[str, str]:
    parts = path.parts
    filename = parts[-1] if len(parts) >= 1 else ""
    course_folder = parts[-2] if len(parts) >= 2 else ""
    semester_raw = parts[-3] if len(parts) >= 3 else ""
    level_raw = parts[-4] if len(parts) >= 4 else ""
    dept = parts[-5] if len(parts) >= 5 else ""

    level = re.sub(r"(?i)[^0-9]|level", "", level_raw).strip()
    if not re.fullmatch(r"[1-5]00", level): level = ""
    sem = SEM.get(semester_raw.strip().upper().replace(" ", ""), "")

    m = re.search(r"([A-Za-z]{2,})\s*[-_ ]*\s*(\d{2,3})", course_folder)
    code, num = (m.group(1).upper(), m.group(2)) if m else ("", "")
    if not code or not num:
        m2 = re.search(r"([A-Za-z]{2,})\s*[-_ ]*\s*(\d{2,3})", Path(filename).stem)
        if m2: code, num = m2.group(1).upper(), m2.group(2)

    if not level and num and len(num) >= 3: level = num[0] + "00"

    cf_up = course_folder.upper()
    fn_up = filename.upper()
    cat = "PQ" if (cf_up in {"PQ", "PQS", "PASTQUESTIONS"} or "PQ" in fn_up or "PAST QUESTION" in fn_up or "PAST QUESTIONS" in fn_up) else ("GENERAL" if cf_up == "GENERAL" else "")

    group_key = (f"{dept}-{code}-{num}" if (dept and code and num)
                 else f"{code}-{num}" if (code and num)
                 else dept or code or "MISC")

    return {
        "DEPARTMENT": dept, "LEVEL": level, "SEMESTER": sem, "CATEGORY": cat,
        "COURSE_FOLDER": course_folder, "COURSE_CODE": code, "COURSE_NUMBER": num,
        "FILENAME": filename, "STEM": Path(filename).stem, "GROUP_KEY": group_key,
    }

# ====== OCR decision ======

def need_ocr(doc: fitz.Document, sample_pages: int = 8, min_chars_per_page: int = 200) -> bool:
    n = min(sample_pages, len(doc))
    if n == 0: return True
    low = 0
    for i in range(n):
        txt = doc[i].get_text("text")
        if len(txt) < min_chars_per_page: low += 1
    return (low / max(1, n)) >= 0.6

# ====== PaddleOCR (per-process singleton) ======

_PADDLE: Optional[PaddleOCR] = None

def get_paddle_ocr(lang: str = "en") -> Optional[PaddleOCR]:
    """Create/return a per-process PaddleOCR instance."""
    global _PADDLE
    if not PADDLE_AVAILABLE:
        return None
    if _PADDLE is None:
        # fast mode, angle_cls on, use rec+det; adjust det_db_box_thresh if needed
        _PADDLE = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            show_log=False
        )
    return _PADDLE

def _pixmap_to_numpy(pix: fitz.Pixmap) -> np.ndarray:
    """Convert PyMuPDF pixmap to OpenCV BGR numpy array."""
    if not OPENCV_AVAILABLE:
        # fall back: PIL then numpy RGB
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return np.array(img)  # RGB
    # pix.samples is RGB byte string
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    # Paddle works fine with RGB; OpenCV ops expect BGR. We won't do color ops; keep RGB.
    return arr

def _sort_ocr_lines(result: List[Any]) -> List[Tuple[str, float, Tuple[float, float]]]:
    """Flatten Paddle result and sort roughly by reading order (top->bottom, left->right)."""
    lines = []
    for line in result:
        # line: [ [ [x1,y1],...[x4,y4] ], (text, conf) ]
        try:
            box, (text, conf) = line
            xs = [p[0] for p in box]; ys = [p[1] for p in box]
            cx = sum(xs)/4.0; cy = sum(ys)/4.0
            lines.append((text, float(conf), (cx, cy)))
        except Exception:
            continue
    # sort by y then x
    lines.sort(key=lambda t: (round(t[2][1]/16.0), round(t[2][0]/16.0)))
    return lines

def ocr_page_with_paddle(page: fitz.Page, dpi: int = 300, lang: str = "en") -> str:
    """Render a page and run PaddleOCR; return concatenated text."""
    ocr = get_paddle_ocr(lang=lang)
    if ocr is None:
        return ""  # caller will fallback to text layer
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = _pixmap_to_numpy(pix)  # RGB (ok for Paddle)
    # Paddle accepts numpy arrays in RGB
    res = ocr.ocr(img, cls=True)
    # res can be nested per block; flatten
    flat = []
    for blk in res:
        if isinstance(blk, list):
            for item in blk:
                flat.append(item)
        else:
            flat.append(blk)
    lines = _sort_ocr_lines(flat)
    texts = [t for (t, conf, _) in lines if t and conf >= 0.35]
    return "\n".join(texts)

# ====== Text extract (auto OCR with Paddle) ======

def extract_text(path: Path, cache_dir: Path, force_ocr: bool, ocr_policy: str,
                 ocr_dpi: int = 300, ocr_lang: str = "en") -> str:
    # cache key includes file hash + OCR flag + lang + dpi
    fhash = sha256_file(path)[:16]
    key = f"{fhash}.{'ocr' if force_ocr else 'txt'}.paddle.{ocr_lang}.{ocr_dpi}.cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cpath = cache_dir / key
    if cpath.exists():
        txt = cpath.read_text(encoding="utf-8", errors="ignore")
        log(f"[CACHE] {path.name} -> hit")
        return txt

    log(f"[CACHE] {path.name} -> miss")
    with fitz.open(path) as doc:
        do_ocr = force_ocr or need_ocr(doc)
        if do_ocr and not PADDLE_AVAILABLE:
            if ocr_policy == "error":
                raise RuntimeError("paddle_ocr_missing")
            elif ocr_policy == "skip":
                return ""
            else:
                log("[WARN] OCR needed but PaddleOCR unavailable; falling back to text layer.")
                do_ocr = False

        if do_ocr:
            texts = []
            for i in range(len(doc)):
                if i and i % 10 == 0:
                    log(f"[OCR/Paddle] {path.name}: {i}/{len(doc)}")
                page_text = ocr_page_with_paddle(doc[i], dpi=ocr_dpi, lang=ocr_lang)
                texts.append(page_text)
            text = "\n".join(texts)
        else:
            texts = []
            for i in range(len(doc)):
                if i and i % 100 == 0:
                    log(f"[TXT] {path.name}: {i}/{len(doc)}")
                texts.append(doc[i].get_text("text"))
            text = "\n".join(texts)

    cpath.write_text(text, encoding="utf-8")
    return text

# ====== Chunking & dedupe ======

def split_paragraphs(text: str) -> List[str]:
    text = re.sub(r"\r\n?", "\n", text)
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paras if paras else ([text.strip()] if text.strip() else [])

def merge_paras(paras: List[str], min_chars: int, max_chars: int) -> List[str]:
    out, buf = [], ""
    for p in paras:
        if not buf: buf = p; continue
        if len(buf) < min_chars or (len(buf) + 2 + len(p) <= max_chars):
            buf = f"{buf}\n\n{p}"
        else:
            out.append(buf); buf = p
    if buf: out.append(buf)
    return out

def chunk(text: str, min_chars=200, max_chars=1600, overlap=80) -> List[str]:
    base = merge_paras(split_paragraphs(text), min_chars, max_chars)
    if overlap <= 0 or len(base) <= 1: return base
    out = [base[0]]
    for i in range(1, len(base)):
        tail = base[i-1][-overlap:]
        sp = tail.find(" ")
        if sp > 0: tail = tail[sp+1:]
        out.append(f"{tail} {base[i]}")
    return out

def dedupe(chunks: List[str]) -> Tuple[List[str], Dict[int, Tuple[int, str]]]:
    seen: Dict[str, int] = {}
    keep, dup = [], {}
    for i, c in enumerate(chunks):
        h = sha1_text(c)
        if h in seen:
            dup[i] = (seen[h], h)
        else:
            seen[h] = len(keep)
            keep.append(c)
    return keep, dup

# ====== Worker ======

def process_one(pdf_path: str, root: str, export_tmp: str,
                cache_dir: str, cf_acct: str, cf_token: str,
                billing_file: str, embed_batch: int,
                force_ocr: bool, ocr_policy: str,
                ocr_dpi: int, ocr_lang: str) -> Dict[str, Any]:

    # per-process Paddle init (lazy)
    if PADDLE_AVAILABLE:
        get_paddle_ocr(lang=ocr_lang)

    path = Path(pdf_path)
    rel = str(Path(pdf_path).resolve())
    meta_path = parse_path_meta(path)
    text = extract_text(path, Path(cache_dir), force_ocr, ocr_policy, ocr_dpi=ocr_dpi, ocr_lang=ocr_lang)
    if not text.strip():
        return {"file": rel, "skip": True, "reason": "empty_text"}

    chunks_all = chunk(text)
    uniq, dup_map = dedupe(chunks_all)
    if not uniq:
        return {"file": rel, "skip": True, "reason": "no_chunks"}

    # Cloudflare embeddings
    billing = Billing(Path(billing_file))
    cf = CFEmbeddings(cf_acct, cf_token,
                      int(os.getenv("CF_EMBED_MAX_BATCH", "96")), billing)
    vecs = cf.embed(uniq, src_file=rel, batch_size=embed_batch)
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
            if idx in dup_map: continue
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
            out.write(json.dumps({"id": rid, "text": ch, "metadata": md,
                                  "embedding": vecs[k], "embedding_type": "cloudflare-bge-m3"}) + "\n")
            k += 1
        # duplicates (metadata only)
        for idx, (orig_idx, orig_h) in dup_map.items():
            ch = chunks_all[idx]
            rid = sha1_text(f"{doc_hash}:{idx}:{orig_h}:dup")
            md = {"path": str(path), "chunk_index": idx, "total_chunks_in_doc": total,
                  "file_hash": file_hash, "chunk_hash": sha1_text(ch),
                  "is_duplicate": True, "duplicate_of_index": orig_idx,
                  "duplicate_of_hash": orig_h, "skip_index": True, **meta_path}
            out.write(json.dumps({"id": rid, "text": ch, "metadata": md}) + "\n")

    return {"file": rel, "jsonl_tmp": str(jsonl_tmp),
            "chunks": len(uniq), "dups": len(dup_map), "jsonl_name": jsonl_name}

# ====== Progress (resume) ======

def load_progress(p: Path) -> Dict[str, Any]:
    if p.exists():
        try: return json.loads(p.read_text(encoding="utf-8"))
        except Exception: pass
    return {"version": "simple-1", "files": {}}

def save_progress(p: Path, st: Dict[str, Any]) -> None:
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=2), encoding="utf-8")
    tmp.replace(p)

def should_skip(rec: Dict[str, Any], cur_size: int, cur_mtime: int) -> bool:
    return (rec.get("status") == "completed" and
            rec.get("file_size") == cur_size and
            rec.get("file_mtime") == cur_mtime and
            rec.get("jsonl_archived") and
            rec.get("chroma_upserted"))

# ====== Main ======

def main():
    ap = argparse.ArgumentParser("Streamlined BGE-M3 pipeline (PaddleOCR)")
    ap.add_argument("-i", "--input-dir", required=True)
    ap.add_argument("--export-dir", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--omp-threads", type=int, default=int(4))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--with-chroma", action="store_true", default=True)
    ap.add_argument("-c", "--collection", default="pdfs_bge_m3_cloudflare")
    ap.add_argument("-p", "--persist-dir", required=True)
    ap.add_argument("--ocr-on-missing", choices=["fallback","error","skip"], default="fallback")
    ap.add_argument("--force-ocr", action="store_true")
    ap.add_argument("--max-pdfs", type=int, default=0)
    ap.add_argument("--embed-batch", type=int, default=int(os.getenv("CF_EMBED_MAX_BATCH","96")))
    # Paddle controls
    ap.add_argument("--ocr-dpi", type=int, default=300)
    ap.add_argument("--ocr-lang", default=os.getenv("PADDLE_LANG","en"),
                    help="PaddleOCR language code (e.g., en, ch, fr, de, ar, te, ta, hi, etc.)")
    args = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.omp_threads)
    os.environ["OMP_THREAD_LIMIT"] = str(args.omp_threads)

    # Cloudflare credentials (read from env; you can hardcode if you want)
    acct = os.getenv("CLOUDFLARE_ACCOUNT_ID", "c1719c3cf4696ae260e6a5f57b1f3100").strip()
    tok  = os.getenv("CLOUDFLARE_API_TOKEN", "U7hBTssgt-8DAi1cyr3GnwihAVKqLUa37Su2q_-e").strip()
    if not acct or not tok:
        log("ERROR: Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN")
        sys.exit(2)

    root = Path(args.input_dir).resolve()
    export_dir = Path(args.export_dir).resolve()
    export_dir.mkdir(parents=True, exist_ok=True)
    export_tmp = export_dir / "_tmp"; export_tmp.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir).resolve(); cache_dir.mkdir(parents=True, exist_ok=True)
    persist_dir = Path(args.persist_dir).resolve(); persist_dir.mkdir(parents=True, exist_ok=True)
    billing_file = persist_dir / "billing_state.json"
    seen_index_path = persist_dir / "seen_files.json"
    progress_path = export_dir / "progress_state.json"

    # discover PDFs
    pdfs: List[Path] = []
    ignores = {".git","node_modules","__pycache__", ".venv",".idea",".vscode","build","dist"}
    for d, dirnames, files in os.walk(root):
        dirnames[:] = [x for x in dirnames if x not in ignores and not x.startswith(".")]
        for f in files:
            if f.lower().endswith(".pdf"):
                pdfs.append(Path(d)/f)
    pdfs.sort()
    if args.max_pdfs > 0: pdfs = pdfs[:args.max_pdfs]
    log(f"Found {len(pdfs)} PDFs")

    # Chroma
    collection = None; client = None
    if args.with_chroma:
        client = chroma_client(str(persist_dir))
        collection = client.get_or_create_collection(name=args.collection, metadata={"hnsw:space":"cosine"})
        log(f"Chroma collection: {args.collection}")

    # progress & seen duplicates
    prog = load_progress(progress_path)
    files_state = prog.setdefault("files", {})
    try:
        seen = json.loads(seen_index_path.read_text(encoding="utf-8")) if seen_index_path.exists() else {}
    except Exception:
        seen = {}

    tasks = []
    ingest_only: List[Tuple[Path, Dict[str,Any]]] = []

    tracker = ProgressTracker(str(progress_path))
    if args.resume:
        existing_progress = tracker.load_progress()
        if existing_progress:
            tracker.progress = existing_progress
        else:
            tracker.initialize_session(len(pdfs), {})
    else:
        tracker.initialize_session(len(pdfs), {})

    for fp in pdfs:
        file_key = str(fp)
        st = fp.stat()
        fh = sha256_file(fp)[:16]
        # full-file duplicate skip
        if fh in seen and seen[fh] != file_key:
            tracker.update_file_status(file_key, ProcessingStatus.SKIPPED, metadata={"reason": "file_duplicate", "duplicate_of": seen[fh]})
            continue
        seen[fh] = file_key
        seen_index_path.write_text(json.dumps(seen, indent=2), encoding="utf-8")
        tracker.update_file_status(file_key, ProcessingStatus.PENDING)
        tasks.append(fp)

    # save early
    save_progress(progress_path, prog)
    log(f"Queued {len(tasks)} files")

    def archive_tmp(tmp: Path) -> Path:
        final = export_dir / tmp.name
        if final.exists(): final.unlink()
        try: tmp.replace(final)
        except Exception:
            final.write_bytes(tmp.read_bytes()); tmp.unlink(missing_ok=True)
        return final

    processed = 0
    if args.workers == 1:
        for fp in tasks:
            files_state[str(fp)]["status"] = "in_progress"
            files_state[str(fp)]["started_at"] = now_iso()
            save_progress(progress_path, prog)

            res = process_one(str(fp), str(root), str(export_tmp), str(cache_dir),
                              acct, tok, str(billing_file), args.embed_batch,
                              args.force_ocr, args.ocr_on_missing, args.ocr_dpi, args.ocr_lang)
            processed += 1

            if res.get("error"):
                files_state[str(fp)]["status"] = "failed"
                files_state[str(fp)]["error"]  = res["error"]
                files_state[str(fp)]["finished_at"] = now_iso()
                save_progress(progress_path, prog)
                log(f"[FAIL] {fp.name}: {res['error']}")
                continue

            if res.get("skip"):
                files_state[str(fp)].update({"status":"skipped","reason": res.get("reason","unknown"),
                                             "finished_at": now_iso()})
                save_progress(progress_path, prog)
                log(f"[SKIP] {fp.name}: {res.get('reason')}")
                continue

            jsonl_tmp = Path(res["jsonl_tmp"])
            jsonl_final = archive_tmp(jsonl_tmp)

            chroma_done = False
            if args.with_chroma and collection:
                added = chroma_upsert_jsonl(jsonl_final, collection, client, batch=256)
                chroma_done = added > 0
                log(f"[Chroma] {fp.name}: +{added} vectors")

            files_state[str(fp)].update({
                "status": "completed",
                "jsonl_name": jsonl_final.name,
                "jsonl_archived": True,
                "chroma_upserted": chroma_done,
                "chunks": res.get("chunks", 0),
                "duplicates": res.get("dups", 0),
                "finished_at": now_iso()
            })
            save_progress(progress_path, prog)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(process_one, str(fp), str(root), str(export_tmp), str(cache_dir),
                              acct, tok, str(billing_file), args.embed_batch,
                              args.force_ocr, args.ocr_on_missing, args.ocr_dpi, args.ocr_lang) for fp in tasks]
            for fut, fp in zip(as_completed(futs), tasks):
                res = fut.result()
                processed += 1
                if res.get("error"):
                    files_state[str(fp)] = {"status":"failed","error":res["error"],
                                            "finished_at": now_iso(), **files_state.get(str(fp),{})}
                    save_progress(progress_path, prog)
                    log(f"[FAIL] {fp.name}: {res['error']}")
                    continue
                if res.get("skip"):
                    files_state[str(fp)] = {"status":"skipped","reason":res.get("reason","unknown"),
                                            "finished_at": now_iso(), **files_state.get(str(fp),{})}
                    save_progress(progress_path, prog)
                    log(f"[SKIP] {fp.name}: {res.get('reason')}")
                    continue

                jsonl_final = archive_tmp(Path(res["jsonl_tmp"]))
                chroma_done = False
                if args.with_chroma and collection:
                    added = chroma_upsert_jsonl(jsonl_final, collection, client, batch=256)
                    chroma_done = added > 0
                    log(f"[Chroma] {fp.name}: +{added} vectors")

                base = files_state.get(str(fp), {})
                base.update({"status":"completed","jsonl_name": jsonl_final.name,
                             "jsonl_archived": True,"chroma_upserted": chroma_done,
                             "chunks": res.get("chunks",0),"duplicates": res.get("dups",0),
                             "finished_at": now_iso()})
                files_state[str(fp)] = base
                save_progress(progress_path, prog)

    # resume: (kept for parity; nothing to ingest here since we archive immediately)
    log(f"Done. Processed this run: {processed}")

if __name__ == "__main__":
    main()
