import os
import gc
from pathlib import Path
from typing import List, Optional

from src.services.RAG.log_utils import get_logger, snapshot
from src.services.RAG.cache_utils import sha256_file, _key_text, _key_ocr, try_read, write

log = get_logger("ocr")

# Optional imports
try:
    import fitz  # PyMuPDF
except Exception as e:  # pragma: no cover
    fitz = None  # type: ignore
    log.warning(f"PyMuPDF not available: {e}")

try:
    from PIL import Image
except Exception as e:  # pragma: no cover
    Image = None  # type: ignore
    log.warning(f"PIL not available: {e}")

try:
    import numpy as np
except Exception as e:  # pragma: no cover
    np = None  # type: ignore
    log.warning(f"NumPy not available: {e}")

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except Exception as e:  # pragma: no cover
    PaddleOCR = None  # type: ignore
    PADDLE_AVAILABLE = False
    log.warning(f"PaddleOCR not available: {e}")

# Optional EasyOCR (fallback)
try:
    import easyocr  # type: ignore
    EASYOCR_AVAILABLE = True
except Exception as e:  # pragma: no cover
    easyocr = None  # type: ignore
    EASYOCR_AVAILABLE = False
    log.warning(f"EasyOCR not available: {e}")

# Optional OpenCV
try:
    import cv2
    OPENCV_AVAILABLE = True
except Exception:
    cv2 = None  # type: ignore
    OPENCV_AVAILABLE = False

_PADDLE: Optional['PaddleOCR'] = None
_EASYOCR: Optional['easyocr.Reader'] = None  # type: ignore[name-defined]


def _pixmap_to_numpy(pix: 'fitz.Pixmap'):
    if not OPENCV_AVAILABLE or cv2 is None:
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return np.array(img)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    return arr


def get_paddle_ocr(lang: str = "en") -> Optional['PaddleOCR']:
    global _PADDLE
    if not PADDLE_AVAILABLE:
        return None
    if _PADDLE is None:
        local_model_dir = './paddle_models'
        # PaddleOCR 3.x constructor: do not pass det/rec/cls or GPU flags
        kwargs = {"lang": lang, "use_textline_orientation": True}
        if os.path.exists(local_model_dir):
            kwargs.update({
                "det_model_dir": f"{local_model_dir}/det",
                "rec_model_dir": f"{local_model_dir}/rec",
                "cls_model_dir": f"{local_model_dir}/cls",
            })
        try:
            _PADDLE = PaddleOCR(**kwargs)
            log.info(
                f"[OCR] PaddleOCR initialized (lang={lang}, textline=True, local_models={'det_model_dir' in kwargs})"
            )
        except Exception as e:
            log.warning(f"[OCR] Failed to initialize PaddleOCR: {e}")
            _PADDLE = None
    return _PADDLE


def get_easyocr(lang: str = "en") -> Optional['easyocr.Reader']:
    """Create or return a singleton EasyOCR reader for the given language.

    Uses CPU by default (gpu=False). Set env EASYOCR_GPU=1 to enable GPU if available.
    """
    global _EASYOCR
    if not EASYOCR_AVAILABLE:
        return None
    if _EASYOCR is None:
        try:
            gpu = os.getenv("EASYOCR_GPU", "0") == "1"
            # EasyOCR expects a list of languages; fall back to English on failure
            langs = [lang or "en"]
            _EASYOCR = easyocr.Reader(langs, gpu=gpu)  # type: ignore[name-defined]
            log.info(f"[OCR] EasyOCR initialized (lang={langs}, gpu={gpu})")
        except Exception as e:
            log.warning(f"[OCR] Failed to initialize EasyOCR: {e}")
            _EASYOCR = None
    return _EASYOCR


def _ocr_page(page: 'fitz.Page', dpi: int, lang: str) -> str:
    ocr = get_paddle_ocr(lang=lang)
    last_img = None
    ladder = [dpi, 240, 200, 150, 100, 72] if dpi >= 240 else [dpi, 150, 100, 72]
    MAX_OCR_BYTES = int(os.getenv("OCR_MAX_IMAGE_BYTES", str(64 * 1024 * 1024)))

    for attempt_dpi in ladder:
        try:
            zoom = attempt_dpi / 72.0
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            img = _pixmap_to_numpy(pix)
            last_img = img
            nbytes = getattr(img, "nbytes", img.size)
            if nbytes > MAX_OCR_BYTES:
                log.warning(f"[OCR] page-bytes={nbytes} > max={MAX_OCR_BYTES} @ {attempt_dpi} DPI: lowering")
                continue
            if ocr is not None:
                # PaddleOCR 3.x: call without det/rec/cls kwargs
                res = ocr.ocr(img)
                lines = []
                flat = []
                for blk in res:
                    if blk:
                        flat.extend(blk if isinstance(blk, list) else [blk])
                for line in flat:
                    try:
                        if line and len(line) >= 2:
                            box, (text, conf) = line
                            lines.append((text, float(conf)))
                    except Exception:
                        continue
                texts = [t for (t, conf) in lines if t and conf >= 0.35]
                if texts:
                    s = "\n".join(texts)
                    if attempt_dpi != dpi:
                        log.info(f"[OCR] succeeded at {attempt_dpi} DPI (requested={dpi}) snapshot='{snapshot(s)}'")
                    return s
        except Exception as e:
            if any(s in str(e).lower() for s in ("memory", "alloc")):
                log.warning(f"[OCR] OOM at {attempt_dpi} DPI; trying lower DPI")
                continue
            log.warning(f"[OCR] error at {attempt_dpi} DPI: {e}")
            break
        finally:
            # Proactively free large arrays/pixmaps before next attempt
            try:
                del img
            except Exception:
                pass
            try:
                del pix
            except Exception:
                pass
            gc.collect()

    # EasyOCR fallback
    reader = get_easyocr(lang=lang)
    if reader is not None:
        try:
            if last_img is None:
                pix = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
                last_img = _pixmap_to_numpy(pix)
            img_np = last_img
            # Convert to BGR if OpenCV is available (EasyOCR commonly uses BGR)
            if OPENCV_AVAILABLE and cv2 is not None:
                try:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                except Exception:
                    pass
            results = reader.readtext(img_np)
            lines = []
            for res in results:
                try:
                    box, text, conf = res  # type: ignore[misc]
                    # Compute a simple centroid for sorting
                    xs = [p[0] for p in box]; ys = [p[1] for p in box]
                    cx = sum(xs) / len(xs); cy = sum(ys) / len(ys)
                    lines.append((text, float(conf), (cx, cy)))
                except Exception:
                    continue
            # Sort by rows then columns (coarse grid)
            lines.sort(key=lambda t: (round(t[2][1] / 16.0), round(t[2][0] / 16.0)))
            texts = [t for (t, conf, _) in lines if t and conf >= 0.35]
            if texts:
                s = "\n".join(texts)
                log.info(f"[OCR] EasyOCR fallback used snapshot='{snapshot(s)}'")
                return s
        except Exception as e:
            log.warning(f"[OCR] EasyOCR fallback failed: {e}")
    return ""


def extract_text(pdf_path: Path, cache_dir: Path, force_ocr: bool, ocr_policy: str, ocr_dpi: int, ocr_lang: str) -> str:
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is required for extraction")

    fhash = sha256_file(pdf_path)[:16]

    # 1) Try text layer (and cache) if allowed
    if ocr_policy == "fallback" and not force_ocr:
        key = _key_text(fhash, ocr_lang)
        cached = try_read(cache_dir, key)
        if cached and cached.strip():
            log.info(f"[EXTRACT] text-layer cache hit file={pdf_path.name} chars={len(cached)} snapshot='{snapshot(cached)}'")
            return cached

        with fitz.open(pdf_path) as doc:
            texts: List[str] = []
            for i in range(len(doc)):
                if i and i % 50 == 0:
                    log.info(f"[TXT] {pdf_path.name}: {i}/{len(doc)} (text-layer)")
                t = doc[i].get_text("text")
                texts.append(t)
            full = "\n".join(texts)
        eff = len("".join(full.split()))
        log.info(f"[EXTRACT] text-layer file={pdf_path.name} chars={len(full)} eff={eff} snapshot='{snapshot(full)}'")
        if eff >= 50:
            write(cache_dir, key, full)
            return full
        else:
            log.warning(f"[EXTRACT] text-layer near-empty; will try OCR")

    # 2) OCR path (with cache)
    with fitz.open(pdf_path) as doc:
        need = force_ocr or _need_ocr(doc)
        if not need and ocr_policy != "fallback":
            texts = [doc[i].get_text("text") for i in range(len(doc))]
            return "\n".join(texts)

        key = _key_ocr(fhash, ocr_lang, ocr_dpi, ocr_policy)
        cached = try_read(cache_dir, key)
        if cached and cached.strip():
            log.info(f"[EXTRACT] OCR cache hit file={pdf_path.name} chars={len(cached)} snapshot='{snapshot(cached)}'")
            return cached

        pieces: List[str] = []
        total_pages = len(doc)
        for i in range(total_pages):
            # More visible progress for small docs; always log page 1
            if total_pages <= 20 or i % 10 == 0:
                log.info(f"[OCR] {pdf_path.name}: page {i+1}/{total_pages} (dpi={ocr_dpi})")
            s = _ocr_page(doc[i], dpi=ocr_dpi, lang=ocr_lang)
            pieces.append(s)
        full = "\n".join(pieces)
        eff = len("".join(full.split()))
        log.info(f"[EXTRACT] OCR file={pdf_path.name} chars={len(full)} eff={eff} snapshot='{snapshot(full)}'")
        if eff >= 50:
            write(cache_dir, key, full)
            return full
        else:
            log.warning(f"[EXTRACT] OCR near-empty; not caching file={pdf_path.name}")
            return ""


def _need_ocr(doc: 'fitz.Document', sample_pages: int = 8, min_chars_per_page: int = 200) -> bool:
    n = min(sample_pages, len(doc))
    if n == 0:
        return True
    low = 0
    for i in range(n):
        txt = doc[i].get_text("text")
        if len(txt) < min_chars_per_page:
            low += 1
    need = (low / max(1, n)) >= 0.6
    log.info(f"[OCR] need_ocr={need} sample={n} low_pages={low}/{n}")
    return need
