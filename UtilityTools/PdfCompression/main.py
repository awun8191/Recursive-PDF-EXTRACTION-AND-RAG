#!/usr/bin/env python3
"""
Batch PDF & PPTX compressor with smart defaults, parallelism, and safety rails.
- PDFs: Ghostscript (prefer) or pypdf fallback. Optional auto quality for image‑heavy docs.
- PPTX: recompress embedded images in ppt/media using Pillow.

Key upgrades vs. baseline:
- Parallel processing (configurable workers)
- Min file size threshold and min% reduction guard
- Optional backup of originals + metadata preservation
- Dry‑run mode
- JSON/CSV report
- Timeouts and better error surfacing for subprocesses
- More robust pdfimages parsing, Windows-friendly Ghostscript detection
- Graceful handling of encrypted PDFs and edge cases
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------- Logging ---------------------------------
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, handlers=[logging.StreamHandler()])
logger = logging.getLogger("compressor")

# -------------------------- Dependency Checks -------------------------

def which_gs() -> Optional[str]:
    """Return Ghostscript executable if available (cross-platform)."""
    candidates = [
        shutil.which("gs"),            # Linux/macOS
        shutil.which("gswin64c"),      # Windows 64-bit
        shutil.which("gswin32c"),      # Windows 32-bit
    ]
    return next((c for c in candidates if c), None)


def is_tool_available(name: str) -> bool:
    return shutil.which(name) is not None

GS_BIN = which_gs()
PDFIMAGES_AVAILABLE = is_tool_available("pdfimages")

try:
    from pypdf import PdfWriter, PdfReader
    PYPDF_AVAILABLE = True
except Exception:  # pragma: no cover
    PYPDF_AVAILABLE = False

try:
    from PIL import Image
    PILLOW_AVAILABLE = True
except Exception:  # pragma: no cover
    PILLOW_AVAILABLE = False

# ----------------------------- Config ---------------------------------

@dataclass
class PDFConfig:
    resolution: int = 120                 # DPI downsample target
    base_quality: int = 85                # JPEG quality for mainly text/vector PDFs
    image_heavy_quality: int = 75         # JPEG quality if image ratio exceeds threshold
    image_ratio_threshold: float = 0.50   # Consider PDF image-heavy if >= 50%
    method: str = "auto"                  # ghostscript|pypdf|auto
    timeout_s: int = 600                  # GS timeout per file


@dataclass
class PPTXConfig:
    image_quality: int = 80               # JPEG quality for recompressing images


@dataclass
class RunConfig:
    input_path: Path
    output_folder: Optional[Path] = None
    workers: int = max(1, (os.cpu_count() or 2) // 2)
    min_size_mb: float = 0.5              # skip tiny files
    min_reduction_pct: float = 1.0        # require at least X% reduction to keep compressed
    backup: bool = False                  # save *.bak beside original before replacing
    dry_run: bool = False                 # compute plan, don't write
    preserve_times: bool = True           # copy atime/mtime from original
    report_json: Optional[Path] = None
    report_csv: Optional[Path] = None


SUPPORTED_EXTS = {".pdf", ".pptx"}

# --------------------------- Utilities --------------------------------

def human(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} TB"


def safe_temp_path(target: Path) -> Path:
    return target.with_suffix(target.suffix + ".tmp")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def copy_stat(src: Path, dst: Path) -> None:
    try:
        shutil.copystat(src, dst)
    except Exception:
        pass


# ------------------------ PDF Content Diagnose ------------------------

def diagnose_pdf_image_ratio(input_path: Path) -> Tuple[float, Optional[str]]:
    """Return (image/size ratio, error) using `pdfimages -list` if available."""
    if not PDFIMAGES_AVAILABLE:
        return 0.0, "pdfimages not available"
    try:
        result = subprocess.run(
            ["pdfimages", "-list", str(input_path)],
            capture_output=True, text=True, check=True
        )
        # Skip header (first two lines). Size appears as e.g. "123K", "2.3M" etc.
        total = 0.0
        size_re = re.compile(r"\b([0-9]+(?:\.[0-9]+)?)([BKMG])\b", re.IGNORECASE)
        for line in result.stdout.splitlines()[2:]:
            m = size_re.search(line)
            if not m:
                continue
            val = float(m.group(1))
            unit = m.group(2).upper()
            if unit == "B":
                total += val
            elif unit == "K":
                total += val * 1024
            elif unit == "M":
                total += val * 1024 ** 2
            elif unit == "G":
                total += val * 1024 ** 3
        file_size = input_path.stat().st_size or 1
        return total / file_size, None
    except subprocess.CalledProcessError as e:
        return 0.0, f"pdfimages failed: rc={e.returncode}"
    except Exception as e:  # pragma: no cover
        return 0.0, str(e)


# ---------------------------- PDF Compress ----------------------------

def compress_pdf_with_gs(input_path: Path, output_path: Path, cfg: PDFConfig) -> bool:
    if not GS_BIN:
        return False

    # Choose quality based on image ratio
    ratio, _err = diagnose_pdf_image_ratio(input_path)
    quality = cfg.image_heavy_quality if ratio >= cfg.image_ratio_threshold else cfg.base_quality
    logger.info(f"[GS] {input_path.name}: image ratio≈{ratio:.0%} → quality={quality}, res={cfg.resolution}dpi")

    cmd = [
        GS_BIN,
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.4",
        "-dPDFSETTINGS=/default",
        "-dNOPAUSE",
        "-dQUIET",
        "-dBATCH",
        f"-dColorImageResolution={cfg.resolution}",
        f"-dGrayImageResolution={cfg.resolution}",
        f"-dMonoImageResolution={cfg.resolution}",
        "-dDownsampleColorImages=true",
        "-dDownsampleGrayImages=true",
        "-dDownsampleMonoImages=true",
        "-dColorImageDownsampleType=/Bicubic",
        "-dGrayImageDownsampleType=/Bicubic",
        f"-dJPEGQuality={quality}",
        f"-sOutputFile={str(output_path)}",
        str(input_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=cfg.timeout_s)
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"[GS] Timeout after {cfg.timeout_s}s: {input_path}")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"[GS] Failed {input_path}\nSTDERR: {e.stderr[:4000]}")
        return False


def compress_pdf_with_pypdf(input_path: Path, output_path: Path) -> bool:
    if not PYPDF_AVAILABLE:
        return False
    try:
        reader = PdfReader(str(input_path))
        if reader.is_encrypted:
            try:
                reader.decrypt("")  # try empty password
            except Exception:
                logger.warning(f"[pypdf] Encrypted PDF (skipping): {input_path.name}")
                return False
        writer = PdfWriter()
        for p in reader.pages:
            writer.add_page(p)
        writer.compress_content_streams()
        with open(output_path, "wb") as f:
            writer.write(f)
        return True
    except Exception as e:
        logger.error(f"[pypdf] Failed {input_path}: {e}")
        return False


def compress_pdf(input_path: Path, final_dest: Path, pdf_cfg: PDFConfig, min_reduction_pct: float, dry_run: bool, backup: bool, preserve_times: bool) -> Dict[str, Any]:
    orig_size = input_path.stat().st_size
    if orig_size < int(pdf_cfg.resolution) and False:
        pass  # placeholder (not used)

    tmp_out = safe_temp_path(final_dest)
    ensure_parent(tmp_out)

    method = pdf_cfg.method
    if method == "auto":
        method = "ghostscript" if GS_BIN else ("pypdf" if PYPDF_AVAILABLE else "none")

    ok = False
    if not dry_run:
        if method == "ghostscript":
            ok = compress_pdf_with_gs(input_path, tmp_out, pdf_cfg)
        elif method == "pypdf":
            ok = compress_pdf_with_pypdf(input_path, tmp_out)
        else:
            return {"success": False, "reason": "No available PDF compression backend"}
    else:
        ok = True  # pretend success to evaluate potential reduction
        # create a fake size estimate by assuming 20% reduction
        est_size = int(orig_size * 0.8)
        with open(tmp_out, "wb") as f:
            f.truncate(est_size)

    if not ok or not tmp_out.exists() or tmp_out.stat().st_size == 0:
        if tmp_out.exists():
            tmp_out.unlink(missing_ok=True)
        return {"success": False, "reason": "Compression failed or empty output"}

    new_size = tmp_out.stat().st_size
    reduction = (1 - new_size / max(1, orig_size)) * 100

    action = "kept_original"
    if reduction >= max(0.0, min_reduction_pct):
        action = "compressed"
        if not dry_run:
            if backup:
                bak = final_dest.with_suffix(final_dest.suffix + ".bak")
                try:
                    shutil.copy2(input_path, bak)
                except Exception:
                    logger.warning(f"Could not write backup: {bak}")
            shutil.move(str(tmp_out), str(final_dest))
            if preserve_times:
                copy_stat(input_path, final_dest)
        else:
            tmp_out.unlink(missing_ok=True)
    else:
        # no real improvement
        if not dry_run:
            tmp_out.unlink(missing_ok=True)
            if final_dest != input_path and not final_dest.exists():
                shutil.copy2(input_path, final_dest)
                if preserve_times:
                    copy_stat(input_path, final_dest)
        action = "copied_original" if final_dest != input_path else "kept_original"

    return {
        "success": True,
        "type": "pdf",
        "method": method,
        "action": action,
        "original_size": orig_size,
        "final_size": new_size if action == "compressed" else orig_size,
        "reduction_pct": reduction if action == "compressed" else 0.0,
    }

# --------------------------- PPTX Compress ----------------------------

def compress_pptx(input_path: Path, final_dest: Path, pptx_cfg: PPTXConfig, min_reduction_pct: float, dry_run: bool, backup: bool, preserve_times: bool) -> Dict[str, Any]:
    if not PILLOW_AVAILABLE:
        return {"success": False, "reason": "Pillow not installed", "type": "pptx"}

    orig_size = input_path.stat().st_size
    tmp_out = safe_temp_path(final_dest)
    ensure_parent(tmp_out)

    if dry_run:
        est_size = int(orig_size * 0.85)
        with open(tmp_out, "wb") as f:
            f.truncate(est_size)
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix=f"pptx_{input_path.stem}_"))
        try:
            with tempfile.TemporaryDirectory() as td:
                td_path = Path(td)
                # Unzip
                shutil.unpack_archive(str(input_path), str(td_path), format="zip")
                media = td_path / "ppt" / "media"
                if media.exists():
                    for img_path in media.iterdir():
                        if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif"} and img_path.is_file():
                            try:
                                with Image.open(img_path) as im:
                                    im = im.convert("RGB") if im.mode in ("RGBA", "P", "LA") else im
                                    im.save(img_path, optimize=True, quality=pptx_cfg.image_quality)
                            except Exception as e:
                                logger.debug(f"PPTX image skipped {img_path.name}: {e}")
                shutil.make_archive(str(tmp_out.with_suffix("")), "zip", str(td_path))
            # ensure .pptx extension
            if tmp_out.exists():
                tmp_out.unlink()
            tmp_zip = tmp_out.with_suffix("")
            tmp_zip.rename(tmp_out)
        except Exception as e:
            logger.error(f"PPTX compress failed {input_path.name}: {e}")
            return {"success": False, "reason": str(e), "type": "pptx"}

    if not tmp_out.exists() or tmp_out.stat().st_size == 0:
        tmp_out.unlink(missing_ok=True)
        return {"success": False, "reason": "Empty output", "type": "pptx"}

    new_size = tmp_out.stat().st_size
    reduction = (1 - new_size / max(1, orig_size)) * 100

    action = "kept_original"
    if reduction >= max(0.0, min_reduction_pct):
        action = "compressed"
        if not dry_run:
            if backup:
                bak = final_dest.with_suffix(final_dest.suffix + ".bak")
                try:
                    shutil.copy2(input_path, bak)
                except Exception:
                    logger.warning(f"Could not write backup: {bak}")
            shutil.move(str(tmp_out), str(final_dest))
            if preserve_times:
                copy_stat(input_path, final_dest)
        else:
            tmp_out.unlink(missing_ok=True)
    else:
        if not dry_run:
            tmp_out.unlink(missing_ok=True)
            if final_dest != input_path and not final_dest.exists():
                shutil.copy2(input_path, final_dest)
                if preserve_times:
                    copy_stat(input_path, final_dest)
        action = "copied_original" if final_dest != input_path else "kept_original"

    return {
        "success": True,
        "type": "pptx",
        "action": action,
        "original_size": orig_size,
        "final_size": new_size if action == "compressed" else orig_size,
        "reduction_pct": reduction if action == "compressed" else 0.0,
    }

# --------------------------- Batch Engine -----------------------------

def gather_files(inputs: List[Path]) -> List[Path]:
    files: List[Path] = []
    for p in inputs:
        if not p.exists():
            logger.warning(f"Path does not exist, skipping: {p}")
            continue
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            files.append(p)
        elif p.is_dir():
            for ext in SUPPORTED_EXTS:
                found = list(p.rglob(f"*{ext}"))
                files.extend(found)
    return sorted(set(files))


def process_one(path: Path, run_cfg: RunConfig, pdf_cfg: PDFConfig, pptx_cfg: PPTXConfig) -> Dict[str, Any]:
    try:
        ext = path.suffix.lower()
        if path.stat().st_size < run_cfg.min_size_mb * 1024 * 1024:
            return {"success": True, "action": "skipped_small", "original_size": path.stat().st_size, "final_size": path.stat().st_size, "type": ext[1:], "file": str(path)}

        out_path = (run_cfg.output_folder / path.name) if run_cfg.output_folder else path

        if ext == ".pdf":
            res = compress_pdf(path, out_path, pdf_cfg, run_cfg.min_reduction_pct, run_cfg.dry_run, run_cfg.backup, run_cfg.preserve_times)
        elif ext == ".pptx":
            res = compress_pptx(path, out_path, pptx_cfg, run_cfg.min_reduction_pct, run_cfg.dry_run, run_cfg.backup, run_cfg.preserve_times)
        else:
            return {"success": False, "reason": "unsupported", "file": str(path)}

        res.update({"file": str(path)})
        return res
    except Exception as e:
        return {"success": False, "reason": str(e), "file": str(path)}


def write_reports(results: List[Dict[str, Any]], run_cfg: RunConfig) -> None:
    if run_cfg.report_json:
        ensure_parent(run_cfg.report_json)
        with open(run_cfg.report_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    if run_cfg.report_csv:
        ensure_parent(run_cfg.report_csv)
        fields = sorted({k for r in results for k in r.keys()})
        with open(run_cfg.report_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in results:
                w.writerow(r)


# ------------------------------ CLI -----------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compress PDF and PPTX files (in place or to an output folder) with smart heuristics.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("input", type=str, help="File or folder to process (recurses into folders).")
    p.add_argument("--output", type=str, default=None, help="Optional output folder (mirror filenames). Default: in-place.")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)//2), help="Parallel workers (default: ~half cores).")
    p.add_argument("--min-size-mb", type=float, default=0.5, help="Skip files smaller than this (default: 0.5MB).")
    p.add_argument("--min-reduction", type=float, default=1.0, help="Only keep compressed file if reduction >= this percent.")
    p.add_argument("--backup", action="store_true", help="Write a .bak copy of original before replacing.")
    p.add_argument("--no-preserve-times", dest="preserve_times", action="store_false", help="Do not preserve file timestamps.")
    p.add_argument("--dry-run", action="store_true", help="Plan only; do not write outputs.")
    p.add_argument("--report-json", type=str, default=None, help="Write a JSON report to this path.")
    p.add_argument("--report-csv", type=str, default=None, help="Write a CSV report to this path.")

    # PDF options
    p.add_argument("--pdf-method", choices=["ghostscript", "pypdf", "auto"], default="auto", help="Backend for PDFs.")
    p.add_argument("--pdf-resolution", type=int, default=120, help="Downsample target DPI (GS).")
    p.add_argument("--pdf-quality-base", type=int, default=85, help="JPEG quality when not image-heavy (GS).")
    p.add_argument("--pdf-quality-image", type=int, default=75, help="JPEG quality when image-heavy (GS).")
    p.add_argument("--pdf-image-threshold", type=float, default=0.50, help="Image-heavy threshold ratio (0-1).")
    p.add_argument("--pdf-timeout", type=int, default=600, help="Per-file Ghostscript timeout (s).")

    # PPTX options
    p.add_argument("--pptx-quality", type=int, default=80, help="JPEG quality for PPTX images.")

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input path not found: {input_path}")
        return 2

    output_folder = Path(args.output).resolve() if args.output else None
    if output_folder and not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)

    # Dependency banner
    logger.info("--- Dependency Check ---")
    if not GS_BIN:
        logger.warning("Ghostscript not found; PDF compression will use pypdf if available.")
    if not PYPDF_AVAILABLE:
        logger.warning("pypdf not available; cannot fallback for PDFs.")
    if not PDFIMAGES_AVAILABLE:
        logger.warning("pdfimages not found; image-heavy detection disabled.")
    if not PILLOW_AVAILABLE:
        logger.warning("Pillow not found; PPTX compression disabled.")
    logger.info("------------------------")

    pdf_cfg = PDFConfig(
        resolution=args.pdf_resolution,
        base_quality=args.pdf_quality_base,
        image_heavy_quality=args.pdf_quality_image,
        image_ratio_threshold=args.pdf_image_threshold,
        method=args.pdf_method,
        timeout_s=args.pdf_timeout,
    )
    pptx_cfg = PPTXConfig(image_quality=args.pptx_quality)
    run_cfg = RunConfig(
        input_path=input_path,
        output_folder=output_folder,
        workers=max(1, args.workers),
        min_size_mb=max(0.0, args.min_size_mb),
        min_reduction_pct=max(0.0, args.min_reduction),
        backup=bool(args.backup),
        dry_run=bool(args.dry_run),
        preserve_times=bool(args.preserve_times),
        report_json=Path(args.report_json) if args.report_json else None,
        report_csv=Path(args.report_csv) if args.report_csv else None,
    )

    # Gather files
    targets = gather_files([input_path])
    targets = [t for t in targets if t.suffix.lower() in SUPPORTED_EXTS]
    logger.info(f"Found {len(targets)} file(s) to process.")

    results: List[Dict[str, Any]] = []
    if not targets:
        write_reports(results, run_cfg)
        logger.info("Nothing to do.")
        return 0

    # Parallel processing (Ghostscript is external, threads are fine for I/O)
    with ThreadPoolExecutor(max_workers=run_cfg.workers) as ex:
        fut2path = {ex.submit(process_one, p, run_cfg, pdf_cfg, pptx_cfg): p for p in targets}
        for fut in as_completed(fut2path):
            p = fut2path[fut]
            try:
                res = fut.result()
            except Exception as e:  # pragma: no cover
                res = {"success": False, "reason": str(e), "file": str(p)}
            results.append(res)
            # log succinct status per file
            if res.get("success"):
                action = res.get("action", "?")
                red = res.get("reduction_pct", 0.0)
                o = res.get("original_size", 0)
                f = res.get("final_size", 0)
                logger.info(f"{p.name}: {action} (orig {human(o)} → final {human(f)}; -{red:.1f}%)")
            else:
                logger.error(f"{p.name}: failed ({res.get('reason')})")

    # Summaries
    tot = len(results)
    compressed = sum(1 for r in results if r.get("action") == "compressed")
    copied = sum(1 for r in results if r.get("action") in {"copied_original", "kept_original"})
    skipped = sum(1 for r in results if r.get("action") == "skipped_small")
    failed = sum(1 for r in results if not r.get("success"))

    logger.info("\n--- Batch Complete ---")
    logger.info(f"Compressed: {compressed}")
    logger.info(f"Copied/Kept: {copied}")
    logger.info(f"Skipped small: {skipped}")
    logger.info(f"Failed: {failed}")
    logger.info("----------------------")

    # Reports
    write_reports(results, run_cfg)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
