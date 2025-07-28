import os
import re
import shutil
import logging
import subprocess
import zipfile
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any

# --- Dependency Checks ---
def is_tool_available(name: str) -> bool:
    """Check whether `name` is on PATH and marked as executable."""
    return shutil.which(name) is not None

try:
    from pypdf import PdfWriter, PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    from PIL import Image
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False

GHOSTSCRIPT_AVAILABLE = is_tool_available("gs")
PDFIMAGES_AVAILABLE = is_tool_available("pdfimages")

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class FileCompressor:
    """
    A file compression tool for both PDF and PPTX files.
    """

    def __init__(self):
        self._check_dependencies()
        # --- Compression Specifications ---
        self.gs_resolution = 80
        self.gs_base_quality = 90
        self.gs_image_heavy_quality = 90
        self.image_ratio_threshold = 0.5  # 50%
        self.pptx_image_quality = 85

    def _check_dependencies(self):
        """Checks for all required external and library dependencies."""
        logger.info("--- Dependency Check ---")
        if not PYPDF_AVAILABLE:
            logger.warning("pypdf not found. PDF compression via pypdf will be unavailable. (pip install pypdf)")
        if not GHOSTSCRIPT_AVAILABLE:
            logger.warning("Ghostscript ('gs') not found. PDF compression via Ghostscript will be unavailable.")
        if not PDFIMAGES_AVAILABLE:
            logger.warning("pdfimages not found. Smart PDF quality control will be disabled.")
        if not PILLOW_AVAILABLE:
            logger.warning("Pillow not found. PPTX compression will be unavailable. (pip install Pillow)")
        logger.info("------------------------")

    def _diagnose_pdf_content(self, input_path: Path) -> Dict[str, Any]:
        """Uses pdfimages to determine if a PDF is image-heavy."""
        if not PDFIMAGES_AVAILABLE:
            return {"image_to_file_ratio": 0.0, "error": "pdfimages not available"}
        try:
            result = subprocess.run(['pdfimages', '-list', str(input_path)], capture_output=True, text=True, check=True)
            total_image_size_bytes = 0
            size_pattern = re.compile(r'\s+([0-9\.]+)(B|K|M|G)\s+[0-9\.]*\s*'
)
            for line in result.stdout.splitlines()[2:]:
                match = size_pattern.search(line)
                if match:
                    size_val, size_unit = float(match.group(1)), match.group(2).upper()
                    if size_unit == 'B': total_image_size_bytes += size_val
                    elif size_unit == 'K': total_image_size_bytes += size_val * 1024
                    elif size_unit == 'M': total_image_size_bytes += size_val * 1024 ** 2
                    elif size_unit == 'G': total_image_size_bytes += size_val * 1024 ** 3
            file_size_bytes = input_path.stat().st_size
            ratio = total_image_size_bytes / file_size_bytes if file_size_bytes > 0 else 0
            return {"image_to_file_ratio": ratio}
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(f"Failed to diagnose PDF with pdfimages: {e}")
            return {"image_to_file_ratio": 0.0, "error": str(e)}

    def compress_with_ghostscript(self, input_path: Path, output_path: Path, quality: int) -> bool:
        """Compresses a PDF using Ghostscript."""
        if not GHOSTSCRIPT_AVAILABLE: return False
        logger.info(f"Compressing with Ghostscript (Resolution: {self.gs_resolution} DPI, Quality: {quality}%)")
        command = [
            "gs", "-sDEVICE=pdfwrite", "-dCompatibilityLevel=1.4", "-dPDFSETTINGS=/default",
            "-dNOPAUSE", "-dQUIET", "-dBATCH", f"-dColorImageResolution={self.gs_resolution}",
            f"-dGrayImageResolution={self.gs_resolution}", f"-dMonoImageResolution={self.gs_resolution}",
            f"-dJPEGQuality={quality}", "-dDownsampleColorImages=true", "-dDownsampleGrayImages=true",
            "-dDownsampleMonoImages=true", "-dColorImageDownsampleType=/Bicubic",
            "-dGrayImageDownsampleType=/Bicubic", f"-sOutputFile={output_path}", str(input_path)
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Ghostscript failed. Return Code: {e.returncode}\nStderr: {e.stderr}")
            return False

    def compress_with_pypdf(self, input_path: Path, output_path: Path) -> bool:
        """Compresses a PDF using the pypdf library."""
        if not PYPDF_AVAILABLE: return False
        logger.info("Compressing with pypdf (standard content stream compression).")
        try:
            reader = PdfReader(input_path)
            writer = PdfWriter()
            for page in reader.pages:
                writer.add_page(page)
            writer.compress_content_streams()
            with open(output_path, "wb") as f:
                writer.write(f)
            return True
        except Exception as e:
            logger.error(f"pypdf compression failed for {input_path}: {e}", exc_info=True)
            return False

    def _handle_compression_result(self, original_size: int, compressed_path: Path, final_path: Path, original_path: Path) -> Dict[str, Any]:
        """Compares file sizes, moves/copies the file, and returns a result dictionary."""
        if not compressed_path.exists() or compressed_path.stat().st_size == 0:
            if compressed_path.exists(): os.remove(compressed_path)
            return {"success": False, "reason": "Compression resulted in an empty or missing file."}

        compressed_size = compressed_path.stat().st_size
        if compressed_size < original_size:
            reduction_percent = (1 - compressed_size / original_size) * 100
            logger.info(f"\u2713 Success! Size reduced by {reduction_percent:.1f}%. ({original_size:,} -> {compressed_size:,} bytes)")
            shutil.move(str(compressed_path), str(final_path))
            logger.info(f"Saved compressed file to: {final_path}")
            return {"success": True, "action": "compressed", "original_size": original_size, "final_size": compressed_size}
        else:
            logger.warning(f"\u2717 No improvement. Compressed size ({compressed_size:,}) is not smaller than original ({original_size:,}).")
            os.remove(compressed_path)
            if final_path != original_path:
                logger.info("Copying original file to destination.")
                shutil.copy2(original_path, final_path)
                logger.info(f"Saved original file to: {final_path}")
            return {"success": True, "action": "copied_original", "original_size": original_size, "final_size": original_size}

    def compress_pdf(self, input_path: Path, output_path: Path, method: str) -> Dict[str, Any]:
        """Handles the compression logic for a single PDF file."""
        try:
            original_size = input_path.stat().st_size
            logger.info(f"Processing PDF: '{input_path.name}' ({original_size / 1024 ** 2:.2f} MB)")
            temp_output_path = output_path.with_suffix(f"{output_path.suffix}.tmp")

            chosen_method = method
            if chosen_method == "auto":
                chosen_method = "ghostscript" if GHOSTSCRIPT_AVAILABLE else "pypdf"

            compression_success = False
            if chosen_method == "ghostscript":
                diagnosis = self._diagnose_pdf_content(input_path)
                image_ratio = diagnosis.get("image_to_file_ratio", 0.0)
                final_quality = self.gs_image_heavy_quality if image_ratio > self.image_ratio_threshold else self.gs_base_quality
                logger.info(f"PDF content is {image_ratio:.0%} image-heavy. Using quality: {final_quality}%.")
                compression_success = self.compress_with_ghostscript(input_path, temp_output_path, final_quality)
            elif chosen_method == "pypdf":
                compression_success = self.compress_with_pypdf(input_path, temp_output_path)
            else:
                logger.error(f"Unknown or unavailable method for PDF: {method}")
                return {"success": False, "reason": "Unknown PDF compression method"}

            if not compression_success:
                if temp_output_path.exists(): os.remove(temp_output_path)
                return {"success": False, "reason": "The compression process failed."}

            return self._handle_compression_result(original_size, temp_output_path, output_path, input_path)
        except Exception as e:
            logger.error(f"An unexpected error occurred while processing {input_path.name}: {e}", exc_info=True)
            return {"success": False, "reason": str(e)}

    def compress_pptx(self, input_path: Path, output_path: Path) -> Dict[str, Any]:
        """Compresses a PPTX file by re-compressing its internal images."""
        if not PILLOW_AVAILABLE: return {"success": False, "reason": "Pillow library not installed"}

        original_size = input_path.stat().st_size
        logger.info(f"Processing PPTX: '{input_path.name}' ({original_size / 1024 ** 2:.2f} MB)")
        temp_output_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
        temp_dir = Path(output_path.parent / f"temp_{input_path.stem}")

        try:
            if temp_dir.exists(): shutil.rmtree(temp_dir)
            temp_dir.mkdir()

            with zipfile.ZipFile(input_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            media_dir = temp_dir / "ppt" / "media"
            if media_dir.exists():
                logger.info(f"Optimizing images in PPTX with quality={self.pptx_image_quality}%.")
                for img_path in media_dir.iterdir():
                    if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']:
                        try:
                            with Image.open(img_path) as img:
                                img.save(img_path, optimize=True, quality=self.pptx_image_quality)
                        except Exception as img_e:
                            logger.warning(f"Could not compress image '{img_path.name}' in PPTX: {img_e}")

            with zipfile.ZipFile(temp_output_path, 'w', zipfile.ZIP_DEFLATED) as zip_out:
                for file_path in temp_dir.rglob('*'):
                    arcname = file_path.relative_to(temp_dir)
                    zip_out.write(file_path, arcname)

            return self._handle_compression_result(original_size, temp_output_path, output_path, input_path)

        except Exception as e:
            logger.error(f"Failed to process PPTX '{input_path.name}': {e}", exc_info=True)
            return {"success": False, "reason": str(e)}
        finally:
            if temp_dir.exists(): shutil.rmtree(temp_dir)

    def process_batch(self, input_paths: List[Path], pdf_method: str):
        """Processes a list of files, routing them to the correct compressor and replacing them."""
        total_files = len(input_paths)
        logger.info(f"Found {total_files} file(s) to process.")
        summary = {"compressed": 0, "copied": 0, "failed": 0}

        for i, file_path in enumerate(input_paths, 1):
            logger.info(f"\n--- [{i}/{total_files}] Processing: {file_path.name} ---")
            final_output_path = file_path

            result = {}
            ext = file_path.suffix.lower()
            if ext == '.pdf':
                result = self.compress_pdf(file_path, final_output_path, pdf_method)
            elif ext == '.pptx':
                result = self.compress_pptx(file_path, final_output_path)
            else:
                logger.warning(f"Skipping unsupported file type: '{file_path.name}'")
                summary["failed"] += 1
                continue

            if result.get("success"):
                action = result.get("action", "failed")
                if action == "compressed": summary["compressed"] += 1
                elif action == "copied_original": summary["copied"] += 1
            else:
                summary["failed"] += 1
                logger.error(f"Failed to process '{file_path.name}'. Reason: {result.get('reason', 'Unknown')}")

        logger.info("\n--- Batch Processing Complete ---")
        logger.info(f"Successfully compressed: {summary['compressed']}")
        logger.info(f"Originals copied (no size improvement): {summary['copied']}")
        logger.info(f"Failed or skipped: {summary['failed']}")
        logger.info("---------------------------------")

def gather_files(input_paths: List[Path]) -> List[Path]:
    """Gathers all supported files from the given list of paths."""
    files_to_process: List[Path] = []
    supported_extensions = ['.pdf', '.pptx']
    for path in input_paths:
        if not path.exists():
            logger.warning(f"Path does not exist, skipping: '{path}'")
            continue
        if path.is_file() and path.suffix.lower() in supported_extensions:
            files_to_process.append(path)
        elif path.is_dir():
            for ext in supported_extensions:
                found_files = sorted(list(path.rglob(f"*{ext}")))
                if found_files:
                    logger.info(f"Found {len(found_files)} {ext.upper()} file(s) in folder '{path.name}'.")
                    files_to_process.extend(found_files)
    return files_to_process

def main():
    """
    Main function to parse arguments and run the compression process.
    """
    parser = argparse.ArgumentParser(
        description="Compress PDF and PPTX files in a folder, replacing the originals.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input_folder", type=str, help="Path to the folder containing files to compress.")
    parser.add_argument(
        "--pdf_method",
        type=str,
        choices=["ghostscript", "pypdf", "auto"],
        default="ghostscript",
        help="Compression method for PDF files.\n"
             "  - ghostscript: (Recommended) Best compression, requires Ghostscript to be installed.\n"
             "  - pypdf: Safe, but less effective. Works without external dependencies.\n"
             "  - auto: Uses Ghostscript if available, otherwise falls back to pypdf.\n"
             "Default: ghostscript"
    )
    args = parser.parse_args()

    input_path = Path(args.input_folder)
    if not input_path.is_dir():
        logger.error(f"Error: The provided path '{input_path}' is not a valid directory.")
        return

    logger.info("--- File Compression Utility (PDF & PPTX) ---")

    files_to_process = gather_files([input_path])

    if not files_to_process:
        logger.info("No valid PDF or PPTX files were found in the provided folder. Exiting.")
        return

    compressor = FileCompressor()
    compressor.process_batch(
        input_paths=files_to_process,
        pdf_method=args.pdf_method
    )


if __name__ == "__main__":
    main()
