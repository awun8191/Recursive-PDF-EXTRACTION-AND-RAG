import logging
import re
from pathlib import Path
from typing import Dict

import fitz

from ..UtilityTools.Caching.cache import Cache

ACCEPTABLE_TEXT_PERCENTAGE = 0.85
CACHE_FILE = "pdf_cache.json"


def open_pdf(file_path: str) -> fitz.Document:
    """Open a PDF file and return the document object."""
    return fitz.open(str(file_path))


def is_image_focused(file_path: str, text_threshold: int = 100,
                     cache: Cache | None = None) -> bool:
    """Determine if a PDF is primarily composed of image-only pages.

    ``text_threshold`` defines the minimum length of text for a page to be
    considered text-based. A page is considered "image only" when the length of
    its extracted text (``len(page.get_text().strip())``) is below this
    threshold and the page contains at least one image. If more than
    ``ACCEPTABLE_TEXT_PERCENTAGE`` (85% by default) of pages are image-only, the
    PDF is treated as requiring OCR.
    """
    cache_key = f"analysis_{Path(file_path).stem}"
    doc = open_pdf(file_path)
    total_pages = len(doc)

    image_only_pages = 0
    for page in doc:
        text_length = len(page.get_text().strip())
        has_image = bool(page.get_images(full=True))
        if text_length < text_threshold and has_image:
            image_only_pages += 1

    ratio = image_only_pages / total_pages if total_pages else 0
    result = ratio >= ACCEPTABLE_TEXT_PERCENTAGE

    if cache:
        try:
            analysis_data = {
                "file_path": str(file_path),
                "requires_ocr": result,
                "total_pages": total_pages,
                "image_only_pages": image_only_pages,
                "ratio": ratio,
                "timestamp": str(Path(file_path).stat().st_mtime),
            }
            cache.update_cache(cache_key, analysis_data)
        except Exception as e:
            logging.warning(f"Failed to cache analysis results: {e}")

    return result


def validate_latex_expressions(text: str) -> str:
    corrections = {
        r"\(\s*([^\)]+)\s*\)": r"$\1$",
        r"\[\s*([^\]]+)\s*\]": r"$$\1$$",
        r"\\frac\{([^{}]+)\}\{([^{}]+)\}": r"\\frac{\1}{\2}",
        r"\\sum_\{([^{}]+)\}": r"\\sum_{\1}",
        r"\\int_\{([^{}]+)\}": r"\\int_{\1}",
    }
    for pattern, replacement in corrections.items():
        text = re.sub(pattern, replacement, text)
    return text


def extract_engineering_content(content_type: str, text: str) -> Dict:
    structured_content = {
        "type": content_type,
        "raw_text": text,
        "equations": [],
        "tables": [],
        "diagrams": [],
        "code_snippets": [],
    }
    equation_pattern = r"\$([^$]+)\$|\$\$([^$]+)\$\$"
    equations = re.findall(equation_pattern, text)
    structured_content["equations"] = [eq[0] if eq[0] else eq[1] for eq in equations]

    table_pattern = r"\|(.+)\|[\r\n]+\|[-:\| ]+\|[\r\n]+((?:\|.+\|[\r\n]*)+)"
    tables = re.findall(table_pattern, text)
    structured_content["tables"] = [
        {"headers": t[0], "rows": t[1]} for t in tables
    ]

    code_pattern = r"```(\w+)?\n(.*?)```"
    code_snippets = re.findall(code_pattern, text, re.DOTALL)
    structured_content["code_snippets"] = [
        {"language": lang or "text", "code": code} for lang, code in code_snippets
    ]
    return structured_content


def ocr_text_extraction(
    file_path: str,
    gemini_service,
    cache: Cache | None = None,
    *,
    combine_pages: bool = False,
) -> str:
    """Perform OCR on a PDF and cache the results.

    If ``combine_pages`` is True the returned text will contain no page break markers,
    allowing downstream paragraph-based chunking.
    """
    cache_key = Path(file_path).stem

    if cache:
        try:
            cached = cache.read_cache()
            if cache_key in cached and "pages" in cached[cache_key]:
                pages = cached[cache_key]["pages"]
                if pages and all(str(i + 1) in pages for i in range(len(pages))):
                    return "\n\n--- PAGE BREAK ---\n\n".join(
                        pages[str(i + 1)]["text"] for i in range(len(pages))
                    )
        except Exception as e:
            logging.warning(f"Cache check failed for OCR: {e}")

    doc = open_pdf(file_path)
    images = [
        {"mime_type": "image/png", "data": page.get_pixmap(dpi=400).tobytes("png")}
        for page in doc
    ]

    pages_data = {}
    for idx, img in enumerate(images):
        page_num = str(idx + 1)
        text = ""
        try:
            response = gemini_service.ocr([img])
            text = response.get("text", "") if isinstance(response, dict) else ""
            text = validate_latex_expressions(text)
        except Exception as e:
            logging.error(f"OCR failed for page {page_num}: {e}")
        pages_data[page_num] = {
            "text": text,
            "text_length": len(text),
            "structured_content": extract_engineering_content("ocr", text),
        }

    if cache:
        try:
            cached = cache.read_cache()
            if cache_key not in cached:
                cached[cache_key] = {"metadata": {}, "pages": {}}
            cached[cache_key]["pages"].update(pages_data)
            cached[cache_key]["metadata"].update({
                "method": "ocr",
                "processing_timestamp": str(Path(file_path).stat().st_mtime),
                "total_chars": sum(len(p["text"]) for p in pages_data.values()),
                "ocr_pages_processed": len(pages_data),
            })
            cache.update_cache(cache_key, cached[cache_key])
        except Exception as e:
            logging.warning(f"Failed to cache OCR results: {e}")

    joined = "\n\n--- PAGE BREAK ---\n\n".join(
        pages_data[str(i + 1)]["text"] for i in range(len(images))
    )
    if combine_pages:
        joined = re.sub(r"\n?-+ PAGE BREAK -+\n?", "\n\n", joined)
    return joined
