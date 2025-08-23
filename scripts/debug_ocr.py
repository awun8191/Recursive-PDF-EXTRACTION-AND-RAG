import os
from pathlib import Path
import fitz
from PIL import Image
import pytesseract

# Force your tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Show current environment configuration (do not override)
print("TESSDATA_PREFIX:", os.getenv("TESSDATA_PREFIX"))

pdf_path = Path(r"C:\\Users\\awun8\\Documents\\SCHOOL\\COMPILATION\\EEE\\300\\1\\EEE 313\\EEE 313.pdf")
print(f"Using PDF: {pdf_path}")

if not pdf_path.exists():
    raise SystemExit("PDF not found")

with fitz.open(pdf_path) as doc:
    p = doc[0]
    dpi = int(os.getenv("OCR_DPI", "300"))
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = p.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    print("Tesseract path:", pytesseract.pytesseract.tesseract_cmd)
    try:
        print("Tesseract version:", pytesseract.get_tesseract_version())
    except Exception as e:
        print("Failed to get tesseract version:", e)

    try:
        text = pytesseract.image_to_string(img, lang=os.getenv("OCR_LANG", "eng"), config="--oem 1 --psm 3")
        print("OCR length:", len(text))
        print(text[:500])
    except Exception as e:
        print("OCR failed:", repr(e))
        raise

