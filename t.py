from pathlib import Path
import json, cv2, numpy as np
from pdf2image import convert_from_path
from PIL import Image
from paddleocr import PaddleOCR           # pip install paddleocr==2.7.0.3
# Handwriting: HuggingFace TrOCR
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

# --- OCR engines ---
paddle = PaddleOCR(lang='en', use_angle_cls=True, show_log=False)  # printed text
paddle_struct = PaddleOCR(lang='en', use_angle_cls=True, show_log=False, structure_version='PP-Structure')
trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").eval()

def to_cv(img_pil):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def crop(img_cv, box):
    x0,y0,x1,y1 = map(int, box)
    return img_cv[y0:y1, x0:x1].copy()

def trocr_read(img_cv):
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    inputs = trocr_processor(images=img_pil, return_tensors="pt")
    with torch.no_grad():
        out_ids = trocr_model.generate(**inputs, max_new_tokens=256)
    return trocr_processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip()

def needs_trocr(text, conf):
    # Heuristics: low conf or “handwriting-ish” artifacts
    if conf < 0.75: return True
    # Many broken words / weird symbols → try TrOCR
    bad = sum(ch in "§¶¬~`^•¨" for ch in text)
    return bad >= 3

def ocr_pdf_image_only(pdf_path, dpi=500, outdir="out"):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    pages = convert_from_path(pdf_path, dpi=dpi)
    doc = {"doc_id": Path(pdf_path).stem, "pages": []}

    for page_idx, page in enumerate(pages, 1):
        w, h = page.size
        page_cv = to_cv(page)
        blocks_out = []

        # Layout detection (PP-Structure)
        # NOTE: if using paddleocr>=2.7, use 'structure' API for layout & table.
        result = paddle_struct.ocr(np.array(page), cls=True, det=True, rec=True, type='structure')
        # Fallback: if structure API differs, substitute with layoutparser or doclayout models.

        for blk in result:
            btype = blk.get('type', 'text')
            bbox = blk.get('bbox', [0,0,w,h])
            roi = crop(page_cv, bbox)

            if btype in ('text','title'):
                # Run printed OCR first
                txt_res = paddle.ocr(roi, cls=True)
                text, conf = "", 0.0
                if txt_res and txt_res[0]:
                    lines = []
                    confs = []
                    for line in txt_res[0]:
                        lines.append(line[1][0])
                        confs.append(float(line[1][1]))
                    text = " ".join(lines)
                    conf = float(np.mean(confs)) if confs else 0.0

                # Escalate to TrOCR if needed
                src = 'paddle'
                if needs_trocr(text, conf):
                    try:
                        text2 = trocr_read(roi)
                        if text2 and (len(text2) > len(text) or conf < 0.7):
                            text, src = text2, 'trocr'
                    except Exception:
                        pass

                blocks_out.append({
                    "id": f"p{page_idx}_b{len(blocks_out)+1}",
                    "type": btype,
                    "bbox": bbox,
                    "text": text or None,
                    "html": None,
                    "image_path": None,
                    "source": src,
                    "confidence": float(conf)
                })

            elif btype == 'table':
                # Save crop for later table parsing (Paddle table or TT)
                img_path = outdir / f"{Path(pdf_path).stem}_p{page_idx}_table_{len(blocks_out)+1}.png"
                cv2.imwrite(str(img_path), roi)
                # You can call Paddle table structure here to get HTML
                blocks_out.append({
                    "id": f"p{page_idx}_b{len(blocks_out)+1}",
                    "type": "table",
                    "bbox": bbox,
                    "text": None,
                    "html": None,             # fill with table HTML after table pass
                    "image_path": str(img_path),
                    "source": "paddle-struct",
                    "confidence": 1.0
                })

            else: # figure/equation/caption etc.
                img_path = outdir / f"{Path(pdf_path).stem}_p{page_idx}_fig_{len(blocks_out)+1}.png"
                cv2.imwrite(str(img_path), roi)
                blocks_out.append({
                    "id": f"p{page_idx}_b{len(blocks_out)+1}",
                    "type": btype if btype in ('figure','equation') else 'figure',
                    "bbox": bbox,
                    "text": None,
                    "html": None,
                    "image_path": str(img_path),
                    "source": "paddle-struct",
                    "confidence": 1.0
                })

        doc["pages"].append({"page": page_idx, "width_px": w, "height_px": h, "blocks": blocks_out})

    return doc

# Example:
# result = ocr_pdf_image_only("/path/to/scanned_textbook.pdf", outdir="ocr_out")
# Path("ocr_out/result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
