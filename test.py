from paddleocr import PaddleOCR

ocr = PaddleOCR(
    lang="en",
    use_textline_orientation=True,  # replaces use_angle_cls
    use_gpu=True                    # GPU if your Paddle install is GPU-enabled
)

# later, when you run it:
result = ocr.ocr("/home/raregazetto/Pictures/Screenshots/Screenshot From 2025-09-01 23-50-12.png", det=True, rec=True, cls=True)
print(result[:1])
