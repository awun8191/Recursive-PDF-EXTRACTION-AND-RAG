import sys
from pathlib import Path

import google.generativeai as genai

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.Gemini.gemini_service import GeminiService
from src.data_models.ocr_data_model import OCRData

class DummyResponse:
    def __init__(self, text):
        self.text = text

class DummyModel:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
    def generate_content(self, parts):
        return DummyResponse('{"text": "dummy"}')


def test_ocr_returns_ocrdata(monkeypatch):
    monkeypatch.setattr(genai, "GenerativeModel", DummyModel)
    service = GeminiService()
    result = service.ocr([{"mime_type": "image/png", "data": b""}])
    assert isinstance(result, OCRData)
    assert result.text == "dummy"
