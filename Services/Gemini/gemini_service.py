import json
from typing import Any, Dict, Iterable, List, Optional, Type, TypeVar

import google.generativeai as genai
from pydantic import BaseModel

from DataModels.gemini_config import GeminiConfig

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "gemini-2.5-flash"
OCR_MODEL = "gemini-2.5-flash-lite"

class GeminiService:
    """Simple wrapper around ``google.generativeai`` with typed configuration.

    Parameters
    ----------
    model:
        The default model used for text generation requests.
    ocr_model:
        The model specifically used when performing OCR operations.
    generation_config:
        Optional :class:`GeminiConfig` providing defaults for generation
        settings. If omitted a new ``GeminiConfig`` instance with all default
        values is used.
    """

    def __init__(self, model: str = DEFAULT_MODEL, ocr_model: str = OCR_MODEL,
                 generation_config: Optional[GeminiConfig] = None) -> None:
        """Initialize the service with optional model overrides."""

        self.model = model
        self.ocr_model = ocr_model
        # Use the provided generation configuration or fall back to defaults.
        self.default_config = generation_config or GeminiConfig()

    def _to_generation_config(self, config: Optional[GeminiConfig | Dict[str, Any]]) -> genai.types.GenerationConfig:
        """Convert various config representations into ``GenerationConfig``."""

        if config is None:
            config = self.default_config
        elif isinstance(config, dict):
            config = GeminiConfig(**config)
        elif not isinstance(config, GeminiConfig):
            raise TypeError("generation_config must be GeminiConfig or dict")

        return genai.types.GenerationConfig(
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            top_p=config.top_p,
            top_k=config.top_k,
            response_mime_type="application/json",
        )

    def _generate(self, parts: Iterable[Any], model: str,
                  generation_config: Optional[GeminiConfig | Dict[str, Any]],
                  response_model: Optional[Type[T]] = None) -> T | Dict[str, Any]:
        """Internal helper to perform a generation request."""

        gen_config = self._to_generation_config(generation_config)
        gen_model = genai.GenerativeModel(model, generation_config=gen_config)
        response = gen_model.generate_content(list(parts))
        data = json.loads(response.text)
        if response_model:
            return response_model.model_validate(data)
        return data

    def generate(self, prompt: str, *, model: Optional[str] = None,
                 generation_config: Optional[GeminiConfig | Dict[str, Any]] = None,
                 response_model: Optional[Type[T]] = None) -> T | Dict[str, Any]:
        """Generate text from a prompt using the specified model."""
        return self._generate([prompt], model or self.model, generation_config, response_model)

    def ocr(self, images: List[Dict[str, Any]], prompt: str = "Extract text from this document image.", *,
            model: Optional[str] = None,
            generation_config: Optional[GeminiConfig | Dict[str, Any]] = None,
            response_model: Optional[Type[T]] = None) -> T | Dict[str, Any]:
        """Perform OCR on provided images using Gemini flash-lite by default."""
        parts = [prompt] + images
        return self._generate(parts, model or self.ocr_model, generation_config, response_model)
