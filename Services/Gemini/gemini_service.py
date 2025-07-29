"""Wrapper around the Google Generative AI client with typed configuration."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Type, TypeVar

import google.generativeai as genai
from pydantic import BaseModel

from DataModels.gemini_config import GeminiConfig
from DataModels.ocr_data_model import OCRData

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "gemini-2.5-flash"
OCR_MODEL = "gemini-2.5-flash-lite"


class GeminiService:
    """Simple wrapper around ``google.generativeai`` with typed configuration."""

    def __init__(self, model: str = DEFAULT_MODEL, ocr_model: str = OCR_MODEL,
                 generation_config: Optional[GeminiConfig] = None) -> None:
        self.model = model
        self.ocr_model = ocr_model
        self.default_config = generation_config or GeminiConfig()

    def _to_generation_config(
        self, config: Optional[GeminiConfig | Dict[str, Any]]
    ) -> tuple[genai.types.GenerationConfig, Optional[list[dict[str, Any]]]]:
        """Convert configuration data to a ``GenerationConfig`` and tools."""
        if config is None:
            config = self.default_config
        elif isinstance(config, dict):
            config = GeminiConfig(**config)
        elif not isinstance(config, GeminiConfig):
            raise TypeError("generation_config must be GeminiConfig or dict")

        gen_config = genai.types.GenerationConfig(
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            top_p=config.top_p,
            top_k=config.top_k,
            response_mime_type="application/json" if config.response_schema else None,
        )

        tools = None
        if config.response_schema:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "response",
                        "description": "Response schema",
                        "parameters": config.response_schema.model_json_schema(),
                    },
                }
            ]
        return gen_config, tools

    def _generate(
        self,
        parts: Iterable[Any],
        model: str,
        generation_config: Optional[GeminiConfig | Dict[str, Any]],
        response_model: Optional[Type[T]] = None,
    ) -> T | Dict[str, Any]:
        """Internal helper to perform a generation request."""
        gen_config, tools = self._to_generation_config(generation_config)
        gen_model = genai.GenerativeModel(model, generation_config=gen_config, tools=tools)
        response = gen_model.generate_content(list(parts))
        data = json.loads(response.text)
        if response_model:
            return response_model.model_validate(data)
        return data

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        generation_config: Optional[GeminiConfig | Dict[str, Any]] = None,
        response_model: Optional[Type[T]] = None,
    ) -> T | Dict[str, Any]:
        """Generate text from a prompt using the specified model."""
        return self._generate([prompt], model or self.model, generation_config, response_model)

    def ocr(
        self,
        images: List[Dict[str, Any]],
        prompt: str = "Extract text from this document image.",
        *,
        model: Optional[str] = None,
        generation_config: Optional[GeminiConfig | Dict[str, Any]] = None,
        response_model: Optional[Type[T]] = None,
    ) -> T | Dict[str, Any]:
        """Perform OCR on images using the lite model by default."""
        parts = [prompt] + images
        if generation_config is None:
            generation_config = GeminiConfig(
                temperature=0.9,
                top_p=0,
                top_k=None,
                max_output_tokens=8000,
                response_schema=OCRData,
            )
        else:
            if isinstance(generation_config, dict):
                generation_config = GeminiConfig(**generation_config)
            if generation_config.response_schema is None:
                generation_config.response_schema = OCRData
        return self._generate(
            parts,
            model or self.ocr_model,
            generation_config,
            response_model or OCRData,
        )
