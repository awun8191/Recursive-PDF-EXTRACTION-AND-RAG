"""Wrapper around the Google Generative AI client with typed configuration."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Type, TypeVar

import google.generativeai as genai
from google.generativeai.types.generation_types import StopCandidateException
from pydantic import BaseModel

from DataModels.gemini_config import GeminiConfig
from DataModels.ocr_data_model import OCRData
from DataModels.ocr_response_model import OCRItem, OCRResponse
from Services.Gemini.api_key_manager import ApiKeyManager

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "gemini-2.5-flash"
OCR_MODEL = "gemini-2.5-flash-lite"


class GeminiService:
    """Simple wrapper around ``google.generativeai`` with typed configuration."""

    def __init__(self, api_keys: List[str], model: str = DEFAULT_MODEL, ocr_model: str = OCR_MODEL,
                 generation_config: Optional[GeminiConfig] = None, api_key_manager: Optional[ApiKeyManager] = None) -> None:
        self.model = model
        self.ocr_model = ocr_model
        self.default_config = generation_config or GeminiConfig()
        self.api_key_manager = api_key_manager or ApiKeyManager(api_keys)
        self._configure_genai()

    def _configure_genai(self):
        """Configure the genai library with the current API key."""
        genai.configure(api_key=self.api_key_manager.get_key())

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
        try:
            print("Generation Started")
            gen_config, tools = self._to_generation_config(generation_config)
            print("Step 1")
            gen_model = genai.GenerativeModel(model, generation_config=gen_config)
            print("Step 2")
            response = gen_model.generate_content(list(parts))
            print(f"Response: {response.text}")

            # Update usage
            key = self.api_key_manager.get_key()
            model_name = self._get_model_name(model)
            # A simple way to estimate tokens, not perfect.
            tokens = len(response.text) / 4
            self.api_key_manager.update_usage(key, model_name, int(tokens))

            # For plain text responses (OCR), return the text directly
            if generation_config and generation_config.response_schema is None:
                return {"result": response.text.strip()}

            # For structured responses, handle JSON parsing
            cleaned_text = response.text.strip()
            if cleaned_text.startswith('```json'):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.endswith('```'):
                cleaned_text = cleaned_text[:-3]
            cleaned_text = cleaned_text.strip()

            try:
                data: List[Dict[str, Any]] = json.loads(cleaned_text)
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                # Fallback: try to extract JSON from the response
                import re
                json_match = re.search(r'\[.*\]', cleaned_text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    raise

            print(data)
            if response_model:
                return response_model.model_validate(data)
            return {
                "result": data
            }
        except StopCandidateException as e:
            # This exception is often thrown for quota-related issues.
            print(f"API key failed: {e}")
            self.api_key_manager.rotate_key()
            self._configure_genai()
            return self._generate(parts, model, generation_config, response_model)

    def _get_model_name(self, model_str: str) -> str:
        if "flash" in model_str:
            return "flash"
        if "lite" in model_str:
            return "lite"
        if "pro" in model_str:
            return "pro"
        return "flash"  # Default

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
        prompt: str = "Extract all text from this document image. Return only the extracted text with proper formatting and line breaks where need add additional commentary.",
        *,
        model: Optional[str] = None,
        generation_config: Optional[GeminiConfig | Dict[str, Any]] = None,
        response_model: Optional[Type[T]] = None,
    ) -> T | Dict[str, Any]:
        """Perform OCR on images using the lite model by default."""
        parts = [prompt] + images
        if generation_config is None:
            generation_config = GeminiConfig(
                temperature=0.1,  # Lower temperature for more consistent OCR
                top_p=0,
                top_k=None,
                max_output_tokens=8000,
                response_schema=None,  # Use plain text instead of structured output
            )
        else:
            if isinstance(generation_config, dict):
                generation_config = GeminiConfig(**generation_config)
            generation_config.response_schema = None  # Force plain text

        print(generation_config)

        result = self._generate(
            parts,
            model or self.ocr_model,
            generation_config,
            response_model=None,  # Use plain text
        )

        # Handle plain text response
        if isinstance(result, dict):
            text = result.get("result", "")
        else:
            text = str(result)
        
        print(text)
        return OCRData(text=text.strip())
