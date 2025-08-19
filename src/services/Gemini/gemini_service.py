"""Wrapper around the Google Generative AI client with typed configuration."""

from __future__ import annotations

import json
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

import google.generativeai as genai
from google.generativeai.types.generation_types import StopCandidateException
from pydantic import BaseModel

from src.data_models.gemini_config import GeminiConfig
from .api_key_manager import ApiKeyManager

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "gemini-embedding-001"

class CourseOutline(BaseModel):
    course: str
    topics: List[str]
    description: str


class CoursesResponse(BaseModel):
    courses: List[CourseOutline]



class GeminiService:
    """Simple wrapper around ``google.generativeai``
    with typed configuration.
    """

    def __init__(
        self,
        api_keys: List[str] = None,
        model: str = DEFAULT_MODEL,
        generation_config: Optional[GeminiConfig] = None,
        api_key_manager: Optional[ApiKeyManager] = None,
    ) -> None:
        self.model = model
        self.default_config = generation_config or GeminiConfig()
        
        # Use default API keys if none provided
        if api_keys is None and api_key_manager is None:
            from .gemini_api_keys import GeminiApiKeys
            gemini_keys = GeminiApiKeys()
            api_keys = gemini_keys.get_keys()
        
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
            response_mime_type=(
                "application/json" if config.response_schema else None
            ),
        )

        tools = None
        if config.response_schema:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "response",
                        "description": "Response schema",
                        "parameters": (
                            config.response_schema.model_json_schema()
                        ),
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
            gen_model = genai.GenerativeModel(
                model, generation_config=gen_config
            )
            print("Step 2")
            response = gen_model.generate_content(list(parts))
            print(f"Response: {response.text}")

            # Update usage
            key = self.api_key_manager.get_key()
            model_name = self._get_model_name(model)
            # A simple way to estimate tokens, not perfect.
            tokens = len(response.text) / 4
            self.api_key_manager.update_usage(
                key, model_name, int(tokens)
            )

            # For plain text responses, return the text directly
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
            return self._generate(
                parts, model, generation_config, response_model
            )

    def _get_model_name(self, model_str: str) -> str:
        if "lite" in model_str:
            return "lite"
        if "flash" in model_str:
            return "flash"
        if "pro" in model_str:
            return "pro"
        if "embedding" in model_str:
            return "embedding"
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
        return self._generate(
            [prompt], model or self.model, generation_config, response_model
        )

    def embed(self, texts: List[str], model: Optional[str] = None, target_dim: Optional[int] = None) -> List[List[float]]:
        """Generate embeddings for a list of texts using Gemini embedding model."""
        if not texts:
            return []
        
        embedding_model = model or EMBEDDING_MODEL
        
        try:
            print(f"🔮 Generating embeddings for {len(texts)} texts using {embedding_model}")
            
            # Configure genai with current API key
            self._configure_genai()
            
            # Generate embeddings
            result = genai.embed_content(
                model=embedding_model,
                content=texts,
                task_type="retrieval_document"
            )
            
            embeddings = result['embedding'] if isinstance(result['embedding'][0], list) else [result['embedding']]
            
            # Optionally reduce dimensions if target_dim is specified
            if target_dim and embeddings and len(embeddings[0]) > target_dim:
                print(f"🔧 Reducing embedding dimensions from {len(embeddings[0])} to {target_dim}")
                # Use PCA-like reduction by taking the first N dimensions
                embeddings = [emb[:target_dim] for emb in embeddings]
            
            # Update usage tracking
            key = self.api_key_manager.get_key()
            model_name = self._get_model_name(embedding_model)
            # Estimate tokens for embeddings (rough approximation)
            total_tokens = sum(len(text.split()) for text in texts)
            self.api_key_manager.update_usage(key, model_name, total_tokens)
            
            print(f"✨ Successfully generated {len(embeddings)} embeddings with dimension {len(embeddings[0]) if embeddings else 0}")
            return embeddings
            
        except StopCandidateException as e:
            print(f"🔄 API key failed during embedding: {e}")
            self.api_key_manager.rotate_key()
            self._configure_genai()
            return self.embed(texts, model, target_dim)
        except Exception as e:
            print(f"❌ Error generating embeddings: {e}")
            raise

