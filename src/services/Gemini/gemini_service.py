"""Wrapper around the Google Generative AI client with typed configuration."""

from __future__ import annotations

import json
import time
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
from google.api_core.exceptions import ResourceExhausted
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

    def _configure_genai(self, model: str = "flash"):
        """Configure the genai library with the current API key for the given model."""
        genai.configure(api_key=self.api_key_manager.get_key(model))

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
            # Configure per-model key (flash/lite/pro)
            model_name = self._get_model_name(model)
            self._configure_genai(model_name)
            gen_model = genai.GenerativeModel(
                model, generation_config=gen_config
            )
            print("Step 2")
            response = gen_model.generate_content(list(parts))
            print(f"Response: {response.text}")

            # Update usage
            key = self.api_key_manager.get_key(model_name)
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
        except (StopCandidateException, ResourceExhausted) as e:
            # This exception is often thrown for quota-related issues.
            print(f"API key failed: {e}")
            # Rotate within the same model family
            self.api_key_manager.rotate_key(model_name)
            self._configure_genai(model_name)
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

    # -------------------- Helper methods for embedding batching --------------------
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count using a simple 4-characters-per-token heuristic."""
        return max(1, len(text) // 4)

    def _batch_texts(self, texts: List[str], max_tokens: int = 2000) -> List[List[str]]:
        """Split ``texts`` so each batch stays within ``max_tokens`` heuristic limit."""
        batches: List[List[str]] = []
        current: List[str] = []
        current_tokens = 0
        for t in texts:
            t_tok = self._estimate_tokens(t)
            # Extremely long text: send alone to avoid infinite loop
            if t_tok > max_tokens:
                if current:
                    batches.append(current)
                    current, current_tokens = [], 0
                batches.append([t])
                continue
            if current_tokens + t_tok > max_tokens and current:
                batches.append(current)
                current, current_tokens = [t], t_tok
            else:
                current.append(t)
                current_tokens += t_tok
        if current:
            batches.append(current)
        return batches

    # -------------------- Main embedding method --------------------
    def embed(self, texts: List[str], model: Optional[str] = None, target_dim: Optional[int] = 768) -> List[List[float]]:
        """Generate embeddings while respecting ~2 k tokens/request and ≤5 RPM limits."""
        if not texts:
            return []

        embedding_model = model or EMBEDDING_MODEL
        batches = self._batch_texts(texts, max_tokens=2000)
        
        # Track rotation cycles for intelligent delay
        rotation_cycles = 0
        max_cycles_before_delay = 5

        all_embeddings: List[List[float]] = []
        for batch_idx, batch in enumerate(batches, 1):
            # Check if we need to wait for RPM limits BEFORE making the request
            should_delay = self._wait_for_rpm_if_needed("embedding", rotation_cycles, max_cycles_before_delay)
            if should_delay:
                rotation_cycles += 1
            
            try:
                approx_tokens = sum(self._estimate_tokens(t) for t in batch)
                print(
                    f"🔮 Embedding batch {batch_idx}/{len(batches)} "
                    f"({len(batch)} texts, ≈{approx_tokens} tokens) using {embedding_model}"
                )

                # Refresh/validate API key before each request (per embedding model)
                self._configure_genai("embedding")

                result = genai.embed_content(
                    model=embedding_model,
                    content=batch if len(batch) > 1 else batch[0],
                    task_type="retrieval_document",
                )

                embeddings = (
                    result["embedding"]
                    if isinstance(result["embedding"][0], list)
                    else [result["embedding"]]
                )

                # Optional dimensionality reduction
                if target_dim and embeddings and len(embeddings[0]) > target_dim:
                    embeddings = [emb[:target_dim] for emb in embeddings]

                all_embeddings.extend(embeddings)

                # Update usage (tokens counted via simple whitespace split)
                key = self.api_key_manager.get_key("embedding")
                model_name = self._get_model_name(embedding_model)
                self.api_key_manager.update_usage(
                    key, model_name, sum(len(t.split()) for t in batch)
                )

            except (StopCandidateException, ResourceExhausted) as e:
                print(f"🔄 API key failed during embedding: {e}")
                try:
                    self.api_key_manager.rotate_key("embedding")
                    self._configure_genai("embedding")
                    rotation_cycles += 1
                except ValueError as rotate_error:
                    if "All API keys are over their limits" in str(rotate_error):
                        # All keys exhausted, implement delay based on rotation cycles
                        if rotation_cycles >= max_cycles_before_delay:
                            delay_time = 70  # 70 seconds to ensure RPM window resets
                            print(f"⏰ All API keys exhausted after {rotation_cycles} rotations. "
                                  f"Waiting {delay_time}s for rate limits to reset...")
                            import time
                            time.sleep(delay_time)
                            rotation_cycles = 0  # Reset counter after delay
                            # Reset to first key and try again
                            self.api_key_manager.current_key_index = 0
                            self._configure_genai("embedding")
                        else:
                            raise rotate_error
                    else:
                        raise rotate_error
                
                # Retry the same batch with the next key
                retry_embeddings = self.embed(batch, model, target_dim)
                all_embeddings.extend(retry_embeddings)
            except Exception as e:
                print(f"❌ Error generating embeddings for batch {batch_idx}: {e}")
                raise

        print(
            f"✨ Successfully generated {len(all_embeddings)} embeddings across {len(batches)} request(s)"
        )
        return all_embeddings

    def _wait_for_rpm_if_needed(self, model: str = "flash", rotation_cycles: int = 0, max_cycles: int = 5) -> bool:
        """Wait if the current key is approaching its RPM limit. Returns True if a rotation cycle occurred."""
        from .rate_limit_data import RATE_LIMITS
        
        rate_limit = RATE_LIMITS.get(model, RATE_LIMITS["flash"])
        # Use per-model timestamps
        current_timestamps = self.api_key_manager.rpm_timestamps[self.api_key_manager.current_key_index][model]
        
        now = time.time()
        # Clean up old timestamps (older than 60 seconds)
        current_timestamps[:] = [t for t in current_timestamps if now - t < 60]
        
        # If we're at or near the RPM limit, wait until the oldest request is >60s old
        if len(current_timestamps) >= rate_limit.per_minute:
            oldest_request = min(current_timestamps)
            wait_time = 60 - (now - oldest_request) + 1  # +1 second buffer
            
            if wait_time > 0:
                # Check if we should implement extended delay after multiple rotations
                if rotation_cycles >= max_cycles:
                    extended_wait = wait_time + 60  # Additional minute after 5+ rotations
                    print(f"⏳ RPM limit reached after {rotation_cycles} rotations ({len(current_timestamps)}/{rate_limit.per_minute}). "
                          f"Extended wait: {extended_wait:.1f}s to fully reset rate limits...")
                    time.sleep(extended_wait)
                else:
                    print(f"⏳ RPM limit reached ({len(current_timestamps)}/{rate_limit.per_minute}). "
                          f"Waiting {wait_time:.1f}s to respect rate limits...")
                    time.sleep(wait_time)
                return True  # Indicate a rotation cycle occurred
        return False

