"""Wrapper around the Ollama client for embedding generation."""

from __future__ import annotations

import time
import hashlib
from typing import List, Optional

try:
    import ollama
except ImportError:
    ollama = None

from src.utils.logging_utils import get_rag_logger


class OllamaService:
    """Service wrapper for Ollama embedding generation with caching and error handling."""

    def __init__(
        self,
        model: str = "bge-m3:latest",
        host: str = "http://localhost:11434",
        timeout: int = 30,
    ) -> None:
        self.model = model
        self.host = host
        self.timeout = timeout
        self.logger = get_rag_logger("OllamaService")
        
        if ollama is None:
            raise ImportError("ollama package is required. Install with: pip install ollama")
        
        # Initialize the Ollama client
        self.client = ollama.Client(host=host)
        
        # Verify the model is available
        self._verify_model()
        
        self.logger.success(f"Initialized Ollama service with model: {model}")

    def _verify_model(self) -> None:
        """Verify that the specified model is available."""
        try:
            # Test with a simple prompt to ensure model is available
            test_response = self.client.embeddings(
                model=self.model,
                prompt="test"
            )
            if "embedding" not in test_response:
                raise ValueError(f"Model {self.model} did not return expected embedding format")
            
            self.logger.info(f"Model {self.model} verified successfully")
            
        except Exception as e:
            self.logger.error(f"Model verification failed: {e}")
            raise RuntimeError(
                f"Failed to verify Ollama model '{self.model}'. "
                f"Please ensure Ollama is running and the model is pulled. "
                f"Run: ollama pull {self.model}"
            ) from e

    def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if not text.strip():
            self.logger.warning("Empty text provided for embedding")
            return []

        try:
            response = self.client.embeddings(
                model=self.model,
                prompt=text
            )
            
            if "embedding" not in response:
                raise ValueError("Invalid response format from Ollama")
            
            embedding = response["embedding"]
            self.logger.debug(f"Generated embedding with {len(embedding)} dimensions")
            return embedding
            
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {e}")
            raise

    def embed(self, texts: List[str], target_dim: Optional[int] = None) -> List[List[float]]:
        """Generate embeddings for multiple texts with optional dimensionality control."""
        if not texts:
            return []

        self.logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = []
        
        for i, text in enumerate(texts, 1):
            if i % 10 == 0:
                self.logger.info(f"Processing text {i}/{len(texts)}")
            
            try:
                embedding = self.embed_single(text)
                
                # Apply dimensionality reduction if requested
                if target_dim and len(embedding) > target_dim:
                    embedding = embedding[:target_dim]
                elif target_dim and len(embedding) < target_dim:
                    # Pad with zeros if needed
                    embedding.extend([0.0] * (target_dim - len(embedding)))
                
                embeddings.append(embedding)
                
            except Exception as e:
                self.logger.error(f"Failed to embed text {i}: {e}")
                # Add empty embedding to maintain list alignment
                dim = target_dim or 1024  # Default BGE-M3 dimension
                embeddings.append([0.0] * dim)
        
        self.logger.success(f"Generated {len(embeddings)} embeddings")
        return embeddings

    def get_model_info(self) -> dict:
        """Get information about the current model."""
        try:
            # Get model list to find our model's info
            models = self.client.list()
            for model in models.get("models", []):
                if model["name"] == self.model:
                    return {
                        "name": model["name"],
                        "size": model.get("size", "unknown"),
                        "digest": model.get("digest", "unknown"),
                        "modified_at": model.get("modified_at", "unknown")
                    }
            
            return {"name": self.model, "status": "not_found"}
            
        except Exception as e:
            self.logger.error(f"Failed to get model info: {e}")
            return {"name": self.model, "error": str(e)}

    def health_check(self) -> bool:
        """Check if Ollama service is healthy and responsive."""
        try:
            # Simple health check with minimal text
            response = self.client.embeddings(
                model=self.model,
                prompt="health check"
            )
            return "embedding" in response
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False