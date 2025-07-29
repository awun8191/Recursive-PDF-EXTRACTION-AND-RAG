from pydantic import BaseModel, Field
from typing import Optional

class GeminiConfig(BaseModel):
    """Configuration for Gemini generation requests."""
    temperature: float = Field(0.2, description="Sampling temperature")
    max_output_tokens: int = Field(10000, description="Maximum tokens in the response")
    top_p: Optional[float] = Field(0.8, description="Nucleus sampling p value")
    top_k: Optional[int] = Field(None, description="Top-k sampling value")
