import os

class GeminiApiKeys:
    """Provides a list of Gemini API keys."""

    def __init__(self, api_keys: list[str] = None):
        if api_keys is None:
            # Load keys from environment variables or a config file
            keys_str = os.getenv("GEMINI_API_KEYS", "")
            self.api_keys = [key.strip() for key in keys_str.split(",") if key.strip()]
        else:
            self.api_keys = api_keys

    def get_keys(self) -> list[str]:
        """Returns the list of API keys."""
        return self.api_keys
