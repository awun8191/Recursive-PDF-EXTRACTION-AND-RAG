import os

class GeminiApiKeys:
    """Provides a list of Gemini API keys."""

    def __init__(self, api_keys: list[str] = None):
        if api_keys is None:
            # Load keys from environment variables or a config file

            self.api_keys = ["AIzaSyCP6igfyX0FTLiWxN0os50nvN748gn6YiA", "AIzaSyDzTpwiUeMqWVXG9tEpNHnhpvLU2Zf3RE0"]
        else:
            self.api_keys = api_keys

    def get_keys(self) -> list[str]:
        """Returns the list of API keys."""
        return self.api_keys
