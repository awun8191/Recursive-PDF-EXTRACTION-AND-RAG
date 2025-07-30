import datetime
import json
import os
from typing import List, Optional

from Services.UtilityTools.Caching.cache import Cache

class ApiKeyManager:
    """Manages a pool of API keys, rotating them as needed."""

    def __init__(self, api_keys: List[str], cache_file: str = "api_key_cache.json"):
        self.api_keys = api_keys
        self.cache = Cache(cache_file)
        self.cache_data = self._load_cache()
        self.current_key_index = self.cache_data.get("current_key_index", 0)

    def _load_cache(self) -> dict:
        """Load cache data and reset if it's a new day."""
        cache_data = self.cache.read_cache()
        today = datetime.date.today().isoformat()

        if cache_data.get("date") != today:
            cache_data = {"date": today, "current_key_index": 0, "used_keys": []}
            self.cache.write_cache(cache_data)

        return cache_data

    def get_key(self) -> str:
        """Get the current API key."""
        if not self.api_keys:
            raise ValueError("No API keys configured.")

        if self.current_key_index >= len(self.api_keys):
            raise ValueError("All API keys have been used.")

        return self.api_keys[self.current_key_index]

    def rotate_key(self) -> str:
        """Rotate to the next available API key."""
        used_keys = self.cache_data.get("used_keys", [])
        used_keys.append(self.current_key_index)
        self.cache_data["used_keys"] = used_keys

        # Find the next unused key
        next_index = -1
        for i in range(len(self.api_keys)):
            if i not in used_keys:
                next_index = i
                break

        if next_index == -1:
            self.current_key_index = len(self.api_keys)
            self.cache_data["current_key_index"] = self.current_key_index
            self.cache.write_cache(self.cache_data)
            raise ValueError("All API keys have been used.")

        self.current_key_index = next_index
        self.cache_data["current_key_index"] = self.current_key_index
        self.cache.write_cache(self.cache_data)

        return self.get_key()
