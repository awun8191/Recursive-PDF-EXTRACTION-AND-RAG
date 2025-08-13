import datetime
import json
import os
import time
from typing import List, Optional

from UtilityTools.Caching.cache import Cache
from Services.Gemini.rate_limit_data import RATE_LIMITS

class ApiKeyManager:
    """Manages a pool of API keys, rotating them as needed."""

    def __init__(self, api_keys: List[str], cache_file: str = "api_key_cache.json"):
        self.api_keys = api_keys
        self.cache = Cache(cache_file)
        self.cache_data = self._load_cache()
        self.current_key_index = self.cache_data.get("current_key_index", 0)
        self.rpm_timestamps: list[list[float]] = [[] for _ in api_keys]

    def _load_cache(self) -> dict:
        """Load cache data and reset if it's a new day."""
        cache_data = self.cache.read_cache()
        today = datetime.date.today().isoformat()

        if cache_data.get("date") != today:
            cache_data = {
                "date": today,
                "current_key_index": 0,
                "keys": {
                    key: {
                        "rpd": 0,
                        "total_tokens": 0,
                        "models": {
                            "flash": {"rpd": 0, "total_tokens": 0},
                            "lite": {"rpd": 0, "total_tokens": 0},
                            "pro": {"rpd": 0, "total_tokens": 0},
                        },
                    }
                    for key in self.api_keys
                },
            }
            self.cache.write_cache(cache_data)

        return cache_data

    def get_key(self) -> str:
        """Get the current API key."""
        if not self.api_keys:
            raise ValueError("No API keys configured.")

        if self.current_key_index >= len(self.api_keys):
            raise ValueError("All API keys have been used.")

        key = self.api_keys[self.current_key_index]
        if not self.is_key_available(key):
            return self.rotate_key()

        return key

    def is_key_available(self, key: str, model: str = "flash") -> bool:
        """Check if a key is within its usage limits."""
        key_data = self.cache_data["keys"][key]
        model_data = key_data["models"][model]
        rate_limit = RATE_LIMITS[model]

        # Check RPD
        if key_data["rpd"] >= rate_limit.per_day or model_data["rpd"] >= rate_limit.per_day:
            return False

        # Check RPM
        now = time.time()
        self.rpm_timestamps[self.current_key_index] = [
            t for t in self.rpm_timestamps[self.current_key_index] if now - t < 60
        ]
        if len(self.rpm_timestamps[self.current_key_index]) >= rate_limit.per_minute:
            return False

        # Check token usage (TPM) - This is a simple approximation
        if key_data["total_tokens"] >= 2_000_000 or model_data["total_tokens"] >= 2_000_000:
            return False

        return True

    def update_usage(self, key: str, model: str, tokens: int):
        """Update usage data for a key and model."""
        key_data = self.cache_data["keys"][key]
        model_data = key_data["models"][model]

        key_data["rpd"] += 1
        key_data["total_tokens"] += tokens
        model_data["rpd"] += 1
        model_data["total_tokens"] += tokens

        self.rpm_timestamps[self.current_key_index].append(time.time())
        self.cache.write_cache(self.cache_data)

    def rotate_key(self) -> str:
        """Rotate to the next available API key."""
        start_index = self.current_key_index
        while True:
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            if self.current_key_index == start_index:
                raise ValueError("All API keys are over their limits.")

            key = self.api_keys[self.current_key_index]
            if self.is_key_available(key):
                self.cache_data["current_key_index"] = self.current_key_index
                self.cache.write_cache(self.cache_data)
                return key
