import os
import unittest
from unittest.mock import patch, mock_open
from Services.Gemini.api_key_manager import ApiKeyManager
from Services.Gemini.rate_limit_data import RATE_LIMITS
import datetime

class TestApiKeyManager(unittest.TestCase):

    def setUp(self):
        self.api_keys = ["key1", "key2", "key3"]
        self.cache_file = "test_cache.json"
        self.today = datetime.date.today().isoformat()

    def tearDown(self):
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)

    def _get_mock_cache(self, used_keys_rpd=None):
        if used_keys_rpd is None:
            used_keys_rpd = {}

        cache = {
            "date": self.today,
            "current_key_index": 0,
            "keys": {},
        }
        for key in self.api_keys:
            cache["keys"][key] = {
                "rpd": used_keys_rpd.get(key, 0),
                "total_tokens": 0,
                "models": {
                    "flash": {"rpd": used_keys_rpd.get(key, 0), "total_tokens": 0},
                    "lite": {"rpd": 0, "total_tokens": 0},
                    "pro": {"rpd": 0, "total_tokens": 0},
                },
            }
        return cache

    @patch("UtilityTools.Caching.cache.Cache.read_cache")
    def test_get_key(self, mock_read_cache):
        mock_read_cache.return_value = self._get_mock_cache()
        manager = ApiKeyManager(self.api_keys, self.cache_file)
        self.assertEqual(manager.get_key(), "key1")

    @patch("UtilityTools.Caching.cache.Cache.read_cache")
    @patch("UtilityTools.Caching.cache.Cache.write_cache")
    def test_rotate_key(self, mock_write_cache, mock_read_cache):
        mock_read_cache.return_value = self._get_mock_cache()
        manager = ApiKeyManager(self.api_keys, self.cache_file)

        # Rotate to key2
        manager.cache_data["keys"]["key1"]["rpd"] = RATE_LIMITS["flash"].per_day
        new_key = manager.rotate_key()
        self.assertEqual(new_key, "key2")
        self.assertEqual(manager.current_key_index, 1)

        # Rotate to key3
        manager.cache_data["keys"]["key2"]["rpd"] = RATE_LIMITS["flash"].per_day
        new_key = manager.rotate_key()
        self.assertEqual(new_key, "key3")
        self.assertEqual(manager.current_key_index, 2)

    @patch("UtilityTools.Caching.cache.Cache.read_cache")
    @patch("UtilityTools.Caching.cache.Cache.write_cache")
    def test_all_keys_used(self, mock_write_cache, mock_read_cache):
        mock_read_cache.return_value = self._get_mock_cache(
            used_keys_rpd={"key1": RATE_LIMITS["flash"].per_day, "key2": RATE_LIMITS["flash"].per_day, "key3": RATE_LIMITS["flash"].per_day}
        )
        manager = ApiKeyManager(self.api_keys, self.cache_file)
        with self.assertRaises(ValueError):
            manager.get_key()

if __name__ == "__main__":
    unittest.main()
