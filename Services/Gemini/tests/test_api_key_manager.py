import os
import unittest
from unittest.mock import patch, mock_open
from Services.Gemini.api_key_manager import ApiKeyManager

class TestApiKeyManager(unittest.TestCase):

    def setUp(self):
        self.api_keys = ["key1", "key2", "key3"]
        self.cache_file = "test_cache.json"

    def tearDown(self):
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)

    @patch("Services.UtilityTools.Caching.cache.Cache.read_cache")
    def test_get_key(self, mock_read_cache):
        mock_read_cache.return_value = {}
        manager = ApiKeyManager(self.api_keys, self.cache_file)
        self.assertEqual(manager.get_key(), "key1")

    @patch("Services.UtilityTools.Caching.cache.Cache.read_cache")
    @patch("Services.UtilityTools.Caching.cache.Cache.write_cache")
    def test_rotate_key(self, mock_write_cache, mock_read_cache):
        mock_read_cache.return_value = {"date": "2023-10-27", "current_key_index": 0, "used_keys": []}
        manager = ApiKeyManager(self.api_keys, self.cache_file)

        # First rotation
        new_key = manager.rotate_key()
        self.assertEqual(new_key, "key2")
        self.assertEqual(manager.current_key_index, 1)

        # Second rotation
        new_key = manager.rotate_key()
        self.assertEqual(new_key, "key3")
        self.assertEqual(manager.current_key_index, 2)

    @patch("Services.UtilityTools.Caching.cache.Cache.read_cache")
    @patch("Services.UtilityTools.Caching.cache.Cache.write_cache")
    def test_all_keys_used(self, mock_write_cache, mock_read_cache):
        mock_read_cache.return_value = {"date": "2023-10-27", "current_key_index": 0, "used_keys": []}
        manager = ApiKeyManager(self.api_keys, self.cache_file)
        manager.rotate_key()  # key 2
        manager.rotate_key()  # key 3
        with self.assertRaises(ValueError):
            manager.rotate_key() # No keys left

if __name__ == "__main__":
    unittest.main()
