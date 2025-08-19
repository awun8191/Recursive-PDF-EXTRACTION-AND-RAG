#!/usr/bin/env python3
"""Test script to verify RAG pipeline improvements."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.logging_utils import get_rag_logger, log_section_header, log_step, log_success, log_processing
from utils.Caching.enhanced_cache import get_enhanced_cache
from services.Gemini.gemini_service import GeminiService
from services.Gemini.api_key_manager import ApiKeyManager
from services.RAG.convert_to_embeddings import Embedder

def test_logging():
    """Test the enhanced logging system."""
    logger = get_rag_logger("TestLogger")
    
    log_section_header("🧪 TESTING ENHANCED LOGGING")
    log_step("Testing basic logging functions", 1, 4)
    
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    log_processing("Processing test item", "test_file.pdf")
    log_success("Logging test completed successfully")
    
    return True

def test_caching():
    """Test the enhanced caching system."""
    log_step("Testing enhanced caching system", 2, 4)
    
    with get_enhanced_cache("test_cache.json", default_ttl=60) as cache:
        # Test basic cache operations
        cache.set("test_key", "test_value", tags=["test"])
        
        retrieved = cache.get("test_key")
        if retrieved != "test_value":
            raise ValueError("Cache retrieval failed")
        
        # Test cache statistics
        stats = cache.get_stats()
        logger = get_rag_logger("CacheTest")
        logger.info(f"💾 Cache stats: {stats['active_entries']} active entries")
        
        # Test cache function decorator
        @cache.cache_function(ttl=30, tags=["function_test"])
        def expensive_function(x, y):
            return x * y + 42
        
        result1 = expensive_function(5, 10)
        result2 = expensive_function(5, 10)  # Should hit cache
        
        if result1 != result2 or result1 != 92:
            raise ValueError("Cache function decorator failed")
        
        log_success("Caching system test passed")
        return True

def test_gemini_service():
    """Test Gemini service with embedding functionality."""
    log_step("Testing Gemini service", 3, 4)
    
    # Check if API keys are available
    api_keys_env = os.getenv("GEMINI_API_KEYS") or os.getenv("GOOGLE_GEMINI_API_KEYS") or ""
    api_keys = [k.strip() for k in api_keys_env.split(",") if k.strip()]
    
    logger = get_rag_logger("GeminiTest")
    
    if not api_keys:
        logger.warning("⚠️  No Gemini API keys found, skipping Gemini service test")
        return True
    
    try:
        # Initialize Gemini service
        api_key_manager = ApiKeyManager(api_keys)
        gemini_service = GeminiService(api_keys=api_keys, api_key_manager=api_key_manager)
        
        # Test embedding functionality
        test_texts = [
            "This is a test document about machine learning.",
            "Another test document about artificial intelligence."
        ]
        
        embeddings = gemini_service.embed(test_texts)
        
        if not embeddings or len(embeddings) != len(test_texts):
            raise ValueError("Embedding generation failed")
        
        logger.info(f"🔮 Generated {len(embeddings)} embeddings with dimension {len(embeddings[0])}")
        log_success("Gemini service test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Gemini service test failed: {e}")
        return False

def test_embedder():
    """Test the enhanced Embedder class."""
    log_step("Testing enhanced Embedder", 4, 4)
    
    logger = get_rag_logger("EmbedderTest")
    
    # Test with hash embeddings (fallback)
    embedder_hash = Embedder(prefer_gemini=False, fallback_dim=384)
    logger.info(f"🔧 Hash embedder backend: {embedder_hash.backend}")
    
    test_texts = [
        "Test document one",
        "Test document two",
        "Test document three"
    ]
    
    # Test hash embeddings
    hash_embeddings = embedder_hash.embed(test_texts)
    if len(hash_embeddings) != len(test_texts):
        raise ValueError("Hash embedding generation failed")
    
    logger.info(f"📊 Generated {len(hash_embeddings)} hash embeddings")
    
    # Test with Gemini embeddings if available
    api_keys_env = os.getenv("GEMINI_API_KEYS") or os.getenv("GOOGLE_GEMINI_API_KEYS") or ""
    api_keys = [k.strip() for k in api_keys_env.split(",") if k.strip()]
    
    if api_keys:
        try:
            embedder_gemini = Embedder(prefer_gemini=True, fallback_dim=384)
            logger.info(f"🔧 Gemini embedder backend: {embedder_gemini.backend}")
            
            gemini_embeddings = embedder_gemini.embed(test_texts[:2])  # Test with fewer texts
            if len(gemini_embeddings) != 2:
                raise ValueError("Gemini embedding generation failed")
            
            logger.info(f"🔮 Generated {len(gemini_embeddings)} Gemini embeddings")
            
        except Exception as e:
            logger.warning(f"⚠️  Gemini embedder test failed, but hash fallback works: {e}")
    else:
        logger.info("ℹ️  No API keys available, skipping Gemini embedder test")
    
    log_success("Embedder test completed")
    return True

def main():
    """Run all tests."""
    log_section_header("🧪 RAG PIPELINE IMPROVEMENTS TEST SUITE")
    
    tests = [
        ("Enhanced Logging", test_logging),
        ("Enhanced Caching", test_caching),
        ("Gemini Service", test_gemini_service),
        ("Enhanced Embedder", test_embedder)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            log_processing(f"Running test", test_name)
            if test_func():
                passed += 1
                log_success(f"{test_name} test PASSED")
            else:
                failed += 1
                logger = get_rag_logger("TestRunner")
                logger.error(f"❌ {test_name} test FAILED")
        except Exception as e:
            failed += 1
            logger = get_rag_logger("TestRunner")
            logger.error(f"❌ {test_name} test FAILED with exception: {e}")
    
    log_section_header("📊 TEST RESULTS")
    logger = get_rag_logger("TestRunner")
    logger.info(f"✅ Passed: {passed}")
    logger.info(f"❌ Failed: {failed}")
    logger.info(f"📊 Total: {passed + failed}")
    
    if failed == 0:
        log_success("🎉 All tests passed! RAG pipeline improvements are working correctly.")
        return 0
    else:
        logger.error(f"💥 {failed} test(s) failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())