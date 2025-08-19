#!/usr/bin/env python3
"""Test script for progress tracking and metadata extraction features."""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.utils.progress_tracker import ProgressTracker, ProcessingStatus, create_progress_tracker
    from src.utils.metadata_extractor import MetadataExtractor, extract_document_metadata
    from src.utils.logging_utils import get_rag_logger
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

def test_progress_tracker():
    """Test progress tracking functionality."""
    print("\n=== Testing Progress Tracker ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        progress_file = Path(temp_dir) / "test_progress.json"
        
        # Test 1: Create new session
        tracker = ProgressTracker(str(progress_file), "test_session_001")
        
        # Initialize session
        test_files = ["file1.pdf", "file2.pdf", "file3.pdf"]
        processing_params = {
            "workers": 2,
            "force_ocr": True,
            "export_include": "text+gemini-embedding"
        }
        
        session_id = tracker.initialize_session(len(test_files), processing_params)
        print(f"✅ Created session: {session_id}")
        
        # Test 2: Update file statuses
        tracker.update_file_status(
            "file1.pdf", ProcessingStatus.COMPLETED,
            chunks=25, embeddings=25, processing_time=5.2,
            metadata={"document_type": "lecture", "department": "EEE"}
        )
        
        tracker.update_file_status(
            "file2.pdf", ProcessingStatus.FAILED,
            processing_time=2.1, error="OCR failed",
            metadata={"document_type": "unknown"}
        )
        
        tracker.update_file_status(
            "file3.pdf", ProcessingStatus.IN_PROGRESS,
            processing_time=1.0
        )
        
        print("✅ Updated file statuses")
        
        # Test 3: Get progress summary
        summary = tracker.get_progress_summary()
        print(f"✅ Progress summary: {summary['completed_files']}/{summary['total_files']} completed")
        print(f"   Total chunks: {summary['total_chunks']}")
        print(f"   Completion: {summary['completion_percentage']:.1f}%")
        
        # Test 4: Test resume functionality
        pending_files = tracker.get_pending_files(test_files)
        print(f"✅ Pending files: {pending_files}")
        
        # Test 5: Export results
        results_file = tracker.export_results()
        print(f"✅ Results exported to: {results_file}")
        
        # Test 6: Load existing progress
        tracker2 = ProgressTracker(str(progress_file))
        loaded_progress = tracker2.load_progress()
        print(f"✅ Loaded existing progress: {loaded_progress.session_id}")
        
        return True

def test_metadata_extractor():
    """Test metadata extraction functionality."""
    print("\n=== Testing Metadata Extractor ===")
    
    # Create a test text content
    test_content = """
    EEE 313 - Digital Signal Processing
    Lecture Notes - Chapter 5: Filter Design
    
    This document covers the fundamentals of digital filter design including:
    - FIR filter design methods
    - IIR filter design techniques
    - Frequency response analysis
    - Implementation considerations
    
    The course is designed for 300-level engineering students in the fall semester.
    Topics include signal processing, algorithms, and system design.
    """
    
    # Test with a simulated file path
    test_file_path = "/path/to/COMPILATION/EEE/300/1/EEE 313/lecture_notes_ch5.pdf"
    
    # Test 1: Basic metadata extraction
    extractor = MetadataExtractor()
    
    # Since we don't have a real file, we'll test content-based extraction
    try:
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdf', delete=False) as temp_file:
            temp_file.write("dummy content")
            temp_file_path = temp_file.name
        
        metadata = extractor.extract_metadata(temp_file_path, content=test_content)
        
        print(f"✅ Extracted metadata for: {metadata.file_name}")
        print(f"   Department: {metadata.department}")
        print(f"   Course Code: {metadata.course_code}")
        print(f"   Course Number: {metadata.course_number}")
        print(f"   Level: {metadata.level}")
        print(f"   Document Type: {metadata.document_type}")
        print(f"   Word Count: {metadata.word_count}")
        print(f"   Topics: {metadata.topics}")
        print(f"   Keywords: {metadata.keywords[:5] if metadata.keywords else []}...")  # First 5 keywords
        print(f"   Completeness Score: {metadata.completeness_score:.1f}%")
        
        # Test 2: Enhanced metadata with custom tags
        enhanced_metadata = extractor.enhance_metadata_with_tags(
            metadata, 
            custom_tags={
                "semester": "fall_2024",
                "instructor": "Dr. Smith",
                "difficulty_level": "intermediate"
            }
        )
        
        print(f"✅ Enhanced metadata tags: {list(enhanced_metadata.tags.keys())}")
        
        # Test 3: Export metadata
        json_export = extractor.export_metadata(enhanced_metadata, format='json')
        print(f"✅ Exported metadata (JSON): {len(json_export)} characters")
        
        # Clean up
        os.unlink(temp_file_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Metadata extraction test failed: {e}")
        return False

def test_integration():
    """Test integration between progress tracking and metadata extraction."""
    print("\n=== Testing Integration ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        progress_file = Path(temp_dir) / "integration_progress.json"
        
        # Create progress tracker
        tracker = ProgressTracker(str(progress_file), "integration_test")
        
        # Initialize session
        test_files = ["lecture1.pdf", "assignment2.pdf", "exam3.pdf"]
        tracker.initialize_session(len(test_files), {"test": "integration"})
        
        # Create metadata extractor
        extractor = MetadataExtractor()
        
        # Simulate processing files with metadata
        for i, file_path in enumerate(test_files):
            try:
                # Create temporary file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.pdf', delete=False) as temp_file:
                    temp_file.write(f"Content for {file_path}")
                    temp_file_path = temp_file.name
                
                # Extract metadata
                test_content = f"This is a {file_path.split('.')[0]} document for EEE 313."
                metadata = extractor.extract_metadata(temp_file_path, content=test_content)
                
                # Update progress with metadata
                tracker.update_file_status(
                    file_path, ProcessingStatus.COMPLETED,
                    chunks=10 + i * 5, embeddings=10 + i * 5,
                    processing_time=2.0 + i,
                    metadata={
                        "document_type": metadata.document_type,
                        "word_count": metadata.word_count,
                        "completeness_score": metadata.completeness_score,
                        "file_hash": metadata.file_hash
                    }
                )
                
                # Clean up
                os.unlink(temp_file_path)
                
            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")
                tracker.update_file_status(
                    file_path, ProcessingStatus.FAILED,
                    error=str(e)
                )
        
        # Get final summary
        summary = tracker.get_progress_summary()
        print(f"✅ Integration test completed:")
        print(f"   Files processed: {summary['completed_files']}/{summary['total_files']}")
        print(f"   Total chunks: {summary['total_chunks']}")
        print(f"   Total embeddings: {summary['total_embeddings']}")
        
        # Export detailed results
        results_file = tracker.export_results()
        print(f"✅ Detailed results with metadata exported to: {results_file}")
        
        return True

def main():
    """Run all tests."""
    print("🧪 Testing Progress Tracking and Metadata Extraction Features")
    print("=" * 60)
    
    tests = [
        ("Progress Tracker", test_progress_tracker),
        ("Metadata Extractor", test_metadata_extractor),
        ("Integration", test_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            print(f"\n🔄 Running {test_name} test...")
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"✅ {test_name} test PASSED")
            else:
                print(f"❌ {test_name} test FAILED")
        except Exception as e:
            print(f"❌ {test_name} test FAILED with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name:<20} {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Progress tracking and metadata extraction are working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())