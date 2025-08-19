#!/usr/bin/env python3
"""Test script for the new tail-based metadata parsing."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.metadata_extractor import MetadataExtractor, extract_document_metadata

def test_tail_based_parsing():
    """Test the new tail-based parsing with user's examples."""
    print("🧪 Testing Tail-Based Metadata Parsing")
    print("=" * 50)
    
    # Test cases based on user's examples
    test_cases = [
        {
            'path': r'C:\Users\awun8\Documents\SCHOOL\COMPILATION\EEE\300\1\EEE 313\EEE 313.pdf',
            'expected': {
                'department': 'EEE',
                'level': '300',
                'semester': '1',
                'course_code': 'EEE',
                'course_number': '313',
                'group_key': 'EEE-EEE-313'
            }
        },
        {
            'path': r'C:\Users\awun8\Documents\SCHOOL\COMPILATION\PTE\500\2\GENERAL\Reservoir Eng.pdf',
            'expected': {
                'department': 'PTE',
                'level': '500',
                'semester': '2',
                'category': 'GENERAL',
                'group_key': 'PTE'
            }
        },
        {
            'path': r'C:\Users\awun8\Documents\SCHOOL\COMPILATION\EEE\1\EEE 405\EEE 405 NOTE.pdf',
            'expected': {
                'department': 'EEE',
                'semester': '1',
                'course_code': 'EEE',
                'course_number': '405',
                'level': '400',  # derived from course number
                'group_key': 'EEE-EEE-405'
            }
        },
        {
            'path': r'C:\Users\awun8\Documents\SCHOOL\COMPILATION\100\1\AESA\'S 100 level PQ.pdf',
            'expected': {
                'level': '100',
                'semester': '1',
                'course_code': 'AESA',
                'course_number': '100',
                'category': 'PQ',
                'group_key': 'AESA-100'
            }
        },
        {
            'path': r'C:\Users\awun8\Documents\SCHOOL\COMPILATION\100\1\AFE  101\SUICIDE.pdf',
            'expected': {
                'level': '100',
                'semester': '1',
                'course_code': 'AFE',
                'course_number': '101',
                'group_key': 'AFE-101'
            }
        }
    ]
    
    extractor = MetadataExtractor()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📁 Test Case {i}: {Path(test_case['path']).name}")
        print(f"   Path: {test_case['path']}")
        
        try:
            # Create a temporary file for testing (we only need the path structure)
            metadata = extractor.extract_metadata(test_case['path'], content="Test content")
            
            print(f"\n✅ Extracted Metadata:")
            print(f"   Department: {metadata.department}")
            print(f"   Level: {metadata.level}")
            print(f"   Semester: {metadata.semester}")
            print(f"   Course Code: {metadata.course_code}")
            print(f"   Course Number: {metadata.course_number}")
            print(f"   Group Key: {metadata.tags.get('group_key', 'N/A') if metadata.tags else 'N/A'}")
            print(f"   Category: {metadata.tags.get('category', 'N/A') if metadata.tags else 'N/A'}")
            
            # Verify against expected results
            expected = test_case['expected']
            passed = True
            
            print(f"\n🔍 Verification:")
            for key, expected_value in expected.items():
                if key == 'group_key':
                    actual_value = metadata.tags.get('group_key') if metadata.tags else None
                elif key == 'category':
                    actual_value = metadata.tags.get('category') if metadata.tags else None
                else:
                    actual_value = getattr(metadata, key, None)
                
                if actual_value == expected_value:
                    print(f"   ✅ {key}: {actual_value} (expected: {expected_value})")
                else:
                    print(f"   ❌ {key}: {actual_value} (expected: {expected_value})")
                    passed = False
            
            if passed:
                print(f"   🎉 Test Case {i}: PASSED")
            else:
                print(f"   ⚠️  Test Case {i}: FAILED")
                
        except Exception as e:
            print(f"   ❌ Test Case {i}: ERROR - {e}")
        
        print("-" * 50)
    
    print("\n🏁 Tail-Based Parsing Test Complete!")

def test_level_normalization():
    """Test level normalization functionality."""
    print("\n🧪 Testing Level Normalization")
    print("=" * 30)
    
    extractor = MetadataExtractor()
    
    test_levels = [
        ('300', '300'),
        ('300L', '300'),
        ('300 LEVEL', '300'),
        ('300 level', '300'),
        ('400L', '400'),
        ('500 LEVEL', '500'),
        ('invalid', None),
        ('', None)
    ]
    
    for input_level, expected in test_levels:
        result = extractor._normalize_level(input_level)
        status = "✅" if result == expected else "❌"
        print(f"   {status} '{input_level}' → '{result}' (expected: '{expected}')")

def test_course_extraction():
    """Test course code/number extraction."""
    print("\n🧪 Testing Course Extraction")
    print("=" * 30)
    
    extractor = MetadataExtractor()
    
    test_cases = [
        ('EEE 313', ('EEE', '313')),
        ('AFE  101', ('AFE', '101')),
        ('AESA\'S 100', ('AESA', '100')),
        ('PTE-405', ('PTE', '405')),
        ('MATH_201', ('MATH', '201')),
        ('GENERAL', (None, None)),
        ('', (None, None))
    ]
    
    for input_text, expected in test_cases:
        result = extractor._extract_course_info(input_text, "")
        status = "✅" if result == expected else "❌"
        print(f"   {status} '{input_text}' → {result} (expected: {expected})")

def main():
    """Run all tests."""
    test_tail_based_parsing()
    test_level_normalization()
    test_course_extraction()
    print("\n🎯 All tests completed!")

if __name__ == "__main__":
    main()