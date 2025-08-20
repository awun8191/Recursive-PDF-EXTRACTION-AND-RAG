#!/usr/bin/env python3
"""
Test script to demonstrate the improved OCR detection capabilities.

This script compares the old vs new OCR detection methods on sample PDFs
to show how the advanced analysis provides better accuracy.
"""

import sys
from pathlib import Path
import fitz  # PyMuPDF

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.services.RAG.convert_to_embeddings import (
    analyze_pdf_sample, 
    analyze_pdf_advanced,
    analyze_text_quality
)
from src.utils.logging_utils import get_rag_logger

def test_text_quality_analysis():
    """Test the text quality analysis function with various text samples."""
    logger = get_rag_logger("TextQualityTest")
    
    test_cases = [
        ("This is a well-formatted document with proper sentences. It contains meaningful content that should be easily readable.", "Good quality text"),
        ("asdkjfh askdjfh askjdfh askjdfh askjdfh", "Random characters"),
        ("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", "Repeated characters"),
        ("Thisisatextwithnospacesbetweenwordswhichiscommoninpoorlyprocessedocr", "No spaces"),
        ("T h i s   i s   t e x t   w i t h   t o o   m a n y   s p a c e s", "Too many spaces"),
        ("This text has good structure. It contains multiple sentences with proper punctuation! Questions work too?", "High quality text"),
        ("", "Empty text"),
        ("123 456 789 !@# $%^ &*()", "Numbers and symbols only")
    ]
    
    logger.info("🧪 Testing text quality analysis:")
    for text, description in test_cases:
        quality_score = analyze_text_quality(text)
        logger.info(f"  {description}: {quality_score:.3f} - {'✅ Good' if quality_score > 0.6 else '❌ Poor'}")
        logger.info(f"    Text sample: '{text[:50]}{'...' if len(text) > 50 else ''}'")

def compare_analysis_methods(pdf_path: Path):
    """Compare legacy vs advanced analysis on a specific PDF."""
    logger = get_rag_logger("AnalysisComparison")
    
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return
    
    logger.info(f"\n📊 Analyzing PDF: {pdf_path.name}")
    
    try:
        with fitz.open(pdf_path) as doc:
            # Legacy analysis
            legacy_result = analyze_pdf_sample(doc, sample_pages=10)
            legacy_needs_ocr = (
                legacy_result.avg_image_area_ratio >= 0.75 or 
                legacy_result.total_chars_text_layer < 200
            )
            
            # Advanced analysis
            advanced_result = analyze_pdf_advanced(doc)
            
            logger.info("\n🔍 LEGACY ANALYSIS:")
            logger.info(f"  Pages analyzed: {legacy_result.pages}")
            logger.info(f"  Image area ratio: {legacy_result.avg_image_area_ratio:.2%}")
            logger.info(f"  Total characters: {legacy_result.total_chars_text_layer}")
            logger.info(f"  OCR needed: {'✅ YES' if legacy_needs_ocr else '❌ NO'}")
            
            logger.info("\n🚀 ADVANCED ANALYSIS:")
            logger.info(f"  Pages analyzed: {advanced_result.pages_analyzed}/{advanced_result.total_pages}")
            logger.info(f"  Text quality score: {advanced_result.text_quality_score:.3f}")
            logger.info(f"  Image area ratio: {advanced_result.avg_image_area_ratio:.2%}")
            logger.info(f"  Scanned content score: {advanced_result.scanned_content_score:.3f}")
            logger.info(f"  Text density score: {advanced_result.text_density_score:.3f}")
            logger.info(f"  Confidence score: {advanced_result.confidence_score:.3f}")
            logger.info(f"  Total characters: {advanced_result.total_chars_text_layer}")
            logger.info(f"  OCR needed: {'✅ YES' if advanced_result.needs_ocr else '❌ NO'}")
            
            if advanced_result.needs_ocr:
                reasons = advanced_result.analysis_details.get('needs_ocr_reasons', [])
                logger.info(f"  OCR reasons: {', '.join(reasons)}")
            
            # Show agreement/disagreement
            if legacy_needs_ocr == advanced_result.needs_ocr:
                logger.info(f"\n✅ AGREEMENT: Both methods agree on OCR decision")
            else:
                logger.info(f"\n⚠️  DISAGREEMENT: Methods disagree on OCR decision")
                logger.info(f"   Legacy: {'OCR' if legacy_needs_ocr else 'Direct'}")
                logger.info(f"   Advanced: {'OCR' if advanced_result.needs_ocr else 'Direct'}")
                logger.info(f"   Advanced analysis likely more accurate due to text quality assessment")
                
    except Exception as e:
        logger.error(f"Error analyzing PDF: {e}")

def find_sample_pdfs(search_dir: Path) -> list[Path]:
    """Find sample PDFs for testing."""
    if not search_dir.exists():
        return []
    
    pdfs = list(search_dir.glob("**/*.pdf"))[:5]  # Limit to 5 PDFs for testing
    return pdfs

def main():
    """Main test function."""
    logger = get_rag_logger("OCRDetectionTest")
    
    logger.info("🧪 IMPROVED OCR DETECTION TEST")
    logger.info("=" * 50)
    
    # Test text quality analysis
    test_text_quality_analysis()
    
    # Find sample PDFs
    search_paths = [
        Path("C:/Users/awun8/Documents/SCHOOL/COMPILATION"),
        Path("data/textbooks"),
        Path("."),
    ]
    
    sample_pdfs = []
    for search_path in search_paths:
        if search_path.exists():
            sample_pdfs.extend(find_sample_pdfs(search_path))
            if sample_pdfs:
                break
    
    if not sample_pdfs:
        logger.warning("\n⚠️  No sample PDFs found for comparison testing")
        logger.info("To test with actual PDFs, place some PDF files in the current directory")
        return
    
    logger.info(f"\n📚 Found {len(sample_pdfs)} sample PDFs for testing")
    
    # Compare analysis methods on sample PDFs
    for pdf_path in sample_pdfs[:3]:  # Test first 3 PDFs
        compare_analysis_methods(pdf_path)
    
    logger.info("\n🎉 Test completed!")
    logger.info("\nKey improvements in the advanced analysis:")
    logger.info("  ✅ Text quality assessment (readability, structure, artifacts)")
    logger.info("  ✅ Intelligent page sampling (representative across document)")
    logger.info("  ✅ Scanned content detection (vs embedded images)")
    logger.info("  ✅ Text density analysis (coverage per page)")
    logger.info("  ✅ Multi-factor decision making with confidence scores")
    logger.info("  ✅ Detailed logging for debugging OCR decisions")

if __name__ == "__main__":
    main()