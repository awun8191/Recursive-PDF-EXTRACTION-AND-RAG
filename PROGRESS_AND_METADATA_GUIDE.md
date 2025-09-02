# Progress Tracking and Metadata Enhancement Guide

## Overview

This guide covers the new progress tracking and metadata extraction features added to the RAG pipeline. These features provide robust resume functionality and comprehensive document metadata for better organization and analysis.

## 🔄 Progress Tracking Features

### Key Capabilities
- **Resume Functionality**: Automatically resume processing from where it left off if interrupted
- **Real-time Progress Monitoring**: Track processing status of individual files
- **Session Management**: Maintain processing sessions with unique identifiers
- **Detailed Statistics**: Comprehensive metrics on processing performance
- **Error Tracking**: Monitor and log failed files with error details

### Usage Examples

#### Basic Usage with Resume
```bash
# Start processing with progress tracking
python ./src/services/RAG/convert_to_embeddings.py \
  -i "C:\path\to\pdfs" \
  --export-dir "./data/exported_data" \
  --cache-dir "./data/ocr_cache" \
  --workers 4 \
  --resume
```

#### Custom Session Management
```bash
# Start with custom session ID
python ./src/services/RAG/convert_to_embeddings.py \
  -i "C:\path\to\pdfs" \
  --session-id "my_custom_session" \
  --progress-file "./data/progress/my_progress.json" \
  --resume
```

#### Clean Up After Completion
```bash
# Process and clean up progress file when done
python ./src/services/RAG/convert_to_embeddings.py \
  -i "C:\path\to\pdfs" \
  --cleanup-session
```

### Progress File Structure

The progress file (`pipeline_progress.json`) contains:

```json
{
  "session_id": "session_1234567890",
  "start_time": "2024-01-15T10:30:00",
  "last_updated": "2024-01-15T11:45:30",
  "total_files": 1000,
  "completed_files": 750,
  "failed_files": 5,
  "skipped_files": 10,
  "total_chunks": 45000,
  "total_embeddings": 45000,
  "processing_parameters": {
    "workers": 4,
    "force_ocr": true,
    "export_include": "text+gemini-embedding"
  },
  "file_progress": {
    "/path/to/file1.pdf": {
      "status": "completed",
      "chunks_extracted": 25,
      "embeddings_generated": 25,
      "processing_time": 5.2,
      "metadata": {...}
    }
  }
}
```

## 📊 Metadata Extraction Features

### Comprehensive Metadata Collection

#### File Information
- File path, name, size, and hash
- Creation and modification dates
- Processing timestamps

#### PDF-Specific Metadata
- Title, author, subject, creator, producer
- Creation and modification dates from PDF metadata
- Page count and structure analysis

#### Academic Metadata (Auto-extracted)
- **Department**: EEE, MATH, CS, etc.
- **Course Code**: EEE, MATH, CS
- **Course Number**: 313, 101, 450
- **Level**: 100, 200, 300, 400, 500
- **Semester**: fall, spring, summer, winter
- **Academic Year**: 2024, 2023, etc.

#### Content Analysis
- Document type classification (lecture, assignment, exam, textbook, etc.)
- Word count and character count
- Language detection
- Topic extraction
- Keyword identification
- Readability scoring

#### Quality Metrics
- **Completeness Score**: 0-100% based on metadata richness
- **Processing Method**: OCR, direct text, or hybrid
- **Text Density**: Characters per page
- **Image Ratio**: Percentage of image content

### Usage Examples

#### Programmatic Metadata Extraction
```python
from src.utils.metadata_extractor import extract_document_metadata

# Extract metadata from a PDF
metadata = extract_document_metadata(
    file_path="/path/to/document.pdf",
    content="extracted text content",
    custom_tags={
        "semester": "fall_2024",
        "instructor": "Dr. Smith"
    }
)

print(f"Department: {metadata.department}")
print(f"Course: {metadata.course_code} {metadata.course_number}")
print(f"Document Type: {metadata.document_type}")
print(f"Completeness: {metadata.completeness_score:.1f}%")
```

#### Batch Metadata Analysis
```python
from src.utils.metadata_extractor import create_metadata_summary

# Analyze multiple documents
metadata_list = [extract_document_metadata(file) for file in pdf_files]
summary = create_metadata_summary(metadata_list)

print(f"Total documents: {summary['total_documents']}")
print(f"Departments: {summary['departments']}")
print(f"Document types: {summary['document_types']}")
print(f"Average completeness: {summary['average_completeness']:.1f}%")
```

### Enhanced JSONL Output

Each chunk now includes comprehensive metadata:

```json
{
  "id": "EEE_313_0",
  "text": "Chapter content...",
  "embedding": [0.1, 0.2, ...],
  "embedding_type": "gemini-embedding",
  "metadata": {
    "source_file": "/path/to/file.pdf",
    "chunk_index": 0,
    "course_key": "EEE_313",
    "document_metadata": {
      "file_hash": "abc123def456",
      "page_count": 45,
      "word_count": 12500,
      "department": "EEE",
      "course_code": "EEE",
      "course_number": "313",
      "level": "300",
      "document_type": "lecture",
      "topics": ["engineering", "signal_processing"],
      "keywords": ["filter", "frequency", "analysis"],
      "processing_method": "ocr",
      "completeness_score": 85.5,
      "tags": {
        "department": "EEE",
        "academic_level": "300",
        "difficulty": "intermediate",
        "content_type": "lecture",
        "length": "long",
        "domains": ["engineering"]
      }
    },
    "extraction_status": "success"
  }
}
```

## 🚀 Advanced Features

### Resume from Specific Point

The system automatically detects interrupted sessions:

```bash
# If you run this command and it gets interrupted...
python ./src/services/RAG/convert_to_embeddings.py -i "./pdfs" --workers 4

# Simply add --resume to continue from where it left off
python ./src/services/RAG/convert_to_embeddings.py -i "./pdfs" --workers 4 --resume
```

### Progress Monitoring

Real-time progress updates show:
- Files completed vs. total
- Processing speed (files/minute)
- Estimated time remaining
- Cache hit rates
- Error counts

### Metadata-Based Filtering

Use metadata for advanced filtering and analysis:

```python
# Filter by department
eee_docs = [doc for doc in documents if doc['metadata']['document_metadata']['department'] == 'EEE']

# Filter by document type
lectures = [doc for doc in documents if doc['metadata']['document_metadata']['document_type'] == 'lecture']

# Filter by completeness score
high_quality = [doc for doc in documents if doc['metadata']['document_metadata']['completeness_score'] > 80]
```

## 📁 File Organization

### Progress Files
- **Location**: `data/progress/pipeline_progress.json`
- **Results**: `data/progress/results_[session_id].json`
- **Automatic cleanup**: Use `--cleanup-session` flag

### Cache Integration
- **Enhanced Cache**: `data/gemini_cache/enhanced_cache.json`
- **Embeddings Cache**: `data/gemini_cache/embeddings_cache.json`
- **API Key Cache**: `data/gemini_cache/api_key_cache.json`

## 🔧 Configuration Options

### Progress Tracking Arguments
```bash
--resume                    # Resume from previous session if available
--progress-file PATH        # Custom progress file path
--session-id ID            # Custom session identifier
--cleanup-session          # Clean up progress file after completion
```

### Metadata Enhancement
Metadata extraction is automatically enabled and includes:
- Path-based academic information extraction
- Content analysis and classification
- Quality scoring and completeness metrics
- Custom tagging and categorization

## 📈 Performance Benefits

### Resume Functionality
- **Zero Waste**: Never reprocess completed files
- **Fault Tolerance**: Survive system crashes and interruptions
- **Flexible Scheduling**: Stop and resume processing as needed

### Enhanced Metadata
- **Better Organization**: Automatic categorization and tagging
- **Quality Assessment**: Identify high/low quality documents
- **Academic Structure**: Understand course organization
- **Search Enhancement**: Rich metadata for better retrieval

## 🧪 Testing

Run the comprehensive test suite:

```bash
python ./test_progress_and_metadata.py
```

This tests:
- Progress tracking functionality
- Metadata extraction accuracy
- Integration between systems
- Resume capability
- Error handling

## 🎯 Best Practices

### For Large Datasets
1. **Use Resume**: Always enable `--resume` for large processing jobs
2. **Monitor Progress**: Check progress files periodically
3. **Backup Progress**: Keep copies of progress files for critical jobs
4. **Clean Up**: Use `--cleanup-session` after successful completion

### For Metadata Quality
1. **Consistent Naming**: Use consistent file naming conventions
2. **Path Structure**: Organize files in academic hierarchy
3. **Content Quality**: Ensure PDFs have good text extraction
4. **Custom Tags**: Add domain-specific tags as needed

### For Performance
1. **Optimal Workers**: Use 4-8 workers for best performance
2. **Cache Utilization**: Let the system build up cache over time
3. **Progress Saves**: System saves progress every 10 files automatically
4. **Error Recovery**: Failed files are automatically retried on resume

## 🔍 Troubleshooting

### Common Issues

**Progress file corruption**:
```bash
# Delete corrupted progress file and restart
rm data/progress/pipeline_progress.json
python ./src/services/RAG/convert_to_embeddings.py -i "./pdfs"
```

**Metadata extraction failures**:
- Check file permissions
- Ensure PDFs are not corrupted
- Verify file paths are accessible

**Resume not working**:
- Ensure same input directory
- Check progress file exists
- Verify session compatibility

### Debug Mode

Enable detailed logging:
```bash
export PYTHONPATH="./src"
python -v ./src/services/RAG/convert_to_embeddings.py --resume
```

## 📚 API Reference

### ProgressTracker Class
```python
class ProgressTracker:
    def __init__(self, progress_file: str = None, session_id: str = None)
    def initialize_session(self, total_files: int, processing_params: Dict) -> str
    def update_file_status(self, file_path: str, status: ProcessingStatus, ...)
    def get_pending_files(self, all_files: List[str]) -> List[str]
    def get_progress_summary(self) -> Dict[str, Any]
    def can_resume(self) -> bool
    def export_results(self, export_path: str = None) -> str
```

### MetadataExtractor Class
```python
class MetadataExtractor:
    def extract_metadata(self, file_path: str, content: str = None) -> DocumentMetadata
    def enhance_metadata_with_tags(self, metadata: DocumentMetadata, custom_tags: Dict) -> DocumentMetadata
    def export_metadata(self, metadata: DocumentMetadata, format: str = 'json') -> str
```

This comprehensive system ensures robust, resumable processing with rich metadata for enhanced document understanding and organization.