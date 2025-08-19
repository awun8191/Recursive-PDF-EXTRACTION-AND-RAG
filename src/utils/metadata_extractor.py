"""Enhanced metadata extraction and tagging system for RAG pipeline."""

import re
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import fitz  # PyMuPDF

@dataclass
class DocumentMetadata:
    """Comprehensive document metadata."""
    # File information
    file_path: str
    file_name: str
    file_size: int
    file_hash: str
    created_date: str
    modified_date: str
    
    # PDF-specific metadata
    pdf_title: Optional[str] = None
    pdf_author: Optional[str] = None
    pdf_subject: Optional[str] = None
    pdf_creator: Optional[str] = None
    pdf_producer: Optional[str] = None
    pdf_creation_date: Optional[str] = None
    pdf_modification_date: Optional[str] = None
    
    # Content analysis
    page_count: int = 0
    total_characters: int = 0
    word_count: int = 0
    image_ratio: float = 0.0
    text_density: float = 0.0
    language: Optional[str] = None
    
    # Academic/Course metadata (extracted from path and content)
    department: Optional[str] = None
    course_code: Optional[str] = None
    course_number: Optional[str] = None
    level: Optional[str] = None
    semester: Optional[str] = None
    academic_year: Optional[str] = None
    
    # Content classification
    document_type: Optional[str] = None  # lecture, assignment, exam, textbook, etc.
    topics: List[str] = None
    keywords: List[str] = None
    
    # Processing metadata
    processing_method: str = "unknown"  # ocr, direct_text, hybrid
    ocr_confidence: Optional[float] = None
    extraction_timestamp: str = ""
    
    # Quality metrics
    readability_score: Optional[float] = None
    completeness_score: float = 0.0
    
    # Custom tags
    tags: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.topics is None:
            self.topics = []
        if self.keywords is None:
            self.keywords = []
        if self.tags is None:
            self.tags = {}
        if not self.extraction_timestamp:
            self.extraction_timestamp = datetime.now().isoformat()

class MetadataExtractor:
    """Enhanced metadata extraction with academic focus."""
    
    def __init__(self):
        # Course code/number extraction pattern (handles apostrophes)
        self.course_pattern = re.compile(r"([A-Za-z]+(?:'[A-Za-z]*)?[A-Za-z]*)\s*[-_ ]*\s*(\d{2,3})", re.IGNORECASE)
        
        # Level normalization patterns
        self.level_normalize = {
            r'(\d{3})L': r'\1',
            r'(\d{3})\s+LEVEL': r'\1',
            r'(\d{3})\s+level': r'\1',
        }
        
        # Document type patterns (simplified)
        self.document_type_patterns = {
            r'\b(lecture|notes?)\b': 'lecture',
            r'\b(assignment|homework|hw)\b': 'assignment',
            r'\b(exam|test|quiz|midterm|final)\b': 'exam',
            r'\b(textbook|book|manual)\b': 'textbook',
            r'\b(syllabus|outline)\b': 'syllabus',
            r'\b(lab|laboratory|practical)\b': 'lab',
            r'\b(project|report)\b': 'project',
            r'\b(solution|answer|key)\b': 'solution',
            r'\bpq\b': 'exam',  # Past questions
        }
        
        # Common academic keywords
        self.academic_keywords = {
            'mathematics': ['calculus', 'algebra', 'geometry', 'statistics', 'probability'],
            'engineering': ['circuit', 'design', 'analysis', 'system', 'control'],
            'computer_science': ['algorithm', 'programming', 'software', 'database', 'network'],
            'physics': ['mechanics', 'thermodynamics', 'electromagnetism', 'quantum'],
            'chemistry': ['organic', 'inorganic', 'analytical', 'physical', 'biochemistry'],
        }
    
    def extract_metadata(self, file_path: str, content: str = None, 
                        pdf_analysis: Dict = None) -> DocumentMetadata:
        """Extract comprehensive metadata from a document."""
        path = Path(file_path)
        
        # Basic file information (handle missing files gracefully)
        try:
            file_stats = path.stat()
            file_size = file_stats.st_size
            created_date = datetime.fromtimestamp(file_stats.st_ctime).isoformat()
            modified_date = datetime.fromtimestamp(file_stats.st_mtime).isoformat()
        except Exception:
            file_size = 0
            created_date = ""
            modified_date = ""
        
        file_hash = self._calculate_file_hash(file_path)
        
        metadata = DocumentMetadata(
            file_path=str(path.resolve()),
            file_name=path.name,
            file_size=file_size,
            file_hash=file_hash,
            created_date=created_date,
            modified_date=modified_date
        )
        
        # Extract PDF metadata
        if path.suffix.lower() == '.pdf':
            self._extract_pdf_metadata(file_path, metadata)
        
        # Extract path-based academic metadata
        self._extract_path_metadata(file_path, metadata)
        
        # Extract content-based metadata
        if content:
            self._extract_content_metadata(content, metadata)
        
        # Use PDF analysis if provided
        if pdf_analysis:
            self._integrate_pdf_analysis(pdf_analysis, metadata)
        
        # Generate GROUP_KEY
        group_key = self._generate_group_key(metadata)
        metadata.tags = metadata.tags or {}
        metadata.tags['group_key'] = group_key
        
        # Calculate quality scores
        self._calculate_quality_scores(metadata)
        
        return metadata
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of the file."""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()[:16]  # First 16 characters
        except Exception:
            return "unknown"
    
    def _extract_pdf_metadata(self, file_path: str, metadata: DocumentMetadata):
        """Extract PDF-specific metadata."""
        try:
            doc = fitz.open(file_path)
            pdf_meta = doc.metadata
            
            metadata.pdf_title = pdf_meta.get('title', '').strip() or None
            metadata.pdf_author = pdf_meta.get('author', '').strip() or None
            metadata.pdf_subject = pdf_meta.get('subject', '').strip() or None
            metadata.pdf_creator = pdf_meta.get('creator', '').strip() or None
            metadata.pdf_producer = pdf_meta.get('producer', '').strip() or None
            metadata.pdf_creation_date = pdf_meta.get('creationDate', '').strip() or None
            metadata.pdf_modification_date = pdf_meta.get('modDate', '').strip() or None
            
            metadata.page_count = len(doc)
            doc.close()
        except Exception as e:
            print(f"Warning: Could not extract PDF metadata from {file_path}: {e}")
    
    def _extract_path_metadata(self, file_path: str, metadata: DocumentMetadata):
        """Extract academic metadata using tail-based parsing."""
        # Split path and get segments from the end
        path_parts = Path(file_path).parts
        if len(path_parts) < 2:
            return
        
        # Tail-based parsing: [-1]=FILENAME, [-2]=COURSE_FOLDER, [-3]=SEMESTER, [-4]=LEVEL, [-5]=DEPARTMENT
        filename = path_parts[-1]
        stem = Path(filename).stem
        
        course_folder = path_parts[-2] if len(path_parts) >= 2 else ""
        semester = path_parts[-3] if len(path_parts) >= 3 else ""
        level = path_parts[-4] if len(path_parts) >= 4 else ""
        department = path_parts[-5] if len(path_parts) >= 5 else ""
        
        # Extract DEPARTMENT
        if department and re.match(r'^[A-Z]{2,4}$', department.upper()):
            metadata.department = department.upper()
            metadata.tags = metadata.tags or {}
            metadata.tags['explicit_department'] = True
        
        # Extract LEVEL and normalize
        if level:
            normalized_level = self._normalize_level(level)
            if normalized_level:
                metadata.level = normalized_level
        
        # Extract SEMESTER (use directly if "1" or "2")
        if semester in ['1', '2']:
            metadata.semester = semester
        
        # Extract COURSE_CODE & COURSE_NUMBER
        course_code, course_number = self._extract_course_info(course_folder, stem)
        if course_code:
            metadata.course_code = course_code.upper()
            metadata.department = metadata.department or course_code.upper()
        if course_number:
            metadata.course_number = course_number
            
            # Derive LEVEL from COURSE_NUMBER if LEVEL is missing
            if not metadata.level and len(course_number) >= 3:
                derived_level = course_number[0] + '00'
                if derived_level in ['100', '200', '300', '400', '500']:
                    metadata.level = derived_level
        
        # Extract CATEGORY
        category = self._extract_category(course_folder, filename)
        if category:
            metadata.tags = metadata.tags or {}
            metadata.tags['category'] = category
        
        # Store additional path info
        metadata.tags = metadata.tags or {}
        metadata.tags['filename'] = filename
        metadata.tags['stem'] = stem
        if course_folder:
            metadata.tags['course_folder'] = course_folder
    
    def _normalize_level(self, level_str: str) -> Optional[str]:
        """Normalize level variants like '300L', '300 LEVEL' to '300'."""
        if not level_str:
            return None
        
        level_str = level_str.strip()
        
        # Apply normalization patterns
        for pattern, replacement in self.level_normalize.items():
            level_str = re.sub(pattern, replacement, level_str, flags=re.IGNORECASE)
        
        # Check if it's a valid 3-digit level
        if re.match(r'^[1-5]00$', level_str):
            return level_str
        
        return None
    
    def _extract_course_info(self, course_folder: str, stem: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract course code and number strictly from the course folder segment.

        The extractor intentionally avoids parsing the filename ``stem`` to reduce
        false positives and ensure metadata is derived from the directory
        hierarchy as specified by the user.
        """
        if course_folder:
            match = self.course_pattern.search(course_folder)
            if match:
                code = match.group(1).replace("'S", "").replace("'s", "")  # Clean up possessives
                return code, match.group(2)

        # Fallback to filename stem when course folder is absent or no match found
        if not course_folder and stem:
            match = self.course_pattern.search(stem)
            if match:
                code = match.group(1).replace("'S", "").replace("'s", "")
                return code, match.group(2)

        return None, None
    
    def _extract_category(self, course_folder: str, filename: str) -> Optional[str]:
        """Extract category (GENERAL or PQ) from folder or filename."""
        # Check course folder
        if course_folder and course_folder.upper() == 'GENERAL':
            return 'GENERAL'
        
        # Check for PQ variants in folder
        if course_folder and course_folder.upper() in ['PQ', 'PQS', 'PASTQUESTIONS', 'PAST QUESTIONS']:
            return 'PQ'
        
        # Check filename for PQ
        if filename and 'PQ' in filename.upper():
            return 'PQ'
        
        return None
    
    def _generate_group_key(self, metadata: DocumentMetadata) -> str:
        """Generate GROUP_KEY following the patterns in the user's examples.
        
        Examples from user:
        1. EEE 313.pdf → GROUP_KEY=EEE-EEE-313 (when dept=code)
        2. Reservoir Eng.pdf → GROUP_KEY=PTE (when only dept available)
        3. EEE 405 NOTE.pdf → GROUP_KEY=EEE-EEE-405 (when dept=code)
        """
        # If we have general content, just use department
        if metadata.tags and metadata.tags.get('category') == 'GENERAL':
            return metadata.department or 'UNKNOWN'
        
        # If we have course number but no department or code, use number alone
        if metadata.course_number and not (metadata.department or metadata.course_code):
            return metadata.course_number
        
        # If we have department but no course code/number, return department
        if metadata.department and not metadata.course_code and not metadata.course_number:
            return metadata.department
        
        # If we have department, course code, and number
        if metadata.department and metadata.course_code and metadata.course_number:
            explicit_dept = metadata.tags.get('explicit_department', False) if metadata.tags else False
            if metadata.department == metadata.course_code and not explicit_dept:
                # Department was derived from course code; avoid duplication
                return f"{metadata.department}-{metadata.course_number}"
            return f"{metadata.department}-{metadata.course_code}-{metadata.course_number}"

        # If we have course code and number → code-num
        if metadata.course_code and metadata.course_number:
            return f"{metadata.course_code}-{metadata.course_number}"

        # If we have department and course number → dept-num
        if metadata.department and metadata.course_number:
            return f"{metadata.department}-{metadata.course_number}"

        # If we have department and course code → dept-code
        if metadata.department and metadata.course_code:
            return f"{metadata.department}-{metadata.course_code}"
        
        # Fallbacks
        if metadata.department:
            return metadata.department
        if metadata.course_code:
            return metadata.course_code
            
        return 'UNKNOWN'
    
    def _extract_content_metadata(self, content: str, metadata: DocumentMetadata):
        """Extract metadata from document content."""
        if not content:
            return
        
        # Basic content statistics
        metadata.total_characters = len(content)
        metadata.word_count = len(content.split())
        
        # Document type classification
        content_lower = content.lower()
        for pattern, doc_type in self.document_type_patterns.items():
            if re.search(pattern, content_lower, re.IGNORECASE):
                metadata.document_type = doc_type
                break
        
        # Extract course information from content (only if not found in path)
        if not metadata.course_code:
            course_code, course_number = self._extract_course_info("", content[:1000])  # Check first 1000 chars
            if course_code:
                metadata.course_code = course_code.upper()
                metadata.department = metadata.department or course_code.upper()
            if course_number:
                metadata.course_number = course_number
                
                # Derive LEVEL from COURSE_NUMBER if LEVEL is missing
                if not metadata.level and len(course_number) >= 3:
                    derived_level = course_number[0] + '00'
                    if derived_level in ['100', '200', '300', '400', '500']:
                        metadata.level = derived_level
        
        # Extract keywords based on content
        self._extract_keywords(content, metadata)
        
        # Simple language detection (basic heuristic)
        metadata.language = self._detect_language(content)
        
        # Calculate readability score (simplified)
        metadata.readability_score = self._calculate_readability(content)
    
    def _extract_keywords(self, content: str, metadata: DocumentMetadata):
        """Extract relevant keywords from content."""
        content_lower = content.lower()
        found_keywords = []
        
        # Check for academic domain keywords
        for domain, keywords in self.academic_keywords.items():
            for keyword in keywords:
                if keyword in content_lower:
                    found_keywords.append(keyword)
                    if domain not in metadata.topics:
                        metadata.topics.append(domain)
        
        # Extract capitalized terms (potential proper nouns/technical terms)
        capitalized_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
        technical_terms = [term for term in capitalized_terms if len(term) > 3 and term.count(' ') <= 2]
        
        # Limit to most frequent terms
        from collections import Counter
        term_counts = Counter(technical_terms)
        frequent_terms = [term for term, count in term_counts.most_common(10) if count > 1]
        
        metadata.keywords = list(set(found_keywords + frequent_terms))
    
    def _detect_language(self, content: str) -> str:
        """Simple language detection based on common words."""
        english_indicators = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with']
        content_lower = content.lower()
        
        english_count = sum(1 for word in english_indicators if f' {word} ' in content_lower)
        
        if english_count > 5:
            return 'english'
        else:
            return 'unknown'
    
    def _calculate_readability(self, content: str) -> float:
        """Calculate a simple readability score."""
        if not content:
            return 0.0
        
        words = content.split()
        sentences = re.split(r'[.!?]+', content)
        
        if len(sentences) == 0 or len(words) == 0:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Simple readability score (lower is easier)
        score = (avg_sentence_length * 0.5) + (avg_word_length * 2.0)
        return min(100.0, max(0.0, 100.0 - score))  # Normalize to 0-100
    
    def _integrate_pdf_analysis(self, pdf_analysis: Dict, metadata: DocumentMetadata):
        """Integrate PDF analysis results into metadata."""
        if 'pages' in pdf_analysis:
            metadata.page_count = pdf_analysis['pages']
        
        if 'avg_image_area_ratio' in pdf_analysis:
            metadata.image_ratio = pdf_analysis['avg_image_area_ratio']
        
        if 'total_chars_text_layer' in pdf_analysis:
            metadata.total_characters = pdf_analysis['total_chars_text_layer']
            if metadata.page_count > 0:
                metadata.text_density = metadata.total_characters / metadata.page_count
        
        # Determine processing method
        if metadata.image_ratio > 0.75:
            metadata.processing_method = 'ocr'
        elif metadata.image_ratio < 0.25:
            metadata.processing_method = 'direct_text'
        else:
            metadata.processing_method = 'hybrid'
    
    def _calculate_quality_scores(self, metadata: DocumentMetadata):
        """Calculate quality and completeness scores."""
        score = 0.0
        max_score = 0.0
        
        # File information completeness
        if metadata.file_hash != 'unknown':
            score += 10
        max_score += 10
        
        # PDF metadata completeness
        pdf_fields = [metadata.pdf_title, metadata.pdf_author, metadata.pdf_subject]
        score += sum(10 for field in pdf_fields if field) 
        max_score += 30
        
        # Academic metadata completeness
        academic_fields = [metadata.department, metadata.course_code, metadata.level]
        score += sum(15 for field in academic_fields if field)
        max_score += 45
        
        # Content analysis completeness
        if metadata.total_characters > 0:
            score += 10
        if metadata.keywords:
            score += 10
        if metadata.document_type:
            score += 10
        max_score += 30
        
        metadata.completeness_score = (score / max_score * 100) if max_score > 0 else 0.0
    
    def enhance_metadata_with_tags(self, metadata: DocumentMetadata, 
                                  custom_tags: Dict[str, Any] = None) -> DocumentMetadata:
        """Add custom tags and enhanced categorization."""
        if custom_tags:
            metadata.tags.update(custom_tags)
        
        # Add automatic tags based on extracted information
        auto_tags = {}
        
        if metadata.department:
            auto_tags['department'] = metadata.department
        
        if metadata.level:
            auto_tags['academic_level'] = metadata.level
            if metadata.level in ['100', '200']:
                auto_tags['difficulty'] = 'beginner'
            elif metadata.level in ['300', '400']:
                auto_tags['difficulty'] = 'intermediate'
            else:
                auto_tags['difficulty'] = 'advanced'
        
        if metadata.document_type:
            auto_tags['content_type'] = metadata.document_type
        
        if metadata.image_ratio > 0.5:
            auto_tags['visual_heavy'] = True
        
        if metadata.word_count > 5000:
            auto_tags['length'] = 'long'
        elif metadata.word_count > 1000:
            auto_tags['length'] = 'medium'
        else:
            auto_tags['length'] = 'short'
        
        if metadata.topics:
            auto_tags['domains'] = metadata.topics
        
        metadata.tags.update(auto_tags)
        return metadata
    
    def export_metadata(self, metadata: DocumentMetadata, format: str = 'json') -> str:
        """Export metadata in specified format."""
        data = asdict(metadata)
        
        if format.lower() == 'json':
            import json
            return json.dumps(data, indent=2, ensure_ascii=False)
        elif format.lower() == 'yaml':
            try:
                import yaml
                return yaml.dump(data, default_flow_style=False, allow_unicode=True)
            except ImportError:
                return "YAML export requires PyYAML package"
        else:
            return str(data)

# Convenience functions
def extract_document_metadata(file_path: str, content: str = None, 
                            pdf_analysis: Dict = None, 
                            custom_tags: Dict[str, Any] = None) -> DocumentMetadata:
    """Extract comprehensive metadata from a document."""
    extractor = MetadataExtractor()
    metadata = extractor.extract_metadata(file_path, content, pdf_analysis)
    
    if custom_tags:
        metadata = extractor.enhance_metadata_with_tags(metadata, custom_tags)
    
    return metadata

def create_metadata_summary(metadata_list: List[DocumentMetadata]) -> Dict[str, Any]:
    """Create a summary of multiple document metadata."""
    if not metadata_list:
        return {}
    
    summary = {
        'total_documents': len(metadata_list),
        'total_pages': sum(m.page_count for m in metadata_list),
        'total_words': sum(m.word_count for m in metadata_list),
        'departments': list(set(m.department for m in metadata_list if m.department)),
        'document_types': list(set(m.document_type for m in metadata_list if m.document_type)),
        'languages': list(set(m.language for m in metadata_list if m.language)),
        'processing_methods': list(set(m.processing_method for m in metadata_list)),
        'average_completeness': sum(m.completeness_score for m in metadata_list) / len(metadata_list),
        'quality_distribution': {
            'high': sum(1 for m in metadata_list if m.completeness_score >= 80),
            'medium': sum(1 for m in metadata_list if 50 <= m.completeness_score < 80),
            'low': sum(1 for m in metadata_list if m.completeness_score < 50)
        }
    }
    
    return summary