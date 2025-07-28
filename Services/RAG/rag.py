import logging
from contextlib import nullcontext
from pathlib import Path
import re
import json
from typing import Dict, List
import google.generativeai as genAI
import chromadb
import fitz

from ...DataModels.file_data_model import FileDataModel
from ..UtilityTools.Caching.cache import Cache

GOOGLE_API_KEY = "AIzaSyB1qxAZA6G327lxiaI8pwkFKYRe1JDRz0o"
OCR_MODEL = "gemini-2.5-flash-lite"
ACCEPTABLE_TEXT_PERCENTAGE = 0.85

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RAG:
    
    def __init__(self):
        self.cache = Cache(cache_file="pdf_cache.json")
        self.gemini = genAI.configure(
            api_key=GOOGLE_API_KEY
        )
        # Add geniAI config
        genAI.configure(api_key=GOOGLE_API_KEY)
        self.geniAi_config = {
            "temperature": 0.1,
            "max_output_tokens": 4000,
            "top_p": 0.8,
            "safety_settings": [
                {
                    "category": "HARM_CATEGORY_DEROGATORY",
                    "threshold": "BLOCK_LOW_AND_ABOVE"
                }
            ]
        }
        self.current_docs = None
        self.file_data: FileDataModel = None


    def _open_pdf(self, file_path: str):
        """Open a PDF file and return the document object."""
        try:
            return fitz.open(str(file_path))
        except Exception as e:
            logging.error(f"Failed to open PDF '{file_path}': {e}")
            raise
    
    def _is_image_focused(self, file, text_threshold = 100) -> bool:
        """Determines if a PDF likely requires OCR."""
        logging.info(f"Analyzing text content of '{file}' to determine if OCR is needed.")
        
        # Check cache first
        cache_key = f"analysis_{Path(file).stem}"
        # try:
        #     cached_data = self.cache.read_cache()
        #     if cache_key in cached_data:
        #         logging.info(f"Using cached analysis for {file}")
        #         return cached_data[cache_key]
        # except Exception as e:
        #     logging.warning(f"Cache read failed for analysis: {e}")
        
        try:
            total_pages = len(self.current_docs)
            if total_pages == 0:
                logging.warning(f"PDF '{file}' has no pages.")
                result = False
            else:
                low_text_pages = 0
                page_texts = []
                
                for page_num, page in enumerate(self.current_docs):
                    text = page.get_text().strip()
                    page_texts.append({
                        'page': page_num + 1,
                        'text_length': len(text),
                        'text': text  # Store first 200 chars for debugging
                    })
                    if len(text) < text_threshold:
                        low_text_pages += 1

                ratio = low_text_pages / total_pages
                result = ratio >= ACCEPTABLE_TEXT_PERCENTAGE
                
                logging.info(
                    f"PDF '{file}' has {low_text_pages}/{total_pages} pages with less than {text_threshold} characters of text (Ratio: {ratio:.2f}). Page threshold is {ACCEPTABLE_TEXT_PERCENTAGE}.")
                
                # Cache the analysis results
                try:
                    analysis_data = {
                        "department": self.file_data.department,
                        "course_code": self.file_data.course_code,
                        "level": self.file_data.level,
                        "semester": self.file_data.semester,
                        "type": self.file_data.type,
                        "file-path": file,
                        'requires_ocr': result,
                        'total_pages': total_pages,
                        'low_text_pages': low_text_pages,
                        'ratio': ratio,
                        'page_analysis': page_texts,
                        'timestamp': str(Path(file).stat().st_mtime)
                    }
                    self.cache.update_cache(f"{cache_key}", analysis_data)
                    logging.info(f"Successfully updated data {analysis_data}")
                except Exception as e:
                    logging.warning(f"Failed to cache analysis results: {e}")
            
            return result
            
        except Exception as e:
            logging.error(f"Could not analyze PDF '{file}' for text content: {e}")
            return False


    def _extract_text_from_pdf(self, file: str):
        """Extract text directly from PDF pages with caching."""
        cache_key = f"direct_text_{Path(file).stem}"
        try:
            cached_data = self.cache.read_cache()
            if cache_key in cached_data:
                logging.info(f"Using cached direct text extraction for {file}")
                return cached_data[cache_key]
        except Exception as e:
            logging.warning(f"Cache read failed for direct text extraction: {e}")
        
        try:
            pages_text = []
            page_details = []
            
            for index, page in enumerate(self.current_docs):
                text = page.get_text()
                pages_text.append(text)
                
                # Store page-level details
                page_details.append({
                    'page': index + 1,
                    'text_length': len(text),
                    'text_preview': text[:100] + "..." if len(text) > 100 else text
                })
            
            # Cache the extracted text and details
            try:
                self.cache.update_cache(cache_key, pages_text)
                extraction_data = {
                    "file_path": f"{file}",
                    "department": self.file_data.department,
                    "course_code": self.file_data.course_code,
                    "level": self.file_data.level,
                    "semester": self.file_data.semester,
                    "type": self.file_data.type,
                    'pages': pages_text,
                    'page_details': page_details,
                    'total_pages': len(pages_text),
                    'total_chars': sum(len(text) for text in pages_text),
                    'method': 'direct_extraction',
                    'timestamp': str(Path(file).stat().st_mtime)
                }
                self.cache.update_cache(f"{cache_key}_details", extraction_data)
            except Exception as e:
                logging.warning(f"Failed to cache direct text extraction: {e}")
            
            return pages_text
            
        except Exception as e:
            logging.error(f"An error occurred extracting text from pdf: {e}")
            return []

    def _validate_latex_expressions(self, text: str) -> str:
        """Validate and correct common LaTeX formatting issues in extracted text."""
        # Common corrections for engineering LaTeX
        corrections = {
            r'\(\s*([^\)]+)\s*\)': r'$\1$',  # Fix parenthesis to dollar signs
            r'\[\s*([^\]]+)\s*\]': r'$$\1$$',  # Fix brackets to double dollars
            r'\\frac\{([^{}]+)\}\{([^{}]+)\}': r'\\frac{\1}{\2}',  # Fix frac formatting
            r'\\sum_\{([^{}]+)\}': r'\\sum_{\1}',  # Fix sum formatting
            r'\\int_\{([^{}]+)\}': r'\\int_{\1}',  # Fix integral formatting
        }
        
        for pattern, replacement in corrections.items():
            text = re.sub(pattern, replacement, text)
        
        return text

    def _extract_engineering_content(self, content_type: str, text: str) -> dict:
        """Parse and structure extracted engineering content."""
        structured_content = {
            'type': content_type,
            'raw_text': text,
            'equations': [],
            'tables': [],
            'diagrams': [],
            'code_snippets': []
        }
        
        # Extract equations
        equation_pattern = r'\$([^$]+)\$|\$\$([^$]+)\$\$'
        equations = re.findall(equation_pattern, text)
        structured_content['equations'] = [eq[0] if eq[0] else eq[1] for eq in equations]
        
        # Extract tables (markdown format)
        table_pattern = r'\|(.+)\|[\r\n]+\|[-:\| ]+\|[\r\n]+((?:\|.+\|[\r\n]*)+)'
        tables = re.findall(table_pattern, text)
        structured_content['tables'] = [{'headers': t[0], 'rows': t[1]} for t in tables]
        
        # Extract code snippets
        code_pattern = r'```(\w+)?\n(.*?)```'
        code_snippets = re.findall(code_pattern, text, re.DOTALL)
        structured_content['code_snippets'] = [{'language': lang or 'text', 'code': code} for lang, code in code_snippets]
        
        return structured_content

    # Python
    def _ocr_text_extraction(self, file: str):
        """Enhanced OCR extraction with page-level storage and unified caching."""
        cache_key = f"{Path(file).stem}"
        self.file_data = self._get_data_from_path(file)

        # Ensure self.current_docs is initialized
        if self.current_docs is None:
            self.current_docs = self._open_pdf(file)

        # Check if OCR results already cached
        try:
            cached_data = self.cache.read_cache()
            if cache_key in cached_data and 'pages' in cached_data[cache_key]:
                pages = cached_data[cache_key]['pages']
                if pages and all(str(i + 1) in pages for i in range(len(self.current_docs))):
                    ocr_pages = [pages[str(i + 1)]['text'] for i in range(len(self.current_docs))]
                    if all('[ERROR:' not in text for text in ocr_pages):
                        logging.info(f"Using cached OCR results for {file}")
                        return "\n\n--- PAGE BREAK ---\n\n".join(ocr_pages)
        except Exception as e:
            logging.warning(f"Cache check failed for OCR: {e}")

        try:
            gemini_ocr = genAI.GenerativeModel(OCR_MODEL)
            prompt = """
            You are an expert engineering document OCR system. Extract all content from this engineering PDF page with high precision.

            **EXTRACTION REQUIREMENTS:**
            - Use LaTeX for all mathematical expressions: $E=mc^2$ or $$\\int f(x)dx$$
            - Include Greek letters: $\\alpha, \\beta, \\Omega, \\mu$
            - Format tables as markdown with units
            - Extract circuit components: R=10kΩ, C=100μF
            - Preserve code snippets with proper formatting
            - Maintain technical accuracy and units

            **OUTPUT FORMAT:**
            Return the complete extracted text for this single page.
            """

            images = []
            logging.info("Extracting images for OCR processing")

            # Extract images for OCR
            for index, page in enumerate(self.current_docs):
                pix = page.get_pixmap(dpi=400)
                img_data = pix.tobytes("png")
                images.append({"mime_type": "image/png", "data": img_data})
                logging.info(f"Extracted page {index + 1} for OCR")

            logging.info(f"Total images extracted: {len(images)}")

            pages_data = {}

            for index, img in enumerate(images):
                page_num = str(index + 1)
                try:
                    try:
                        response = gemini_ocr.generate_content(
                            contents=[prompt, img],
                            generation_config=genAI.types.GenerationConfig(
                                temperature=0.1,
                                max_output_tokens=4000,
                                top_p=0.8
                            ),
                        )
                        # print(response.text)
                        if not response.candidates:
                            raise ValueError("No response generated from the model")
                    except Exception as e:
                        logging.error(f"Generation failed: {str(e)}")
                        raise

                    text = response.text.strip()
                    text = self._validate_latex_expressions(text)

                    print(text)

                    # Structure page data
                    pages_data[page_num] = {
                        "department": f"{self.file_data.department}",
                        "course_code": self.file_data.course_code,
                        "level": self.file_data.level,
                        "semester": self.file_data.semester,
                        "type": self.file_data.type,
                        "file_path": f"{file}",
                        'text': text,
                        'text_length': len(text),
                        'text_preview': text[:100] + "..." if len(text) > 100 else text,
                        'structured_content': self._extract_engineering_content('ocr', text),
                        'ocr_confidence': 'high'  # Placeholder for future confidence scoring
                    }
                    # print("#" * 60)
                    # print(pages_data[page_num])
                    # print("#" * 60)
                    logging.info(f"OCR completed for page {page_num}")

                except Exception as e:
                    logging.error(f"OCR failed for page {page_num}: {str(e)}")
                    pages_data[page_num] = {
                        'text': f"[ERROR: Failed to process page {page_num}]",
                        'text_length': 0,
                        'text_preview': "[ERROR]",
                        'structured_content': {'type': 'error', 'equations': [], 'tables': [], 'diagrams': []},
                        'ocr_confidence': 'failed'
                    }

            # Cache results in unified structure
            try:
                cached_data = self.cache.read_cache()
                if cache_key not in cached_data:
                    cached_data[cache_key] = {'metadata': {}, 'pages': {}}

                cached_data[cache_key]['pages'].update(pages_data)
                cached_data[cache_key]['metadata'].update({
                    'method': 'ocr',
                    'processing_timestamp': str(Path(file).stat().st_mtime),
                    'total_chars': sum(len(page['text']) for page in pages_data.values()),
                    'ocr_pages_processed': len(pages_data)
                })

                self.cache.update_cache(cache_key, cached_data[cache_key])
            except Exception as e:
                logging.warning(f"Failed to cache OCR results: {e}")

            final_text = "\n\n--- PAGE BREAK ---\n\n".join([pages_data[str(i + 1)]['text'] for i in range(len(images))])
            logging.info("Enhanced OCR processing complete")
            return final_text

        except Exception as e:
            logging.error(f"OCR failed for '{file}': {str(e)}")
            return None

    def get_files_from_directory(self, folder_path: str):
        """Walks through a directory and processes all PDF files found."""
        root_dir = Path(folder_path)
        if not root_dir.is_dir():
            logging.error(f"Directory not found: '{folder_path}'")
            return []

        logging.info(f"Starting PDF processing in directory: {root_dir}")
        pdf_files = list(root_dir.rglob("*.pdf"))
        logging.info(f"Found {len(pdf_files)} PDF files to process.")
        
        processed_files = []
        for file_path in pdf_files:
            print("%" * 80)
            print(file_path)
            print(str(file_path))
            print("%" * 80)

            try:
                self.file_data = self._get_data_from_path(file_path)

                logging.info(f"Processing: {file_path}")
                self.current_docs = fitz.open(str(file_path))
                
                # Determine if OCR is needed
                requires_ocr = self._is_image_focused(str(file_path))

                if requires_ocr:
                    logging.info(f"Using OCR for {file_path}")
                    extracted_text = self._ocr_text_extraction(str(file_path))
                else:
                    logging.info(f"Using direct text extraction for {file_path}")
                    extracted_text = "\n".join(self._extract_text_from_pdf(str(file_path)))

                
                if extracted_text:
                    processed_files.append({
                        "department": self.file_data.department,
                        "course_code": self.file_data.course_code,
                        "level": self.file_data.level,
                        "semester": self.file_data.semester,
                        "type": self.file_data.type,
                        'file': str(file_path),
                        'text': extracted_text,
                        'method': 'ocr' if requires_ocr else 'direct'
                    })
                    
            except Exception as e:
                logging.error(f"Failed to process {file_path}: {e}")
                
        logging.info(f"Successfully processed {len(processed_files)} PDF files")
        return processed_files

    def get_page_text(self, file_path: str, page_number: int) -> str:
        """Get text from a specific page of a PDF."""
        try:
            cache_key = f"{Path(file_path).stem}_metadata"
            cached_data = self.cache.read_cache()
            
            if cache_key in cached_data and 'pages' in cached_data[cache_key]:
                page_str = str(page_number)
                if page_str in cached_data[cache_key]['pages']:
                    return cached_data[cache_key]['pages'][page_str]['text']
            
            # If not cached, process the file
            self.current_docs = fitz.open(str(file_path))
            if page_number <= len(self.current_docs):
                if self._is_image_focused(str(file_path)):
                    self._ocr_text_extraction(str(file_path))
                else:
                    self._extract_text_from_pdf(str(file_path))
                
                # Try again after processing
                cached_data = self.cache.read_cache()
                if cache_key in cached_data and 'pages' in cached_data[cache_key]:
                    page_str = str(page_number)
                    if page_str in cached_data[cache_key]['pages']:
                        return cached_data[cache_key]['pages'][page_str]['text']
            
            return f"Page {page_number} not found"
            
        except Exception as e:
            logging.error(f"Failed to get page {page_number} from {file_path}: {e}")
            return f"Error retrieving page {page_number}"

    def get_page_range(self, file_path: str, start_page: int, end_page: int) -> dict:
        """Get text from a range of pages."""
        try:
            pages_text = {}
            for page_num in range(start_page, end_page + 1):
                pages_text[page_num] = self.get_page_text(file_path, page_num)
            return pages_text
        except Exception as e:
            logging.error(f"Failed to get pages {start_page}-{end_page} from {file_path}: {e}")
            return {}

    def get_file_metadata(self, file_path: str) -> dict:
        """Get metadata for a processed file."""
        try:
            cache_key = f"{Path(file_path).stem}_metadata"
            cached_data = self.cache.read_cache()
            
            if cache_key in cached_data:
                return cached_data[cache_key].get('metadata', {})
            
            return {}
        except Exception as e:
            logging.error(f"Failed to get metadata for {file_path}: {e}")
            return {}

    def search_pages(self, file_path: str, search_term: str) -> dict:
        """Search for text across all pages of a PDF."""
        try:
            cache_key = f"{Path(file_path).stem}_metadata"
            cached_data = self.cache.read_cache()
            
            if cache_key not in cached_data or 'pages' not in cached_data[cache_key]:
                return {}
            
            results = {}
            pages = cached_data[cache_key]['pages']
            
            for page_num, page_data in pages.items():
                text = page_data.get('text', '')
                if search_term.lower() in text.lower():
                    # Find all occurrences with context
                    import re
                    matches = []
                    for match in re.finditer(search_term, text, re.IGNORECASE):
                        start = max(0, match.start() - 50)
                        end = min(len(text), match.end() + 50)
                        context = text[start:end].strip()
                        matches.append({
                            'position': match.start(),
                            'context': context
                        })
                    
                    results[int(page_num)] = {
                        'matches': len(matches),
                        'contexts': matches
                    }
            
            return results
            
        except Exception as e:
            logging.error(f"Failed to search {file_path} for '{search_term}': {e}")
            return {}

    def get_all_pages(self, file_path: str) -> dict:
        """Get all pages with their text and metadata."""
        try:
            cache_key = f"{Path(file_path).stem}_metadata"
            cached_data = self.cache.read_cache()
            
            if cache_key in cached_data:
                return cached_data[cache_key]
            
            return {}
        except Exception as e:
            logging.error(f"Failed to get all pages for {file_path}: {e}")
            return {}

    def migrate_old_cache(self):
        """Migrate data from old cache files to new unified format."""
        try:
            old_text_cache = Cache(cache_file="Text_Storage_Cache.json")
            old_processing_cache = Cache(cache_file="Processing_Steps_Cache.json")
            
            # Read old data
            text_data = old_text_cache.read_cache()
            processing_data = old_processing_cache.read_cache()
            
            if not text_data and not processing_data:
                logging.info("No old cache data to migrate")
                return
            
            # Create unified structure
            unified_data = {}
            
            # Migrate processing data
            for key, value in processing_data.items():
                if key.startswith('analysis_') and not key.endswith('_details'):
                    file_stem = key.replace('analysis_', '')
                    if file_stem not in unified_data:
                        unified_data[file_stem] = {'metadata': {}, 'pages': {}}
                    unified_data[file_stem]['metadata']['requires_ocr'] = value
            
            # Migrate text data
            for key, value in text_data.items():
                if key.startswith('structured_'):
                    file_stem = key.replace('structured_', '')
                    if file_stem not in unified_data:
                        unified_data[file_stem] = {'metadata': {}, 'pages': {}}
                    
                    # Convert structured results to new format
                    for i, structured_result in enumerate(value):
                        page_num = str(i + 1)
                        unified_data[file_stem]['pages'][page_num] = {
                            'text': structured_result.get('raw_text', ''),
                            'structured_content': structured_result,
                            'text_length': len(structured_result.get('raw_text', ''))
                        }
            
            # Save unified data
            for file_stem, data in unified_data.items():
                self.cache.update_cache(f"{file_stem}_metadata", data)
            
            logging.info(f"Migrated {len(unified_data)} files to new cache format")
            
        except Exception as e:
            logging.error(f"Migration failed: {e}")


    def _get_data_from_path(self, file_path: str) -> FileDataModel:
        file_path = f"{file_path}".split("/")
        for index, val in enumerate(file_path):
            if "textbooks" == val:
                file_path = file_path[index:]
            elif "past-questions" == val:
                file_path = file_path[index:]

        print(file_path)

        type = file_path[0]
        level = file_path[-5]
        department = file_path[-4]
        semester = file_path[-3]
        course_code = file_path[-2]
        file_name = file_path[-1]

        print(type)
        print(f"{semester} Semester")
        print(f"Course Code: {course_code}")
        if "AAE" == department:
            print("AAE")
        elif "BME" == department:
            print("BME")
        elif "CHE" == department:
            print("CHE")
        elif "COE" == department:
            print("COE")
        elif "CVL" == department:
            print(department)
        elif "EEE" == department:
            print("EEE")
        elif "MME" == department:
            print("MME")
        elif "MCT" == department:
            print("MCT")
        elif "PTE" == department:
            print("PTE")
        elif "100" == department:
            print("!00 Level Student")
        else:
            print("None")

        data = FileDataModel(
            file_path="/".join(file_path) if isinstance(file_path, list) else file_path,
            type=type,
            level=level,
            department=department,
            semester=semester,
            course_code=course_code
        )

        self.file_data = data

        # print(data.model_dump())

        return data



if __name__ == '__main__':
    from config import load_config
    config = load_config()
    rag = RAG()

    # Test the enhanced OCR system
    test_directory = config.test_directory
    processed_files = rag.get_files_from_directory(test_directory)

    if processed_files:
        print(f"\nProcessed {len(processed_files)} files successfully")
        for file_info in processed_files[:3]:  # Show first 3 files
            print(f"\nFile: {file_info['file']}")
            print(f"Method: {file_info['method']}")
            print(f"Text length: {len(file_info['text'])} characters")
            print(f"Text: {file_info['text'][:100]}...")  # Show first 100 characters
    else:
        print("No files processed successfully.")

    # Example OCR call
    # ocr = rag.ocr_text_extraction(config.sample_pdf_path)
    # print(ocr)