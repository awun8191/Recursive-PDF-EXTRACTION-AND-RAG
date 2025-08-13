"""Utility for ingesting PDFs into ChromaDB with optional OCR."""

import os
import fitz  # PyMuPDF
from pathlib import Path
import google.generativeai as genai
from Services import GeminiService
from DataModels.ocr_data_model import OCRData
import chromadb
from dotenv import load_dotenv
import re
import logging
import json
import argparse
from config import load_config
from UtilityTools.Caching.cache import Cache
from Services.RAG.helpers import is_image_focused
from Services.Gemini.gemini_api_keys import GeminiApiKeys
from Services.Gemini.api_key_manager import ApiKeyManager

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
logging.info("Attempting to load environment variables from .env file.")
config = load_config()

# Will be initialised once the API key is confirmed available
gemini_service: GeminiService | None = None

api_keys = GeminiApiKeys().get_keys()
if not api_keys:
    logging.error("CRITICAL: No Gemini API keys found. Script cannot proceed.")
else:
    logging.info(f"Successfully loaded {len(api_keys)} Gemini API keys.")
    api_key_manager = ApiKeyManager(api_keys)
    gemini_service = GeminiService(api_keys, api_key_manager=api_key_manager)

# Model used when creating text embeddings for the ChromaDB documents
EMBED_MODEL = "models/embedding-001"
# Directory where ChromaDB will persist its data
CHROMA_PATH = "./chromadb_storage"
# Metadata about processed PDF collections is stored here
COLLECTIONS_JSON_PATH = "./collections.json"

# --- ChromaDB Client ---
chroma_client = None
if api_keys:
    try:
        logging.info(f"Initializing ChromaDB client at path: {CHROMA_PATH}")
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        logging.info("ChromaDB client initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize ChromaDB client: {e}")

# --- Core Functions ---

def get_cache_key(pdf_path: str) -> str:
    """Generate a cache key based on file path and modification time."""
    mod_time = os.path.getmtime(pdf_path)
    return f"{pdf_path}:{mod_time}"

def pdf_needs_ocr(pdf_path: str, image_ratio_threshold: float = 0.85) -> bool:
    """Determine if a PDF requires OCR."""
    cache = Cache("pdf_cache.json")
    cache_key = get_cache_key(pdf_path)
    cached_data = cache.read_cache()

    if cache_key in cached_data and "needs_ocr" in cached_data[cache_key]:
        logging.info(f"Found cached OCR requirement for '{pdf_path}'.")
        return cached_data[cache_key]["needs_ocr"]

    try:
        result = is_image_focused(pdf_path, text_threshold=100, cache=cache)
        if result:
            logging.info(f"'{pdf_path}' likely requires OCR.")
        else:
            logging.info(f"'{pdf_path}' can be processed with direct text extraction.")

        # Update cache
        if cache_key not in cached_data:
            cached_data[cache_key] = {}
        cached_data[cache_key]["needs_ocr"] = result
        cache.write_cache(cached_data)

        return result
    except Exception as e:
        logging.error(f"Could not analyze PDF '{pdf_path}' for image content: {e}")
        return False

def extract_text_from_pdf(pdf_path: str) -> list[str]:
    """Extract text from a standard, text-based PDF file."""
    cache = Cache("pdf_cache.json")
    cache_key = get_cache_key(pdf_path)
    cached_data = cache.read_cache()

    if cache_key in cached_data and "pages" in cached_data[cache_key]:
        logging.info(f"Found cached text for '{pdf_path}'.")
        return cached_data[cache_key]["pages"]

    logging.info(f"Extracting text directly from '{pdf_path}'.")
    try:
        doc = fitz.open(pdf_path)
        pages = [page.get_text() for page in doc]
        logging.info(f"Successfully extracted text from {len(pages)} pages in '{pdf_path}'.")

        # Update cache
        if cache_key not in cached_data:
            cached_data[cache_key] = {}
        cached_data[cache_key]["pages"] = pages
        cache.write_cache(cached_data)

        return pages
    except Exception as e:
        logging.error(f"Failed to extract text from '{pdf_path}': {e}")
        return []

def ocr_pdf_with_gemini(pdf_path: str) -> list[str]:
    """Performs OCR on a PDF using :class:`GeminiService`.

    The Gemini response for each page is validated against
    :class:`~DataModels.ocr_data_model.OCRData`. Extracted text from all pages is
    returned as a single string separated by ``--- PAGE BREAK ---`` markers.
    """
    cache = Cache("pdf_cache.json")
    cache_key = get_cache_key(pdf_path)
    cached_data = cache.read_cache()

    if cache_key in cached_data and "pages" in cached_data[cache_key]:
        logging.info(f"Found cached OCR text for '{pdf_path}'.")
        return cached_data[cache_key]["pages"]

    print("Starting OCR")
    if gemini_service is None:
        logging.error("Gemini service not initialized.")
        return []
    try:
        doc = fitz.open(pdf_path)
        pages_text: list[str] = []
        for page in doc:
            img = {
                "mime_type": "image/png",
                "data": page.get_pixmap(dpi=400).tobytes("png"),
            }
            ocr_result: OCRData = gemini_service.ocr([img])
            pages_text.append(ocr_result.text)

        # Update cache
        if cache_key not in cached_data:
            cached_data[cache_key] = {}
        cached_data[cache_key]["pages"] = pages_text
        cache.write_cache(cached_data)

        return pages_text
    except Exception as e:
        print("*" * 60)
        logging.error(f"Failed during OCR for PDF '{pdf_path}': {e}")
        print("*" * 60)
        return []

def chunk_text(pages: list[str], max_chars: int = 1000, by_paragraph: bool = False) -> list[str]:
    """Split text into chunks. If ``by_paragraph`` is True each paragraph is its own chunk."""
    text = "\n\n--- PAGE BREAK ---\n\n".join(pages)
    logging.info(
        f"Chunking text of length {len(text)} using by_paragraph={by_paragraph} and max {max_chars} characters."
    )

    # Remove page break markers that may appear from OCR
    text = re.sub(r"\n?-+ PAGE BREAK -+\n?", "\n\n", text)

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    if by_paragraph:
        logging.info(f"Created {len(paragraphs)} paragraph chunks.")
        return paragraphs

    chunks, current_chunk = [], ""
    for para in paragraphs:
        if len(current_chunk) + len(para) < max_chars:
            current_chunk += para + "\n\n"
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    logging.info(f"Created {len(chunks)} chunks.")
    return chunks

def embed_chunks(chunks: list[str]) -> list[list[float]] | None:
    """Embeds text chunks using the embedding model."""
    logging.info(f"Embedding {len(chunks)} chunks using model '{EMBED_MODEL}'.")
    try:
        result = genai.embed_content(
            model=EMBED_MODEL,
            content=chunks,
            task_type="RETRIEVAL_DOCUMENT"
        )
        embeddings = result["embedding"]
        logging.info(f"Successfully generated {len(embeddings)} embeddings.")
        return embeddings
    except Exception as e:
        logging.error(f"Failed to embed chunks: {e}")
        return None

def update_collections_json(collection_name: str, file_path: str, parent_folder: str):
    """Updates the JSON file with collection metadata."""
    logging.info(f"Updating collections JSON for collection '{collection_name}'.")
    collections_data = {}
    try:
        if os.path.exists(COLLECTIONS_JSON_PATH):
            with open(COLLECTIONS_JSON_PATH, 'r') as f:
                collections_data = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        logging.warning(f"Could not read collections JSON file, a new one will be created: {e}")

    collections_data[file_path] = {
        "collection_name": collection_name,
        "parent_folder": parent_folder
    }

    try:
        with open(COLLECTIONS_JSON_PATH, 'w') as f:
            json.dump(collections_data, f, indent=4)
        logging.info(f"Successfully updated '{COLLECTIONS_JSON_PATH}'.")
    except IOError as e:
        logging.error(f"Could not write to collections JSON file: {e}")

def process_pdf(pdf_path: Path, root_dir: Path):
    """Processes a single PDF file for ingestion into ChromaDB."""
    if not chroma_client:
        logging.error("ChromaDB client not available. Aborting PDF processing.")
        return

    relative_path = pdf_path.relative_to(root_dir).with_suffix("").as_posix()
    logging.info(f"--- Starting processing for: {relative_path} ---")

    pages = ocr_pdf_with_gemini(str(pdf_path)) if pdf_needs_ocr(str(pdf_path)) else extract_text_from_pdf(str(pdf_path))
    if not pages:
        logging.warning(f"Skipping '{relative_path}' due to text extraction failure.")
        return

    # Use paragraph-based chunking for better semantic separation
    chunks = chunk_text(pages, by_paragraph=True)
    if not chunks:
        logging.warning(f"Skipping '{relative_path}' as no text chunks were generated.")
        return

    embeddings = embed_chunks(chunks)
    if not embeddings:
        logging.warning(f"Skipping '{relative_path}' due to embedding failure.")
        return

    collection_name = re.sub(r'[\s/-]+', '_', relative_path)
    logging.info(f"Using collection name: '{collection_name}'")
    collection = chroma_client.get_or_create_collection(collection_name)

    doc_ids = [f"{relative_path}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": str(pdf_path), "chunk": i} for i in range(len(chunks))]

    try:
        logging.info(f"Adding {len(chunks)} chunks to collection '{collection_name}'.")
        collection.add(documents=chunks, ids=doc_ids, embeddings=embeddings, metadatas=metadatas)
        logging.info(f"Successfully added documents for '{relative_path}'.")
        update_collections_json(collection_name, str(pdf_path.resolve()), pdf_path.parent.name)

        # Update cache with embedding info
        cache = Cache("pdf_cache.json")
        cache_key = get_cache_key(str(pdf_path))
        cached_data = cache.read_cache()
        if cache_key not in cached_data:
            cached_data[cache_key] = {}
        cached_data[cache_key]["collection_name"] = collection_name
        cached_data[cache_key]["embedding_sneak_peek"] = embeddings[0][:5]
        cache.write_cache(cached_data)

    except Exception as e:
        logging.error(f"Failed to add documents to collection for '{relative_path}': {e}")
    logging.info(f"--- Finished processing for: {relative_path} ---")

def walk_and_process(root_dir_str: str):
    """Walks through a directory and processes all PDF files found."""
    if not chroma_client:
        logging.error("ChromaDB client not initialized. Aborting directory walk.")
        return

    root_dir = Path(root_dir_str)
    if not root_dir.is_dir():
        logging.error(f"Directory not found: '{root_dir_str}'")
        return

    progress_cache = Cache("progress_cache.json")
    last_processed = progress_cache.read_cache().get("last_processed_pdf")

    logging.info(f"Starting PDF processing in directory: {root_dir}")
    pdf_files = sorted([
        p for p in root_dir.rglob("*")
        if p.is_file() and p.suffix.lower() == ".pdf"
    ])

    start_index = 0
    if last_processed:
        try:
            start_index = pdf_files.index(Path(last_processed)) + 1
            logging.info(f"Resuming after '{last_processed}'.")
        except ValueError:
            logging.info("Last processed file not found, starting from the beginning.")

    logging.info(f"Found {len(pdf_files)} PDF files to process.")
    for i in range(start_index, len(pdf_files)):
        path = pdf_files[i]
        process_pdf(path, root_dir)
        progress_cache.update_cache("last_processed_pdf", str(path))

    logging.info("All PDF processing complete.")

def get_embeddings_by_collection(collection_name: str):
    """Retrieves all embeddings for a given collection name."""
    if not chroma_client:
        logging.error("ChromaDB client not available.")
        return None
    try:
        logging.info(f"Retrieving embeddings for collection: '{collection_name}'")
        collection = chroma_client.get_collection(name=collection_name)
        embeddings = collection.get(include=["embeddings"])["embeddings"]
        logging.info(f"Successfully retrieved {len(embeddings)} embeddings from '{collection_name}'.")
        return embeddings
    except Exception as e:
        logging.error(f"Failed to retrieve embeddings for collection '{collection_name}': {e}")
        return None

# --- Main Execution ---
if __name__ == "__main__":
    logging.info("========== Script Execution Started ==========")
    if api_keys and chroma_client:
        parser = argparse.ArgumentParser(description="Process PDFs for embeddings")
        parser.add_argument("--pdf-directory", default=config.pdf_directory, help="Directory containing PDFs")
        args = parser.parse_args()

        pdf_directory = args.pdf_directory
        if not pdf_directory:
            logging.error("PDF directory not specified.")
        else:
            logging.info(f"Target PDF directory: {pdf_directory}")
            walk_and_process(pdf_directory)

        # Example usage of the new function:
        # if os.path.exists(COLLECTIONS_JSON_PATH):
        #     with open(COLLECTIONS_JSON_PATH, 'r') as f:
        #         collections = json.load(f)
        #         if collections:
        #             first_collection_name = list(collections.keys())[0]
        #             retrieved_embeddings = get_embeddings_by_collection(first_collection_name)
        #             if retrieved_embeddings:
        #                 logging.info(f"Example retrieval: Got {len(retrieved_embeddings)} embeddings for collection '{first_collection_name}'.")

    else:
        logging.error("Script cannot run due to missing API key or ChromaDB client initialization failure.")
    logging.info("========== Script Execution Finished ==========")
    
