import os
import fitz  # PyMuPDF
from pathlib import Path
import google.generativeai as genai
from Services import GeminiService
import chromadb
from dotenv import load_dotenv
import re
import logging
import json
import argparse
from config import load_config

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
logging.info("Attempting to load environment variables from .env file.")
config = load_config()

gemini_service: GeminiService | None = None

GOOGLE_API_KEY = "AIzaSyB1qxAZA6G327lxiaI8pwkFKYRe1JDRz0o"
if not GOOGLE_API_KEY:
    logging.error("CRITICAL: GOOGLE_API_KEY not found in environment variables. Script cannot proceed.")
else:
    logging.info("Successfully loaded GOOGLE_API_KEY.")
    genai.configure(api_key=GOOGLE_API_KEY)
    logging.info("Google Generative AI SDK configured.")
    gemini_service = GeminiService()

EMBED_MODEL = "models/embedding-001"
CHROMA_PATH = "./chromadb_storage"
COLLECTIONS_JSON_PATH = "./collections.json"

# --- ChromaDB Client ---
chroma_client = None
if GOOGLE_API_KEY:
    try:
        logging.info(f"Initializing ChromaDB client at path: {CHROMA_PATH}")
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        logging.info("ChromaDB client initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize ChromaDB client: {e}")

# --- Core Functions ---

def pdf_needs_ocr(pdf_path: str, text_threshold: int = 50, page_threshold: float = 0.9) -> bool:
    """Determines if a PDF likely requires OCR."""
# The reason for the text threshold of 50 is to account for documents that are short like presentations
    logging.info(f"Analyzing text content of '{pdf_path}' to determine if OCR is needed.")
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        if total_pages == 0:
            logging.warning(f"PDF '{pdf_path}' has no pages.")
            return False

        low_text_pages = 0
        for page_num, page in enumerate(doc):
            text = page.get_text().strip()
            if len(text) < text_threshold:
                low_text_pages += 1

        ratio = low_text_pages / total_pages
        logging.info(f"PDF '{pdf_path}' has {low_text_pages}/{total_pages} pages with less than {text_threshold} characters of text (Ratio: {ratio:.2f}). Page threshold is {page_threshold}.")

        return ratio >= page_threshold
    except Exception as e:
        logging.error(f"Could not analyze PDF '{pdf_path}' for text content: {e}")
        return False

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a standard PDF."""
    logging.info(f"Extracting text directly from '{pdf_path}'.")
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text() for page in doc])
        logging.info(f"Successfully extracted {len(text)} characters from '{pdf_path}'.")
        return text
    except Exception as e:
        logging.error(f"Failed to extract text from '{pdf_path}': {e}")
        return ""

def ocr_pdf_with_gemini(pdf_path: str) -> str:
    """Performs OCR on a PDF using :class:`GeminiService`."""
    if gemini_service is None:
        logging.error("Gemini service not initialized.")
        return ""
    logging.info(f"Performing OCR on '{pdf_path}' using GeminiService.")
    try:
        doc = fitz.open(pdf_path)
        images = [
            {"mime_type": "image/png", "data": page.get_pixmap().tobytes("png")}
            for page in doc
        ]
        prompt = (
            "Extract all text from this document image and return JSON as {\"text\": \"<content>\"}."
        )
        result = gemini_service.ocr(images, prompt=prompt)
        return result.get("text", "") if isinstance(result, dict) else ""
    except Exception as e:
        logging.error(f"Failed during OCR for PDF '{pdf_path}': {e}")
        return ""

def chunk_text(text: str, max_chars: int = 1000) -> list[str]:
    """Splits text into semantic chunks."""
    logging.info(f"Chunking text of length {len(text)} into chunks of max {max_chars} characters.")
    chunks, current_chunk = [], ""
    for para in re.split(r'\n\s*\n', text):
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

    collections_data[collection_name] = {
        "absolute_path": file_path,
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

    text = ocr_pdf_with_gemini(str(pdf_path)) if pdf_needs_ocr(str(pdf_path)) else extract_text_from_pdf(str(pdf_path))
    if not text:
        logging.warning(f"Skipping '{relative_path}' due to text extraction failure.")
        return

    chunks = chunk_text(text)
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

    logging.info(f"Starting PDF processing in directory: {root_dir}")
    pdf_files = list(root_dir.rglob("*.pdf"))
    logging.info(f"Found {len(pdf_files)} PDF files to process.")
    for path in pdf_files:
        process_pdf(path, root_dir)
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
    if GOOGLE_API_KEY and chroma_client:
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