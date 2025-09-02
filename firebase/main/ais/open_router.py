# To run this code, you need to install the following dependencies:
# pip install openai

import os
import json
from openai import OpenAI
from openai import APIError
from typing import List, Dict
import fitz
import  chromadb
from chromadb.config import Settings


class CourseOutlineGeneration:
    def __init__(self):
        self.client = chromadb.Client(Settings(persist_directory="./chroma_db"))

    def intelligent_chunking(self, text: str, max_chunk_size_words: int = 300, overlap_words: int = 50) -> List[str]:
        """
        Intelligently chunk text by paragraphs with word-based size control and overlap.
        
        Args:
            text: The input text to be chunked
            max_chunk_size_words: Maximum number of words per chunk
            overlap_words: Number of words to overlap between chunks
            
        Returns:
            List of text chunks with minimum viable size filtering
        """
        if not text.strip():
            return []

        # Clean and normalize the text
        cleaned = text.replace("\n\n", "\n\n")
        paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]

        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            paragraph_words = paragraph.split()
            current_chunk_words = current_chunk.split() if current_chunk else []

            # Check if adding this paragraph would exceed the limit
            if len(current_chunk_words) + len(paragraph_words) <= max_chunk_size_words:
                current_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            else:
                # Save current chunk if it exists
                if current_chunk:
                    chunks.append(current_chunk.strip())

                    # Create overlap for context continuity
                    if len(current_chunk_words) >= overlap_words:
                        overlap = " ".join(current_chunk_words[-overlap_words:])
                        current_chunk = overlap + "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
                else:
                    current_chunk = paragraph

        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        # Filter out chunks that are too small to be meaningful
        filtered_chunks = [chunk for chunk in chunks if len(chunk.split()) >= 20]

        return filtered_chunks

    def pdf_text_extraction(self, path: str):
        with fitz.open(path) as doc:
            text = ""
            for page in doc:
                text += page.get_text()
        return text


    def is_image_heavy(self, path: str):
        with fitz.open(path) as doc:
            for page in doc:
                if page.get_image_count() > 0:
                    return True
        return False
        