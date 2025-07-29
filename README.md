# Recursive-PDF-EXTRACTION-AND-RAG

## Configuration

Paths used by the scripts are now configurable via environment variables or a
`config.json` file in the repository root. Environment variables override values
from the JSON file.

Supported options:

| Variable | Description |
| -------- | ----------- |
| `FIRESTORE_SERVICE_ACCOUNT` | Path to the Firestore service account JSON file |
| `PDF_DIRECTORY` | Default directory containing PDFs for `main.py` |
| `PROCESSED_COURSES_CACHE` | Location of the processed courses cache |
| `COURSES_JSON_PATH` | Path to the courses JSON used by sample data loader |
| `TRANSFER_SERVICE_ACCOUNT` | Service account for the transfer utility |
| `RECEIVING_SERVICE_ACCOUNT` | Receiving database service account |
| `SAMPLE_PDF_PATH` | Sample PDF path for tests |
| `DOCUMENT_SOURCE` | Document source path for course outline generator |
| `TEST_DIRECTORY` | Directory used by the RAG test script |

Create a `config.json` with any of these keys to supply default values. For
example:

```json
{
  "PDF_DIRECTORY": "/data/pdfs",
  "FIRESTORE_SERVICE_ACCOUNT": "/path/to/service.json"
}
```

All scripts will fall back to these settings when command-line arguments are not
provided.

## Gemini Service

The project includes a `GeminiService` class in `Services/Gemini` which wraps the Google Generative AI client. It defaults to `gemini-2.5-flash` for generation and `gemini-2.5-flash-lite` for OCR. A custom Pydantic `GeminiConfig` model provides type-checked generation settings.

The OCR helper now uses a prompt that encourages step-by-step reasoning for improved
accuracy. Responses are validated against a simple `OCRData` Pydantic model which
contains a single `text` field representing the extracted content. The
`GeminiService.ocr` method defaults to returning this model when no custom
`response_model` is supplied.

## Paragraph-based Chunking

Text extraction utilities now support paragraph-level chunking. The `ocr_text_extraction` helper can combine OCR output across pages (pass `combine_pages=True`) and `chunk_text` accepts a `by_paragraph` flag to return one chunk per paragraph.
