import os
import fitz  # PyMuPDF
import tiktoken
import argparse
from concurrent.futures import ThreadPoolExecutor

# Add '.pdf' to the set of supported extensions
SUPPORTED_EXTENSIONS = {
    '.pdf', '.py', '.js', '.jsx', '.ts', '.tsx', '.html', '.css', '.json',
    '.yaml', '.yml', '.xml', '.md', '.rst', '.txt', '.c', '.cpp', '.h',
    '.java', '.sh', '.rb', '.php', '.go'
}


def count_tokens_in_file(file_path, encoding):
    """Counts tokens in a single plain text file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            tokens = encoding.encode(content)
            print(f"-> Counted {len(tokens):,} tokens in (text): {os.path.basename(file_path)}")
            return len(tokens)
    except Exception as e:
        print(f"Could not process text file {os.path.basename(file_path)}: {e}")
        return 0


def count_tokens_in_pdf(file_path, encoding):
    """Extracts text from a PDF and counts tokens."""
    total_text = ""
    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                total_text += page.get_text()

        tokens = encoding.encode(total_text)
        print(f"-> Counted {len(tokens):,} tokens in (PDF):  {os.path.basename(file_path)}")
        return len(tokens)
    except Exception as e:
        print(f"Could not process PDF file {os.path.basename(file_path)}: {e}")
        return 0


def process_file(file_path, encoding):
    """Checks file type and calls the appropriate token counting function."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return count_tokens_in_pdf(file_path, encoding)
    else:
        return count_tokens_in_file(file_path, encoding)


def main():
    """Main function to parse arguments and count tokens in a directory."""
    parser = argparse.ArgumentParser(
        description="Count tokens in all supported files (text and PDF) in a directory."
    )
    parser.add_argument("directory", help="The path to the directory to scan.")
    parser.add_argument("--encoding", default="cl100k_base", help="The tiktoken encoding model.")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: Directory not found at '{args.directory}'")
        return

    try:
        encoding = tiktoken.get_encoding(args.encoding)
    except ValueError as e:
        print(f"Error: Invalid encoding name '{args.encoding}'. {e}")
        return

    file_paths = []
    for root, _, files in os.walk(args.directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in SUPPORTED_EXTENSIONS:
                file_paths.append(os.path.join(root, file))

    if not file_paths:
        print("No supported files found to process in the specified directory.")
        return

    total_tokens = 0
    with ThreadPoolExecutor() as executor:
        results = executor.map(lambda p: process_file(p, encoding), file_paths)
        total_tokens = sum(results)

    print("\n" + "=" * 40)
    print("Scan Complete.")
    print(f"Total files scanned: {len(file_paths):,}")
    print(f"Total tokens found:  {total_tokens:,}")
    print(f"Encoding model used: {args.encoding}")
    print("=" * 40)


if __name__ == "__main__":
    main()