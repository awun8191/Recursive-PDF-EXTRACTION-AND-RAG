#!/usr/bin/env python3
"""
Script to convert docx, txt, ppt files to PDF and delete unwanted file types
while preserving images (jpeg, png, etc.)
"""

import os
import subprocess
import sys
from pathlib import Path
import shutil

def run_command(cmd, cwd=None):
    """Run a shell command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, 
                              capture_output=True, text=True, timeout=300)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        print(f"Command timed out: {cmd}")
        return False, "", "Timeout"
    except Exception as e:
        print(f"Error running command {cmd}: {e}")
        return False, "", str(e)

def convert_to_pdf(file_path):
    """Convert a file to PDF using LibreOffice"""
    file_path = Path(file_path)
    output_dir = file_path.parent
    
    # Use LibreOffice headless mode to convert to PDF
    cmd = f'libreoffice --headless --convert-to pdf --outdir "{output_dir}" "{file_path}"'
    
    success, stdout, stderr = run_command(cmd)
    
    if success:
        # Check if PDF was created
        pdf_path = output_dir / f"{file_path.stem}.pdf"
        if pdf_path.exists():
            print(f"✓ Converted: {file_path.name} -> {pdf_path.name}")
            return True
        else:
            print(f"✗ PDF not created for: {file_path.name}")
            return False
    else:
        print(f"✗ Failed to convert: {file_path.name}")
        if stderr:
            print(f"  Error: {stderr}")
        return False

def should_delete_file(file_path):
    """Check if file should be deleted based on extension"""
    file_path = Path(file_path)
    
    # Extensions to delete (unwanted file types)
    delete_extensions = {
        '.m',           # MATLAB files
        '.matcad',      # Mathcad files
        '.htm', '.html', # HTML files
        '.mp3', '.mp4', # Audio/Video files
        '.aac', '.ogg', # Audio files
        '.mat',         # MATLAB data files
        '.tmp',         # Temporary files
        '.dthumb',      # Thumbnail files
        '.eml',         # Email files
        '.ods', '.xlsx', '.xls',  # Spreadsheets (keep for now)
    }
    
    # Extensions to keep (images and already converted files)
    keep_extensions = {
        '.pdf',         # Already PDF
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp',  # Images

    }
    
    ext = file_path.suffix.lower()
    
    if ext in delete_extensions:
        return True
    elif ext in keep_extensions:
        return False
    elif ext in {'.docx', '.txt', '.ppt', '.pptx', '.odt'}:
        # These should be converted first, then deleted
        return False
    else:
        # Unknown extension - be cautious and keep
        return False

def process_directory(base_dir):
    """Process all files in the directory recursively"""
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Directory not found: {base_dir}")
        return
    
    # Statistics
    converted_count = 0
    deleted_count = 0
    failed_conversions = 0
    
    print(f"Processing directory: {base_path}")
    print("=" * 60)
    
    # First pass: Convert files to PDF
    print("\n1. Converting files to PDF...")
    print("-" * 40)
    
    convertible_extensions = {'.docx', '.txt', '.ppt', '.pptx', '.odt'}
    
    for file_path in base_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in convertible_extensions:
            # Skip temporary files (starting with ~$)
            if file_path.name.startswith('~$'):
                continue
                
            print(f"Converting: {file_path.relative_to(base_path)}")
            
            if convert_to_pdf(file_path):
                converted_count += 1
                # Delete original file after successful conversion
                try:
                    file_path.unlink()
                    print(f"  Deleted original: {file_path.name}")
                except Exception as e:
                    print(f"  Warning: Could not delete original {file_path.name}: {e}")
            else:
                failed_conversions += 1
    
    # Second pass: Delete unwanted files
    print(f"\n2. Deleting unwanted file types...")
    print("-" * 40)
    
    for file_path in base_path.rglob('*'):
        if file_path.is_file() and should_delete_file(file_path):
            try:
                print(f"Deleting: {file_path.relative_to(base_path)}")
                file_path.unlink()
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {file_path.name}: {e}")
    
    # Clean up empty directories
    print(f"\n3. Cleaning up empty directories...")
    print("-" * 40)
    
    empty_dirs_removed = 0
    for dir_path in sorted(base_path.rglob('*'), key=lambda p: len(p.parts), reverse=True):
        if dir_path.is_dir() and not any(dir_path.iterdir()):
            try:
                dir_path.rmdir()
                print(f"Removed empty directory: {dir_path.relative_to(base_path)}")
                empty_dirs_removed += 1
            except Exception as e:
                print(f"Could not remove directory {dir_path.name}: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"Files converted to PDF: {converted_count}")
    print(f"Failed conversions: {failed_conversions}")
    print(f"Unwanted files deleted: {deleted_count}")
    print(f"Empty directories removed: {empty_dirs_removed}")
    print("=" * 60)

if __name__ == "__main__":
    # Get the TEXTBOOKS directory path
    textbooks_dir = "../../TEXTBOOKS"
    
    if not os.path.exists(textbooks_dir):
        print(f"Error: Directory '{textbooks_dir}' not found!")
        sys.exit(1)
    
    print("This script will:")
    print("1. Convert .docx, .txt, .ppt, .pptx files to PDF")
    print("2. Delete original files after successful conversion")
    print("3. Delete unwanted file types (matlab, html, media files, etc.)")
    print("4. Keep images (.jpg, .png, etc.) and PDFs")
    print("5. Clean up empty directories")
    
    response = input("\nDo you want to continue? (y/N): ")
    
    if response.lower() != 'y':
        print("Operation cancelled.")
        sys.exit(0)
    
    process_directory(textbooks_dir)
