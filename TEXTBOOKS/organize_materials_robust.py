import os
import json
import shutil
from pathlib import Path

import google.generativeai as genai
from google.generativeai import types
import pypdf
import time
from pdf2image import convert_from_path
import pytesseract

# --- Configuration ---
# IMPORTANT: Replace with your API key
GEMINI_API_KEY = "AIzaSyBkogtsyCakyqHCeoOyfQmIzQ_HfKpTEyY"
COURSES_JSON_PATH = "courses.json"
COMPILATION_DIR = "COMPILATION"
LOG_FILE = "organization_log_final.txt"
OCR_TIMEOUT_PER_PAGE = 60 # Seconds to wait for OCR on a single page
API_TIMEOUT = 120 # Seconds to wait for a response from the Gemini API
# -------------------

# def get_text_from_image_pdf(pdf_path, num_pages=5):
#     """Extracts text from image-based PDFs using OCR with a timeout."""
#     text = ""
#     try:
#         images = convert_from_path(pdf_path, first_page=1, last_page=num_pages)
#         for i, img in enumerate(images):
#             log(f"    - OCR processing page {i + 1}...", console_only=True)
#             try:
#                 text += pytesseract.image_to_string(img, timeout=OCR_TIMEOUT_PER_PAGE) + "\n"
#             except RuntimeError as timeout_error:
#                 log(f"    - WARNING: Tesseract OCR timed out on page {i + 1}. Skipping page.")
#                 continue
#     except Exception as e:
#         return f"Error during OCR processing: {e}"
#     return text

# def get_pdf_text(pdf_path, num_pages=5):
#     """Extracts text from the first few pages of a PDF, with OCR fallback."""
#     text = ""
#     try:
#         reader = pypdf.PdfReader(pdf_path)
#         if reader.is_encrypted:
#             return "Error: PDF is encrypted."
#         for i in range(min(num_pages, len(reader.pages))):
#             page_text = reader.pages[i].extract_text()
#             if page_text:
#                 text += page_text + "\n"
#     except Exception as e:
#         log(f"PyPDF Error: {e}. Will attempt OCR.")
#         text = ""
#
#     if len(text.strip()) < 150:
#         log("Standard text extraction yielded little content. Attempting OCR...")
#         ocr_text = get_text_from_image_pdf(pdf_path, num_pages)
#         text = text + "\n" + ocr_text
#
#     return text

def analyze_content(file_path: str, context=None):
    """Uses the Gemini API to classify the document and identify the course code."""

    file_name = f"{Path(file_path).stem}"

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash-lite', generation_config=types.GenerationConfig(temperature=0.1))


    if context:
        course_list_str = "\n".join(context['course_list'])
        prompt_context = f"The document is from the '{context['dept_name']}' department, '{context['level']}' level. Please choose from the following specific list:"
    else:
        full_course_list = [f"- {c['code']}: {c['title']}" for d in courses_data.values() for l in d['levels'].values() for s in l.values() for c in s['courses']]
        course_list_str = "\n".join(sorted(list(set(full_course_list))))
        prompt_context = "Please choose from the comprehensive course list below:"

    prompt = f'''
    Analyze the text from an academic document. Perform two tasks:

    1.  **Classify the document type.** Choose ONE from: ["textbook", "past_question", "assignment", "lecture_notes", "lab_manual", "solution_manual", "other"].
    2.  **Identify the most relevant course code.** {prompt_context}

    **Course List:**
    {course_list_str}

    **Document Text:**
    ---
    {text[:4000]}
    ---

    **Respond in JSON format ONLY with the specified keys:**
    {{
      "document_type": "<your classification>",
      "course_code": "<identified course code>"
    }}
    '''

    try:
        response = model.generate_content(prompt, request_options=request_options, generation_config=genai.types.GenerationConfig(temperature=0.1))
        json_response_str = response.text.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(json_response_str)
        doc_type = result.get("document_type")
        course_code = result.get("course_code", "").strip().split(":")[0]
        return doc_type, course_code
    except Exception as e:
        log(f"Gemini API or JSON parsing error: {e}")
        log(f"Failed response text: {response.text if 'response' in locals() else 'No response'}")
        return None, None

def create_course_lookup(courses_data):
    """Creates a mapping from course_code to its location details."""
    lookup = {}
    dept_map = {
        "CHEMICAL": "CHE", "COMPUTER": "COE", "CIVIL": "CVL",
        "ELECTRICAL": "EEE", "MECHANICAL": "MEE", "MECHATRONICS": "MCT",
        "PETROLEUM": "PTE", "BIOMEDICAL": "BME", "AERONAUTICAL": "AAE"
    }
    semester_map = {"FIRST": "1", "SECOND": "2"}

    for dept_key, dept_data in courses_data.items():
        dept_code = dept_map.get(dept_key)
        if not dept_code: continue
        for level, sems in dept_data["levels"].items():
            for sem_key, sem_data in sems.items():
                semester_code = semester_map.get(sem_key)
                if not semester_code: continue
                for course in sem_data["courses"]:
                    if course['code'] in lookup:
                        lookup[course['code']]['dept_key'].append(dept_key)
                        lookup[course['code']]['dept_code'].append(dept_code)
                        lookup[course['code']]['dept_name'].append(dept_data["department_name"])
                    else:
                        lookup[course['code']] = {
                            "dept_key": [dept_key],
                            "dept_code": [dept_code],
                            "level": level,
                            "semester": semester_code,
                            "dept_name": [dept_data["department_name"]]
                    }
    # print(lookup)
    # print(len(lookup))
    return lookup

def log(message, console_only=False):
    """Logs a message to the console and optionally to a file."""
    print(message)
    if not console_only:
        with open(LOG_FILE, "a") as f:
            f.write(message + "\n")

def move_file(file_path, target_dir, filename):
    """Safely moves a file to a target directory."""
    os.makedirs(target_dir, exist_ok=True)
    new_path = os.path.join(target_dir, filename)
    if os.path.abspath(file_path) == os.path.abspath(new_path):
        log(f"File '{filename}' is already in the correct location.")
        return False
    if os.path.exists(new_path):
        log(f"WARNING: File '{new_path}' already exists. Skipping move.")
        return False
    try:
        shutil.move(file_path, new_path)
        log(f"ACTION: Moved '{filename}' to '{target_dir}'")
        return True
    except Exception as e:
        log(f"ERROR: Could not move file: {e}")
        return False

def main():
    """Main function to organize the files."""
    if "YOUR_GEMINI_API_KEY" in GEMINI_API_KEY:
        log("Please replace 'YOUR_GEMINI_API_KEY' with your actual Gemini API key.")
        return

    try:
        with open(COURSES_JSON_PATH, 'r') as f:
            courses_data = json.load(f)
    except FileNotFoundError:
        log(f"Error: '{COURSES_JSON_PATH}' not found.")
        return

    course_lookup = create_course_lookup(courses_data)
    # return ""
    file_count = 0
    actions_taken = 0
    log("--- Starting Robust File Organization Script (v5 - Corrected Second Chance) ---", console_only=True)
    with open(LOG_FILE, "w") as f:
        f.write("--- Starting Robust File Organization Script (v5 - Corrected Second Chance) ---\n")

    all_files = [os.path.join(r, f) for r, _, fs in os.walk(COMPILATION_DIR) for f in fs if f.lower().endswith(".pdf")]
    print(all_files)

    for file_path in all_files:
        log(f"--- Processing: {file_path} ---")
        file_count += 1

        text = get_pdf_text(file_path)
        doc_type, identified_code = analyze_content(text, courses_data)

        if not doc_type or not identified_code:
            log(f"Could not analyze content for {file_path}. Skipping.")
            time.sleep(1)
            continue

        log(f"Initial identification: Type='{doc_type}', Course Code='{identified_code}'")
        
        location_info = course_lookup.get(identified_code)
        
        if not location_info:
            log(f"WARNING: Course code '{identified_code}' is invalid. Attempting second chance analysis...")
            try:
                path_parts = file_path.split(os.sep)
                dept_code_from_path = path_parts[1]
                level_from_path = path_parts[2]
                
                context_courses = []
                context_dept_name = ""
                for code, details in course_lookup.items():
                    if details['dept_code'] == dept_code_from_path and details['level'] == level_from_path:
                        context_dept_name = details['dept_name']
                        # Find the title for this specific course to build the context string
                        for sem_key, sem_data in courses_data[details['dept_key']]['levels'][level_from_path].items():
                            for course in sem_data['courses']:
                                if course['code'] == code:
                                    context_courses.append(f"- {code}: {course['title']}")

                if context_courses:
                    context = {
                        "dept_name": context_dept_name,
                        "level": level_from_path,
                        "course_list": sorted(list(set(context_courses)))
                    }
                    log(f"Second chance context: Dept='{dept_code_from_path}', Level='{level_from_path}'")
                    doc_type, identified_code = analyze_content(text, courses_data, context=context)
                    log(f"Second chance identification: Type='{doc_type}', Course Code='{identified_code}'")
                    location_info = course_lookup.get(identified_code)
                else:
                    log("Could not create context for second chance analysis.")

            except Exception as e:
                log(f"An error occurred during second chance logic: {e}")

        if not location_info:
            log(f"FINAL WARNING: Could not determine correct path for '{identified_code}'. Skipping file.")
            continue

        dept = location_info['dept_code']
        level = location_info['level']
        semester = location_info['semester']
        filename = os.path.basename(file_path)

        moved = False
        if doc_type == "textbook":
            sanitized_code = identified_code.replace("/", "_").replace(" ", "_")
            target_dir = os.path.join(COMPILATION_DIR, dept, level, semester, sanitized_code)
            moved = move_file(file_path, target_dir, filename)
        else:
            folder_map = {
                "past_question": "PQ", "assignment": "Assignments",
                "lecture_notes": "Lecture Notes", "lab_manual": "Lab Manuals",
                "solution_manual": "Solution Manuals", "other": "Other Materials"
            }
            target_folder_name = folder_map.get(doc_type, "Other Materials")
            target_dir = os.path.join(COMPILATION_DIR, dept, level, semester, target_folder_name)
            moved = move_file(file_path, target_dir, filename)
        
        if moved:
            actions_taken += 1
        
        time.sleep(2)

    log(f"\n--- Summary ---")
    log(f"Total files processed: {file_count}")
    log(f"Total files moved: {actions_taken}")

if __name__ == "__main__":
    try:
        import pypdf, google.generativeai, pytesseract
        from pdf2image import convert_from_path
    except ImportError as e:
        print(f"Missing required Python library: {e.name}. Please install it.")
        print("Run: pip install pypdf google-generativeai pdf2image pytesseract")
    else:
        if not shutil.which("pdftoppm"):
            print("ERROR: Poppler not found. It is required by pdf2image.")
            print("Please install Poppler (e.g., 'sudo apt-get install poppler-utils' on Debian/Ubuntu).")
        elif not shutil.which("tesseract"):
            print("ERROR: Tesseract OCR not found.")
            print("Please install Tesseract (e.g., 'sudo apt-get install tesseract-ocr' on Debian/Ubuntu).")
        else:
            main()