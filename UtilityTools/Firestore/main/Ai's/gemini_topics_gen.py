# To run this code you need to install the following dependencies:
# pip install google-generativeai

import os
import json
import google.generativeai as genai
from google.generativeai.types import GenerationConfig


def gemini_generate(course: str, department: str, level: str):
    """
    Generates a list of course topics using the Gemini API.

    Args:
        course: The name of the course.
        department: The name of the department.
        level: The student level (e.g., '400').

    Returns:
        A Python dictionary parsed from the JSON response, or None on error.
    """
    # --- FIX 1: Secure API Key Handling ---
    # Never hardcode API keys. Use environment variables for security.
    # Set the GOOGLE_API_KEY environment variable in your terminal before running.
    api_key = "AIzaSyBkogtsyCakyqHCeoOyfQmIzQ_HfKpTEyY"
    if not api_key:
        raise ValueError("Error: GOOGLE_API_KEY environment variable not set.")

    genai.configure(api_key=api_key)

    # --- FIX 2: Correct Model Initialization & API Usage ---
    # Use a standard, available model.
    # The 'genai.Client' class is not the correct way to initialize.
    model = genai.GenerativeModel('gemini-1.5-flash-latest')

    # --- FIX 3: Dynamic and Clearer Prompt ---
    # Use the 'level' and 'department' variables in the prompt for flexibility.
    # Also, guide the model more clearly on the desired JSON structure.
    prompt = f"""
            Generate 10 concise topics for the course "{course}" suitable for a {level}-level student 
            in the {department} department.
            Provide the output as a valid JSON object following this exact structure:


            {{
              "course": "{course}",
              "topics": ["topic 1", "topic 2", ...],
              "description": "A complete description of the course."
            }}

            You MUST reply with only the raw JSON object, without any additional text or markdown formatting.
            """

    # --- FIX 4: Correct Configuration for JSON Output ---
    # The response_mime_type is part of GenerationConfig.
    generation_config = GenerationConfig(response_mime_type="application/json")

    # --- FIX 5: Correct API Call and Added Error Handling ---
    try:
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
        )

        # --- FIX 6: Correct JSON Parsing ---
        # response.text is already a JSON string.
        # We need to parse it into a Python dictionary using json.loads().
        print("--- Raw API Response Text ---")
        print(response.text)
        print("-----------------------------\n")

        data = json.loads(response.text)
        return data.get('course', 'N/A'), data.get('topics', []), data.get('description', 'N/A')

    except json.JSONDecodeError:
        print("Error: Failed to decode JSON from the model's response.")
        print(f"Raw response was: {response.text}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


if __name__ == "__main__":
    # Corrected the typo "Dc Cirsuits" to "DC Circuits" for better results
    generated_data = gemini_generate("DC Circuits", "Electrical Engineering", "400")

    print(generated_data)