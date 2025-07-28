# To run this code, you need to install the following dependencies:
# pip install openai

import os
import json
from openai import OpenAI
from openai import APIError
from typing import List, Dict

def generate_multiple_topics(courses: List[Dict[str, str]]):
    """
    Generates topics for multiple courses, each with its own department and level.

    Args:
        courses: A list of dictionaries, where each dictionary represents a course
                 and contains "title", "department", and "level" keys.

    Returns:
        A list of dictionaries, where each dictionary contains the course name,
        a list of topics, and a description, or None on error.
    """
    api_key = "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"
    if not api_key:
        raise ValueError("Error: OPENROUTER_API_KEY environment variable not set.")

    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        course_details = []
        for c in courses:
            course_details.append(
                f'- Course: "{c["title"]}", Department: "{c["department"]}", Level: {c["level"]}'
            )
        
        courses_str = "\n".join(course_details)

        prompt = f"""
        You are a course content generator. Your task is to generate a detailed, accurate, and complete JSON response for each course provided.

        For each of the following courses:
        {courses_str}

        Generate a single JSON object with a key named "courses".
        The value of "courses" MUST be a JSON array, where each element is an object that strictly follows this structure:
        {{
          "course": "<course name>",
          "topics": ["<topic 1>", "<topic 2>", "<topic 3>", "<topic 4>", "<topic 5>", "<topic 6>", "<topic 7>", "<topic 8>", "<topic 9>", "<topic 10>"],
          "description": "A comprehensive and detailed description of the course, covering its main objectives and learning outcomes."
        }}
        
        **CRITICAL INSTRUCTIONS:**
        1. The "topics" array MUST contain exactly 10 relevant and distinct topics for the specified course. This is a strict requirement.
        2. The final output MUST be a single, valid JSON object with the "courses" key as the top-level element.
        3. Do NOT include any text, markdown formatting, or explanations outside of the raw JSON object.
        """

        completion = client.chat.completions.create(
            model="deepseek/deepseek-chat:free",
            messages=[
                {"role": "system", "content": "You are an expert in educational content creation and strictly follow JSON formatting rules."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
        )

        response_content = completion.choices[0].message.content
        print("--- Raw API Response Text ---")
        print(response_content)
        print("-----------------------------\n")

        data = json.loads(response_content)
        
        if isinstance(data, dict) and len(data.keys()) == 1:
            key = list(data.keys())[0]
            if isinstance(data[key], list):
                return data[key]

        if isinstance(data, list):
            return data
            
        print("Warning: The response was not a list or a dictionary with a single list.")
        return None


    except APIError as e:
        print(f"An OpenRouter API error occurred: {e}")
        return None
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON from the model's response.")
        print(f"Raw response was: {response_content}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def generate_topics_with_openrouter(course: str, department: str, level: str):
    """
    Generates a list of course topics using an OpenRouter model.

    Args:
        course: The name of the course.
        department: The name of the department.
        level: The student level (e.g., '400').

    Returns:
        A Python dictionary parsed from the JSON response, or None on error.
    """
    # --- FIX 1: Secure API Key Handling ---
    # NEVER hardcode your API key. Use environment variables for security.
    # Set the OPENROUTER_API_KEY environment variable in your terminal before running.
    api_key = "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"
    if not api_key:
        raise ValueError("Error: OPENROUTER_API_KEY environment variable not set.")

    try:
        # --- Correctly initialize the client for OpenRouter ---
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        # --- FIX 2: Dynamic and Clearer Prompt ---
        # This prompt structure guides the model to produce the exact JSON format needed.
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

        # --- FIX 3: Correct API Call with JSON Mode ---
        # We use response_format={"type": "json_object"} for reliable JSON output.
        completion = client.chat.completions.create(
            model="deepseek/deepseek-chat:free",  # Using a reliable free model
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            response_format={"type": "json_object"},
        )

        response_content = completion.choices[0].message.content

        # --- FIX 4: Correct JSON Parsing & Debug Printing ---
        print("--- Raw API Response Text ---")
        print(response_content)
        print("-----------------------------\n")

        # Parse the JSON string from the model into a Python dictionary
        data = json.loads(response_content)
        return data.get('course', 'N/A'), data.get('topics', []), data.get('description', 'N/A')

    # --- FIX 5: Robust Error Handling ---
    except APIError as e:
        print(f"An OpenRouter API error occurred: {e}")
        return None
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON from the model's response.")
        print(f"Raw response was: {response_content}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


if __name__ == "__main__":
    # Example usage:
    courses_to_process = [
        {
            "title": "YORUBA LANGUAGE I",
            "department": "General Studies",
            "level": "100"
        },
        {
            "title": "ENGINEERING GRAPHICS AND SOLID MODELLING",
            "department": "Mechanical Engineering",
            "level": "100"
        },
        {
            "title": "GENERAL PRACTICAL PHYSICS II",
            "department": "Physics",
            "level": "100"
        }
    ]

    topics_data = generate_multiple_topics(courses_to_process)
    if topics_data:
        print(json.dumps(topics_data, indent=2))