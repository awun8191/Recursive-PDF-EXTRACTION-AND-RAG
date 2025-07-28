import logging
import os
import firebase_admin
from firebase_admin import credentials, firestore
import json
import time

from sampleData import DataFormatting
from open_router import generate_multiple_topics

# Path for the cache file
PROCESSED_COURSES_CACHE = "/home/awun/Documents/UNDEFINED MAIN/Scripts/SampleData/firebase/Firestore/processed_courses.json"

class FirestoreManager:
    def __init__(self, service_account_path):
        cred = credentials.Certificate(service_account_path)
        firebase_admin.initialize_app(cred)
        self.db = firestore.client()
        self.courses_data = DataFormatting()
        self.processed_courses = self._load_processed_courses()
        logging.info("Firebase App Initialized and Firestore client obtained.")

    def _load_processed_courses(self):
        if os.path.exists(PROCESSED_COURSES_CACHE):
            with open(PROCESSED_COURSES_CACHE, 'r') as f:
                return set(json.load(f))
        return set()

    def _save_processed_course(self, course_title):
        self.processed_courses.add(course_title)
        with open(PROCESSED_COURSES_CACHE, 'w') as f:
            json.dump(list(self.processed_courses), f)

    def create_course_data(self):
        unprocessed_courses = []
        for dept in self.courses_data.courses:
            for course in dept.courses:
                if course.title not in self.processed_courses:
                    unprocessed_courses.append({
                        "title": course.title,
                        "department": dept.name,
                        "level": course.level,
                        "course_obj": course 
                    })

        # Batch processing
        batch_size = 5
        for i in range(0, len(unprocessed_courses), batch_size):
            batch = unprocessed_courses[i:i + batch_size]
            
            # Prepare batch for AI generation
            courses_to_generate = [{"title": c["title"], "department": c["department"], "level": c["level"]} for c in batch]

            print(f"Processing batch {i//batch_size + 1} with {len(batch)} courses...")
            
            try:
                generated_data = generate_multiple_topics(courses_to_generate)

                if not generated_data:
                    print("Skipping batch due to generation failure.")
                    continue

                for data in generated_data:
                    # Find the original course object
                    original_course_info = next((c for c in batch if c['title'] == data['course']), None)
                    if not original_course_info:
                        continue

                    course = original_course_info["course_obj"]
                    department_name = original_course_info["department"]
                    
                    department_code_map = {
                        "CHEMICAL ENGINEERING": "CHE", "COMPUTER ENGINEERING": "COE",
                        "CIVIL ENGINEERING": "CVL", "ELECTRICAL ENGINEERING": "EEE",
                        "MECHANICAL ENGINEERING": "MEE", "MECHATRONICS ENGINEERING": "MCT",
                        "PETROLEUM ENGINEERING": "PTE", "BIOMEDICAL ENGINEERING": "BME",
                        "AERONAUTICAL ENGINEERING": "AAE"
                    }
                    department_code = department_code_map.get(department_name, "ENG")

                    self.db.collection("course_data").document().set({
                        "department": department_name,
                        "level": course.level,
                        "description": data.get("description", ""),
                        "semester": course.semester,
                        "department_code": department_code,
                        "topics": data.get("topics", []),
                        "title": course.title,
                        "code": course.code,
                        "units": course.units,
                        "type": course.type,
                        "is_elective": course.is_elective
                    })
                    
                    print(f"Successfully added {course.title} to Firestore.")
                    self._save_processed_course(course.title)

            except Exception as e:
                print(f"An error occurred processing batch: {e}")
                # Optional: add a delay before retrying or moving to the next batch
                time.sleep(5)


if __name__ == "__main__":
    manager = FirestoreManager(
        service_account_path="/home/awun/Firebase/undefined/do_not_delete_undefined_325e.json")
    manager.create_course_data()
