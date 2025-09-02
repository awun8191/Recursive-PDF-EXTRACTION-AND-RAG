import logging
from typing import List
import json


class CourseModel:

    def __init__(self):
        self.semester = ""
        self.level = ""
        self.title = ""
        self.code = ""
        self.units = 0
        self.type = ""
        self.is_elective = ""

class DepartmentModel:

    def __init__(self):
        self.name = ""
        self.courses: List[CourseModel] = []





class DataFormatting:

    def __init__(self):
        self.json_document = self._read_json()
        self.courses: List[DepartmentModel] = []
        self.map_data()

        logging.info("Course Data Initialized from Json Document")
        print(len(self.courses))


    def _read_json(self):
        from config import load_config
        config = load_config()
        with open(config.courses_json_path) as courses_file:
            data = json.load(courses_file)

        return data

    def map_data(self):

        for department, info in self.json_document.items():
            dep = DepartmentModel()
            dep.name = info["department_name"]
            for level, level_info in info["levels"].items():
                for semester, semester_info in level_info.items():
                    for course_info in semester_info["courses"]:
                        course = CourseModel()
                        course.title = course_info["title"]
                        course.semester = semester
                        course.type = course_info["type"]
                        course.level = level
                        course.is_elective = course_info["is_elective"]
                        course.units = course_info["units"]
                        course.code = course_info["code"]

                        dep.courses.append(course)
            self.courses.append(dep)




data = DataFormatting()

