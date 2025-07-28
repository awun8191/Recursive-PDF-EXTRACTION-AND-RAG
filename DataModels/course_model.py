from pydantic import BaseModel
from typing import List, Dict

from .course_outline import CourseOutline


class CourseModel(BaseModel):
    code: str
    department: str
    department_code: str
    description: str
    is_elective: bool
    level: str
    semester: str
    title: str
    topics: List[str]
    type: str
    units: int


class CourseData(BaseModel):
    courses: List[CourseModel]
