from pydantic import BaseModel, Field
from typing import List


class Question(BaseModel):
    course_code: str = Field(..., description="This is the course code for this question")
    course_name: str = Field(..., description="This is the name of the course")
    topic_name: str = Field(..., description="This is the name of the topic of which the question would be generated")
    difficulty_ranking: int = Field(..., description="This is a number ranging from 1 to 10 used to quantify difficulty. 1 to 3 are easy questions, 5 to 7 are Medium difficulty questions and 8 to 10 are difficult questions")
    difficulty: str = Field(..., description="This is the difficulty gotten from the difficulty ranking")
    relevance_score: float = Field(..., description="A measure of how relevant the question is to the topic ranging from 1 to 5")
    question: str = Field(..., description="The main question text")
    options: List[str] = Field(..., description="Multiple choice options")
    correct_answer: str = Field(..., description="The correct option letter (e.g., A, B, C, D)")
    explanation: str = Field(..., description="A concise explanation of why the answer is correct (1-2 sentences)")
    solution_steps: List[str] = Field(..., description="Only for calculation-based questions. Max 5 short steps")


class QuestionSet(BaseModel):
    questions: List[Question]
