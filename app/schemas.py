from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    topic: str = Field(..., min_length=1)
    subtopic: str = Field(..., min_length=1)
    difficulty: Literal["easy", "medium", "hard"]
    problem_type: Literal["computational", "proof", "multiple_choice", "word_problem"]
    num_problems: int = Field(..., ge=1, le=10)
    include_solution: bool = True
    include_hints: bool = True
    output_language: Literal["ru", "en"] = "ru"


class ProblemItem(BaseModel):
    id: int
    title: str
    statement: str
    final_answer: Optional[str] = None
    hints: List[str] = Field(default_factory=list)
    solution_steps: List[str] = Field(default_factory=list)


class GenerateResponse(BaseModel):
    topic: str
    subtopic: str
    difficulty: str
    problem_type: str
    output_language: str
    problems: List[ProblemItem]
