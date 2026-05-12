from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas import GenerateRequest


class ToolResult(BaseModel):
    ok: bool
    tool: str
    output: Any = None
    error: str | None = None
    requires_confirmation: bool = False


class ValidateRequestInput(GenerateRequest):
    pass


class KnowledgeRetrieverInput(BaseModel):
    topic: str = Field(..., min_length=1)
    subtopic: str = Field(..., min_length=1)
    difficulty: Literal["easy", "medium", "hard"]
    problem_type: Literal["computational", "proof", "multiple_choice", "word_problem"]
    top_k: int = Field(default=3, ge=1, le=10)


class PromptBuilderInput(BaseModel):
    request: GenerateRequest
    retrieved_context: list[dict] = Field(default_factory=list)


class LLMGenerateInput(BaseModel):
    system_prompt: str = Field(..., min_length=1)
    user_prompt: str = Field(..., min_length=1)
    temperature: float = Field(default=0.4, ge=0.0, le=2.0)
    request_payload: dict = Field(default_factory=dict)


class JSONSchemaValidatorInput(BaseModel):
    raw_response: str = Field(..., min_length=1)


class MathCheckerInput(BaseModel):
    statement: str = Field(..., min_length=1)
    final_answer: str | None = None
    solution_steps: list[str] = Field(default_factory=list)


class AuditLoggerInput(BaseModel):
    event_type: str = Field(..., min_length=1)
    payload: dict = Field(default_factory=dict)


class EvaluationRunnerInput(BaseModel):
    top_k: int = Field(default=3, ge=1, le=10)
    require_confirmation: bool = True
    confirmed: bool = False


class ResultFormatterInput(BaseModel):
    data: dict
    