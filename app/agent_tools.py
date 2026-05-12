import csv
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.agent_tool_models import (
    AuditLoggerInput,
    EvaluationRunnerInput,
    JSONSchemaValidatorInput,
    KnowledgeRetrieverInput,
    LLMGenerateInput,
    MathCheckerInput,
    PromptBuilderInput,
    ResultFormatterInput,
    ToolResult,
    ValidateRequestInput,
)
from app.llm_client import generate_json
from app.prompt_builder import build_prompts
from app.retriever import LexicalRetriever
from app.schemas import GenerateRequest, GenerateResponse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
AGENT_RUNS_DIR = ARTIFACTS_DIR / "agent_runs"

ARTIFACTS_DIR.mkdir(exist_ok=True)
AGENT_RUNS_DIR.mkdir(exist_ok=True)


def validate_request_tool(data: dict) -> ToolResult:
    try:
        validated = ValidateRequestInput.model_validate(data)
        return ToolResult(ok=True, tool="validate_request", output=validated.model_dump())
    except Exception as error:
        return ToolResult(ok=False, tool="validate_request", error=str(error))


def knowledge_retriever_tool(data: dict) -> ToolResult:
    try:
        params = KnowledgeRetrieverInput.model_validate(data)
        retriever = LexicalRetriever(kb_path=PROJECT_ROOT / "knowledge_base.json")
        query = " ".join([params.topic, params.subtopic, params.difficulty, params.problem_type])
        docs = retriever.search(query, top_k=params.top_k)
        return ToolResult(ok=True, tool="knowledge_retriever", output={"query": query, "documents": docs})
    except Exception as error:
        return ToolResult(ok=False, tool="knowledge_retriever", error=str(error))


def prompt_builder_tool(data: dict) -> ToolResult:
    try:
        params = PromptBuilderInput.model_validate(data)
        system_prompt, user_prompt = build_prompts(
            request=params.request,
            retrieved_context=params.retrieved_context,
        )
        return ToolResult(
            ok=True,
            tool="prompt_builder",
            output={"system_prompt": system_prompt, "user_prompt": user_prompt},
        )
    except Exception as error:
        return ToolResult(ok=False, tool="prompt_builder", error=str(error))


def llm_generate_tool(data: dict) -> ToolResult:
    try:
        params = LLMGenerateInput.model_validate(data)
        result = generate_json(
            system_prompt=params.system_prompt,
            user_prompt=params.user_prompt,
            temperature=params.temperature,
            request_payload=params.request_payload,
        )
        return ToolResult(ok=True, tool="llm_generate", output=result)
    except Exception as error:
        return ToolResult(ok=False, tool="llm_generate", error=str(error))


def json_schema_validator_tool(data: dict) -> ToolResult:
    try:
        params = JSONSchemaValidatorInput.model_validate(data)
        validated = GenerateResponse.model_validate_json(params.raw_response)
        return ToolResult(ok=True, tool="json_schema_validator", output=validated.model_dump())
    except Exception as error:
        return ToolResult(ok=False, tool="json_schema_validator", error=str(error))


def math_checker_tool(data: dict) -> ToolResult:
    try:
        params = MathCheckerInput.model_validate(data)

        checks = {
            "has_statement": bool(params.statement.strip()),
            "has_final_answer": bool(params.final_answer and params.final_answer.strip()),
            "has_solution_steps": len(params.solution_steps) > 0,
            "status": "basic_check_passed",
        }

        if "proof" in params.statement.lower():
            checks["status"] = "manual_review_recommended"

        return ToolResult(ok=True, tool="math_checker", output=checks)
    except Exception as error:
        return ToolResult(ok=False, tool="math_checker", error=str(error))


def audit_logger_tool(data: dict) -> ToolResult:
    try:
        params = AuditLoggerInput.model_validate(data)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        path = AGENT_RUNS_DIR / f"agent_audit_{timestamp}.json"

        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event_type": params.event_type,
            "payload": params.payload,
        }

        with path.open("w", encoding="utf-8") as file:
            json.dump(record, file, ensure_ascii=False, indent=2)

        return ToolResult(ok=True, tool="audit_logger", output={"path": str(path)})
    except Exception as error:
        return ToolResult(ok=False, tool="audit_logger", error=str(error))


def evaluation_runner_tool(data: dict) -> ToolResult:
    try:
        params = EvaluationRunnerInput.model_validate(data)

        if params.require_confirmation and not params.confirmed:
            return ToolResult(
                ok=False,
                tool="evaluation_runner",
                error="Evaluation writes files to benchmark/runs and requires confirmation",
                requires_confirmation=True,
            )

        command = [
            "python",
            "-m",
            "scripts.evaluate_retriever",
            "--top-k",
            str(params.top_k),
        ]

        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        output = {
            "command": " ".join(command),
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }

        if completed.returncode != 0:
            return ToolResult(ok=False, tool="evaluation_runner", output=output, error=completed.stderr)

        return ToolResult(ok=True, tool="evaluation_runner", output=output)
    except Exception as error:
        return ToolResult(ok=False, tool="evaluation_runner", error=str(error))


def result_formatter_tool(data: dict) -> ToolResult:
    try:
        params = ResultFormatterInput.model_validate(data)
        validated = GenerateResponse.model_validate(params.data)
        return ToolResult(ok=True, tool="result_formatter", output=validated.model_dump())
    except Exception as error:
        return ToolResult(ok=False, tool="result_formatter", error=str(error))


TOOL_FUNCTIONS = {
    "validate_request": validate_request_tool,
    "knowledge_retriever": knowledge_retriever_tool,
    "prompt_builder": prompt_builder_tool,
    "llm_generate": llm_generate_tool,
    "json_schema_validator": json_schema_validator_tool,
    "math_checker": math_checker_tool,
    "audit_logger": audit_logger_tool,
    "evaluation_runner": evaluation_runner_tool,
    "result_formatter": result_formatter_tool,
}


def tool_schema(name: str, description: str, model: type) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": model.model_json_schema(),
        },
    }


OPENAI_TOOLS = [
    tool_schema(
        "validate_request",
        "Validate user request for math problem generation.",
        ValidateRequestInput,
    ),
    tool_schema(
        "knowledge_retriever",
        "Retrieve relevant math knowledge from local knowledge_base.json.",
        KnowledgeRetrieverInput,
    ),
    tool_schema(
        "prompt_builder",
        "Build system and user prompts for the LLM.",
        PromptBuilderInput,
    ),
    tool_schema(
        "llm_generate",
        "Generate math problems with the OpenAI-compatible LLM.",
        LLMGenerateInput,
    ),
    tool_schema(
        "json_schema_validator",
        "Validate raw LLM response against GenerateResponse schema.",
        JSONSchemaValidatorInput,
    ),
    tool_schema(
        "math_checker",
        "Run basic correctness checks for generated math problem.",
        MathCheckerInput,
    ),
    tool_schema(
        "audit_logger",
        "Save agent event or tool execution audit record.",
        AuditLoggerInput,
    ),
    tool_schema(
        "evaluation_runner",
        "Run retriever evaluation benchmark. This tool requires confirmation because it writes result files.",
        EvaluationRunnerInput,
    ),
    tool_schema(
        "result_formatter",
        "Format and validate final API response.",
        ResultFormatterInput,
    ),
]
