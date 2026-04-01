from app.schemas import GenerateRequest, GenerateResponse
from app.prompt_builder import build_prompts
from app.llm_client import generate_json


def generate_problems(request: GenerateRequest, temperature: float) -> GenerateResponse:
    system_prompt, user_prompt = build_prompts(request)
    raw_data = generate_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        request_payload=request.model_dump(),
    )
    return GenerateResponse.model_validate(raw_data)
