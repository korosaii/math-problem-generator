from app.schemas import GenerateRequest, GenerateResponse
from app.prompt_builder import build_prompts
from app.llm_client import generate_json
from app.retriever import LexicalRetriever


retriever = LexicalRetriever()


def should_use_retrieval(request: GenerateRequest) -> bool:
    return request.problem_type in {"computational", "proof", "word_problem"}


def build_query(request: GenerateRequest) -> str:
    return " ".join([
        request.topic,
        request.subtopic,
        request.difficulty,
        request.problem_type,
    ])


def generate_problems(request: GenerateRequest, temperature: float) -> GenerateResponse:
    retrieved_context = []

    if should_use_retrieval(request):
        query = build_query(request)
        retrieved_context = retriever.search(query, top_k=3)

    system_prompt, user_prompt = build_prompts(
        request=request,
        retrieved_context=retrieved_context,
    )

    raw_data = generate_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        request_payload={
            **request.model_dump(),
            "retrieved_context": retrieved_context,
        },
    )

    return GenerateResponse.model_validate(raw_data)
