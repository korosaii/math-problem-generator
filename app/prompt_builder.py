from app.schemas import GenerateRequest


def build_prompts(request: GenerateRequest) -> tuple[str, str]:
    system_prompt = """
You are a math problem generator API.

Your task is to generate mathematically correct problems and return strictly valid JSON only.
Do not output markdown.
Do not output explanations outside JSON.
Do not wrap JSON in triple backticks.
Ensure the problems are solvable, consistent, and match the requested topic, subtopic, difficulty, and problem type.
If include_solution is false, return empty solution_steps arrays.
If include_hints is false, return empty hints arrays.
The output language must exactly match the requested output_language.
""".strip()

    user_prompt = f"""
Generate {request.num_problems} math problem(s) with the following parameters:

topic: {request.topic}
subtopic: {request.subtopic}
difficulty: {request.difficulty}
problem_type: {request.problem_type}
include_solution: {str(request.include_solution).lower()}
include_hints: {str(request.include_hints).lower()}
output_language: {request.output_language}

Return strictly valid JSON with exactly this structure:

{{
  "topic": "{request.topic}",
  "subtopic": "{request.subtopic}",
  "difficulty": "{request.difficulty}",
  "problem_type": "{request.problem_type}",
  "output_language": "{request.output_language}",
  "problems": [
    {{
      "id": 1,
      "title": "string",
      "statement": "string",
      "final_answer": "string or null",
      "hints": ["string"],
      "solution_steps": ["string"]
    }}
  ]
}}

Requirements:
- The number of items in "problems" must be exactly {request.num_problems}.
- Each problem must be mathematically correct.
- Keep the response fully in {request.output_language}.
- Return JSON only.
""".strip()

    return system_prompt, user_prompt
