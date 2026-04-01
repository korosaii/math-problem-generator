from app.schemas import GenerateRequest


def build_prompts(request: GenerateRequest) -> tuple[str, str]:
    system_prompt = """
<role>
You are an experienced mathematics educator and problem generator.
You create mathematically correct, solvable, and well-structured math problems for an API system.
</role>

<goal>
Generate math problems that exactly match the user request and return the result as strictly valid JSON.
</goal>

<global_rules>
- Return JSON only.
- Do not write markdown.
- Do not add explanations outside JSON.
- Do not wrap the response in triple backticks.
- The output language must exactly match the requested output_language.
</global_rules>
""".strip()

    user_prompt = f"""
<task>
Generate exactly {request.num_problems} math problem(s).
</task>

<context>
The generated response will be parsed automatically by the backend.
If the JSON structure is invalid or if extra text appears outside JSON, the system will fail.
Each problem must be mathematically correct, solvable, and consistent with its final answer.
</context>

<requirements>
- Topic must be: {request.topic}
- Subtopic must be: {request.subtopic}
- Difficulty must be: {request.difficulty}
- Problem type must be: {request.problem_type}
- Number of problems must be exactly: {request.num_problems}
- Language must be exactly: {request.output_language}
- Problem IDs must start from 1 and increase by 1
- If include_solution = false, return an empty array in "solution_steps"
- If include_hints = false, return an empty array in "hints"
</requirements>

<input_data>
{{
  "topic": "{request.topic}",
  "subtopic": "{request.subtopic}",
  "difficulty": "{request.difficulty}",
  "problem_type": "{request.problem_type}",
  "num_problems": {request.num_problems},
  "include_solution": {str(request.include_solution).lower()},
  "include_hints": {str(request.include_hints).lower()},
  "output_language": "{request.output_language}"
}}
</input_data>

<decomposition>
Perform the task in these simple steps:
1. Read and understand the input data.
2. Generate exactly {request.num_problems} problems matching the topic, subtopic, difficulty, and problem type.
3. For each problem, write:
   - title
   - statement
   - final_answer
   - hints
   - solution_steps
4. Check mathematical correctness and consistency of each problem.
5. Format the final response strictly as valid JSON.
</decomposition>

<response_format>
Return the response strictly in this JSON structure:

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
</response_format>
""".strip()

    return system_prompt, user_prompt
