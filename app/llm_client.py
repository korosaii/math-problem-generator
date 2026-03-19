import json
from openai import OpenAI
from app.config import API_KEY, API_BASE, MODEL_NAME


client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE,
)


def generate_json(system_prompt: str, user_prompt: str, temperature: float) -> dict:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("Model returned empty response")

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON: {e}\nRaw content: {content}")
    