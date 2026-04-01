import json
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI
from app.config import API_KEY, API_BASE, MODEL_NAME
from app.schemas import GenerateResponse


client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE,
)

AUDIT_DIR = Path("artifacts")
AUDIT_DIR.mkdir(exist_ok=True)


def save_audit_record(record: dict) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    path = AUDIT_DIR / f"audit_{timestamp}.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


def generate_json(
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    request_payload: dict,
) -> dict:
    response_schema = GenerateResponse.model_json_schema()

    audit_record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": MODEL_NAME,
        "temperature": temperature,
        "request_payload": request_payload,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "response_schema": response_schema,
    }

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=temperature,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "generate_response",
                    "schema": response_schema,
                },
            },
            messages=[
                {"role": "developer", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        message = response.choices[0].message

        if not message.content:
            raise ValueError("Model returned empty response")

        audit_record["raw_response"] = message.content

        validated = GenerateResponse.model_validate_json(message.content)
        parsed = validated.model_dump()

        audit_record["parsed_response"] = parsed
        audit_record["validation_status"] = "success"

        save_audit_record(audit_record)
        return parsed

    except Exception as e:
        audit_record["validation_status"] = "failed"
        audit_record["error"] = str(e)
        save_audit_record(audit_record)
        raise
    