import argparse
import csv
import json
from pathlib import Path

import requests


def load_dataset(dataset_path: str) -> list[dict]:
    path = Path(dataset_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON array (list of examples)")

    return data


def call_generate_api(base_url: str, request_body: dict, temperature: float) -> dict:
    url = f"{base_url.rstrip('/')}/generate"
    response = requests.post(
        url,
        params={"temperature": temperature},
        json=request_body,
        timeout=60,
    )

    try:
        response.raise_for_status()
    except requests.HTTPError as e:
        raise requests.HTTPError(
            f"HTTP {response.status_code}: {response.text}"
        ) from e

    return response.json()


def build_request_text(example: dict) -> str:
    request_repr = {
        "temperature": example.get("temperature", 0.4),
        "request": example.get("request", {}),
    }
    return json.dumps(request_repr, ensure_ascii=False)


def build_result_text(result: dict) -> str:
    return json.dumps(result, ensure_ascii=False)


def save_summary_table(rows: list[dict], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["запрос", "результат"])
        writer.writeheader()
        writer.writerows(rows)


def run_evaluation(dataset_path: str, base_url: str, output_path: str) -> None:
    dataset = load_dataset(dataset_path)
    summary_rows = []

    for idx, example in enumerate(dataset, start=1):
        if "request" not in example:
            result_text = json.dumps(
                {"error": "Dataset example does not contain 'request' field"},
                ensure_ascii=False
            )
            summary_rows.append({
                "запрос": json.dumps(example, ensure_ascii=False),
                "результат": result_text,
            })
            continue

        request_body = example["request"]
        temperature = example.get("temperature", 0.4)

        request_text = build_request_text(example)

        try:
            result = call_generate_api(
                base_url=base_url,
                request_body=request_body,
                temperature=temperature,
            )
            result_text = build_result_text(result)
            print(f"[{idx}/{len(dataset)}] OK")
        except Exception as e:
            result_text = json.dumps({"error": str(e)}, ensure_ascii=False)
            print(f"[{idx}/{len(dataset)}] ERROR: {e}")

        summary_rows.append({
            "запрос": request_text,
            "результат": result_text,
        })

    save_summary_table(summary_rows, output_path)
    print(f"\nSummary table saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation dataset through Math Problem Generator API"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to evaluation dataset JSON file"
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL of FastAPI app"
    )
    parser.add_argument(
        "--output",
        default="artifacts/summary_table.csv",
        help="Path to output CSV file"
    )

    args = parser.parse_args()

    run_evaluation(
        dataset_path=args.dataset,
        base_url=args.base_url,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
