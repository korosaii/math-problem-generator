import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.agent_runner import run_generation_scenario, run_evaluation_scenario


DEFAULT_SCENARIOS_PATH = PROJECT_ROOT / "agent_benchmark" / "scenarios.jsonl"
DEFAULT_RUN_OUTPUT_PATH = PROJECT_ROOT / "agent_benchmark" / "runs" / "v1_agent_run.jsonl"
DEFAULT_METRICS_OUTPUT_PATH = PROJECT_ROOT / "agent_benchmark" / "runs" / "v1_agent_metrics.json"


def load_scenarios(path: Path) -> list[dict]:
    scenarios = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                scenarios.append(json.loads(line))

    return scenarios


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def get_trace_tools(result: dict) -> list[str]:
    return [item.get("tool") for item in result.get("trace", [])]


def is_subsequence(expected: list[str], actual: list[str]) -> bool:
    position = 0

    for tool in actual:
        if position < len(expected) and tool == expected[position]:
            position += 1

    return position == len(expected)


def has_tool_success(result: dict) -> bool:
    trace = result.get("trace", [])

    if not trace:
        return False

    return all(item.get("result", {}).get("ok") is True for item in trace if item.get("tool") != "evaluation_runner")


def has_json_validation_success(result: dict) -> bool:
    for item in result.get("trace", []):
        if item.get("tool") == "json_schema_validator":
            return item.get("result", {}).get("ok") is True

    return False


def has_math_check_success(result: dict) -> bool:
    for item in result.get("trace", []):
        if item.get("tool") == "math_checker":
            output = item.get("result", {}).get("output", {})
            return item.get("result", {}).get("ok") is True and output.get("status") == "basic_check_passed"

    return False


def has_hitl_policy_success(scenario: dict, result: dict) -> bool:
    expected_requires_confirmation = scenario.get("expected_requires_confirmation", False)

    for item in result.get("trace", []):
        if item.get("tool") == "evaluation_runner":
            actual_requires_confirmation = item.get("result", {}).get("requires_confirmation", False)
            return actual_requires_confirmation == expected_requires_confirmation

    return not expected_requires_confirmation


def is_successful_result(scenario: dict, result: dict) -> bool:
    expected_success = scenario.get("expected_success", True)
    final_response = result.get("final_response", "")

    if expected_success:
        return (
            "выполнен" in final_response.lower()
            or "сгенерирована" in final_response.lower()
            or "сгенерирован" in final_response.lower()
        )

    return (
        "требует подтверждения" in final_response.lower()
        or "ошибка" in final_response.lower()
    )


def run_scenario(scenario: dict) -> dict:
    scenario_type = scenario["scenario_type"]

    if scenario_type == "direct_generation":
        return run_generation_scenario(scenario["request"])

    if scenario_type == "eval_without_confirmation":
        return run_evaluation_scenario(confirmed=False)

    if scenario_type == "eval_with_confirmation":
        return run_evaluation_scenario(confirmed=True)

    raise ValueError(f"Unknown scenario_type: {scenario_type}")


def evaluate_scenario(scenario: dict, result: dict, runtime_seconds: float) -> dict:
    expected_tools = scenario.get("expected_tools", [])
    actual_tools = get_trace_tools(result)

    success = is_successful_result(scenario, result)
    tools_ok = is_subsequence(expected_tools, actual_tools)
    tool_calls_ok = has_tool_success(result)
    json_ok = has_json_validation_success(result)
    math_ok = has_math_check_success(result)
    hitl_ok = has_hitl_policy_success(scenario, result)

    if scenario["scenario_type"].startswith("eval_"):
        json_ok = True
        math_ok = True

    return {
        "scenario_id": scenario["scenario_id"],
        "scenario_type": scenario["scenario_type"],
        "success": success,
        "tools_ok": tools_ok,
        "tool_calls_ok": tool_calls_ok,
        "json_ok": json_ok,
        "math_check_ok": math_ok,
        "hitl_ok": hitl_ok,
        "steps": len(actual_tools),
        "runtime_seconds": round(runtime_seconds, 6),
        "expected_tools": expected_tools,
        "actual_tools": actual_tools,
        "final_response": result.get("final_response"),
        "trace": result.get("trace", []),
    }


def mean(values: list[float]) -> float:
    if not values:
        return 0.0

    return sum(values) / len(values)


def compute_metrics(rows: list[dict]) -> dict:
    total = len(rows)

    return {
        "agent": "v1_agent",
        "num_scenarios": total,
        "success_rate": round(mean([row["success"] for row in rows]), 6),
        "tool_sequence_accuracy": round(mean([row["tools_ok"] for row in rows]), 6),
        "tool_call_success_rate": round(mean([row["tool_calls_ok"] for row in rows]), 6),
        "json_validation_rate": round(mean([row["json_ok"] for row in rows]), 6),
        "math_check_pass_rate": round(mean([row["math_check_ok"] for row in rows]), 6),
        "hitl_policy_accuracy": round(mean([row["hitl_ok"] for row in rows]), 6),
        "average_steps": round(mean([row["steps"] for row in rows]), 6),
        "average_runtime": round(mean([row["runtime_seconds"] for row in rows]), 6),
    }


def evaluate(scenarios_path: Path, run_output_path: Path, metrics_output_path: Path) -> None:
    scenarios = load_scenarios(scenarios_path)
    rows = []

    for scenario in scenarios:
        start = time.perf_counter()
        result = run_scenario(scenario)
        runtime_seconds = time.perf_counter() - start

        row = evaluate_scenario(
            scenario=scenario,
            result=result,
            runtime_seconds=runtime_seconds,
        )

        rows.append(row)

    metrics = compute_metrics(rows)

    output = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "results": rows,
    }

    save_jsonl(run_output_path, rows)
    save_json(metrics_output_path, output)

    print(f"Run saved to: {run_output_path}")
    print(f"Metrics saved to: {metrics_output_path}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate agent on benchmark scenarios")

    parser.add_argument(
        "--scenarios",
        default=str(DEFAULT_SCENARIOS_PATH),
        help="Path to agent benchmark scenarios JSONL",
    )
    parser.add_argument(
        "--run-output",
        default=str(DEFAULT_RUN_OUTPUT_PATH),
        help="Path to output run JSONL file",
    )
    parser.add_argument(
        "--metrics-output",
        default=str(DEFAULT_METRICS_OUTPUT_PATH),
        help="Path to output metrics JSON file",
    )

    args = parser.parse_args()

    evaluate(
        scenarios_path=Path(args.scenarios),
        run_output_path=Path(args.run_output),
        metrics_output_path=Path(args.metrics_output),
    )


if __name__ == "__main__":
    main()
    