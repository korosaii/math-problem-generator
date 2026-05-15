import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


OUTPUT_ROOT = Path("finetuning/outputs")
SUMMARY_PATH = OUTPUT_ROOT / "summary.csv"
PLOT_PATH = OUTPUT_ROOT / "price_quality_curve.png"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    rows = []

    for run_dir in sorted(OUTPUT_ROOT.iterdir()):
        if not run_dir.is_dir():
            continue

        metrics_path = run_dir / "metrics.json"
        config_path = run_dir / "experiment_config.json"

        if not metrics_path.exists() or not config_path.exists():
            continue

        metrics = load_json(metrics_path)
        config = load_json(config_path)

        rows.append({
            "run_name": config["run_name"],
            "model_name": config["model_name"],
            "lora_r": config["lora_r"],
            "lora_alpha": config["lora_alpha"],
            "batch_size": config["batch_size"],
            "max_length": config["max_length"],
            "learning_rate": config["learning_rate"],
            "eval_accuracy": metrics["eval_accuracy"],
            "eval_f1": metrics["eval_f1"],
            "eval_loss": metrics["eval_loss"],
            "train_runtime": metrics["train_runtime"],
            "total_runtime_seconds": metrics["total_runtime_seconds"],
            "adapter_size_mb": metrics["adapter_size_mb"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(SUMMARY_PATH, index=False)

    plt.figure(figsize=(8, 5))
    plt.scatter(df["total_runtime_seconds"], df["eval_f1"])

    for _, row in df.iterrows():
        plt.annotate(
            row["run_name"],
            (row["total_runtime_seconds"], row["eval_f1"]),
            fontsize=8,
        )

    plt.xlabel("Training time, seconds")
    plt.ylabel("Validation F1")
    plt.title("Price-quality curve for LoRA fine-tuning")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200)

    print(f"Summary saved to: {SUMMARY_PATH}")
    print(f"Plot saved to: {PLOT_PATH}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
    