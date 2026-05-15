import argparse
import json
import time
from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


DATA_DIR = Path("finetuning/data/imdb")
OUTPUT_ROOT = Path("finetuning/outputs")


def get_device_name() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def tokenize_dataset(dataset, tokenizer, max_length: int):
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    return dataset.map(tokenize, batched=True)


def build_compute_metrics():
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        accuracy_result = accuracy.compute(
            predictions=predictions,
            references=labels,
        )

        f1_result = f1.compute(
            predictions=predictions,
            references=labels,
        )

        return {
            "accuracy": accuracy_result["accuracy"],
            "f1": f1_result["f1"],
        }

    return compute_metrics


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def get_directory_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0

    size_bytes = sum(file.stat().st_size for file in path.rglob("*") if file.is_file())
    return round(size_bytes / 1024 / 1024, 4)


def train(args) -> None:
    start_time = time.perf_counter()

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(DATA_DIR / "train.jsonl"),
            "validation": str(DATA_DIR / "validation.jsonl"),
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules.split(","),
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenized = tokenize_dataset(dataset, tokenizer, args.max_length)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    output_dir = OUTPUT_ROOT / args.run_name

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(),
    )

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))

    total_runtime = time.perf_counter() - start_time

    config = {
        "run_name": args.run_name,
        "model_name": args.model_name,
        "dataset": "stanfordnlp/imdb",
        "task": "sentiment_classification",
        "strategy": "progression",
        "method": "LoRA",
        "device": get_device_name(),
        "train_size": len(dataset["train"]),
        "validation_size": len(dataset["validation"]),
        "max_length": args.max_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": args.target_modules.split(","),
        "seed": args.seed,
    }

    metrics = {
        "run_name": args.run_name,
        "model_name": args.model_name,
        "eval_accuracy": eval_metrics.get("eval_accuracy"),
        "eval_f1": eval_metrics.get("eval_f1"),
        "eval_loss": eval_metrics.get("eval_loss"),
        "train_runtime": train_result.metrics.get("train_runtime"),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second"),
        "total_runtime_seconds": round(total_runtime, 4),
        "adapter_size_mb": get_directory_size_mb(adapter_dir),
    }

    save_json(output_dir / "experiment_config.json", config)
    save_json(output_dir / "metrics.json", metrics)

    print("Training finished")
    print(f"run: {args.run_name}")
    print(f"model: {args.model_name}")
    print(f"output: {output_dir}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--target-modules", required=True)

    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)

    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
