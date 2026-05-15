import subprocess


EXPERIMENTS = [
    {
        "run_name": "01_distilbert_lora_r8",
        "model_name": "distilbert-base-uncased",
        "target_modules": "q_lin,v_lin",
        "lora_r": "8",
        "lora_alpha": "16",
    },
    {
        "run_name": "02_bert_lora_r8",
        "model_name": "bert-base-uncased",
        "target_modules": "query,value",
        "lora_r": "8",
        "lora_alpha": "16",
    },
    {
        "run_name": "03_roberta_lora_r8",
        "model_name": "roberta-base",
        "target_modules": "query,value",
        "lora_r": "8",
        "lora_alpha": "16",
    },
    {
        "run_name": "04_distilbert_lora_r4",
        "model_name": "distilbert-base-uncased",
        "target_modules": "q_lin,v_lin",
        "lora_r": "4",
        "lora_alpha": "8",
    },
    {
        "run_name": "05_distilbert_lora_r16",
        "model_name": "distilbert-base-uncased",
        "target_modules": "q_lin,v_lin",
        "lora_r": "16",
        "lora_alpha": "32",
    },
]


def run_experiment(experiment: dict) -> None:
    command = [
        "python",
        "-m",
        "finetuning.train_lora_classifier",
        "--run-name",
        experiment["run_name"],
        "--model-name",
        experiment["model_name"],
        "--target-modules",
        experiment["target_modules"],
        "--epochs",
        "1",
        "--batch-size",
        "4",
        "--max-length",
        "128",
        "--learning-rate",
        "2e-4",
        "--lora-r",
        experiment["lora_r"],
        "--lora-alpha",
        experiment["lora_alpha"],
        "--lora-dropout",
        "0.05",
    ]

    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    for experiment in EXPERIMENTS:
        run_experiment(experiment)


if __name__ == "__main__":
    main()
    