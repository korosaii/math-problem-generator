from pathlib import Path

from datasets import DatasetDict, load_dataset


DATASET_NAME = "stanfordnlp/imdb"
OUTPUT_DIR = Path("finetuning/data/imdb")

TRAIN_SIZE = 5000
VALIDATION_SIZE = 1000
SEED = 42


def save_split(dataset: DatasetDict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset["train"].to_json(OUTPUT_DIR / "train.jsonl", force_ascii=False)
    dataset["validation"].to_json(OUTPUT_DIR / "validation.jsonl", force_ascii=False)
    dataset["test"].to_json(OUTPUT_DIR / "test.jsonl", force_ascii=False)


def main() -> None:
    dataset = load_dataset(DATASET_NAME)

    train_full = dataset["train"].shuffle(seed=SEED)
    test_full = dataset["test"].shuffle(seed=SEED)

    train_subset = train_full.select(range(TRAIN_SIZE + VALIDATION_SIZE))
    split = train_subset.train_test_split(test_size=VALIDATION_SIZE, seed=SEED)

    prepared = DatasetDict({
        "train": split["train"],
        "validation": split["test"],
        "test": test_full.select(range(1000)),
    })

    save_split(prepared)

    print("Dataset prepared")
    print(f"source: {DATASET_NAME}")
    print(f"train: {len(prepared['train'])}")
    print(f"validation: {len(prepared['validation'])}")
    print(f"test: {len(prepared['test'])}")
    print(f"saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
    