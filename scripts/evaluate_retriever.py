import argparse
import csv
import json
import math
from pathlib import Path

from app.retriever import LexicalRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_KB_PATH = PROJECT_ROOT / "knowledge_base.json"
DEFAULT_QUERIES_PATH = PROJECT_ROOT / "benchmark" / "queries.jsonl"
DEFAULT_QRELS_PATH = PROJECT_ROOT / "benchmark" / "qrels" / "test.tsv"
DEFAULT_RUN_OUTPUT_PATH = PROJECT_ROOT / "benchmark" / "runs" / "v1_lexical.tsv"
DEFAULT_METRICS_OUTPUT_PATH = PROJECT_ROOT / "benchmark" / "runs" / "v1_lexical_metrics.json"


def load_queries(path: Path) -> list[dict]:
    queries = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                queries.append(json.loads(line))

    return queries


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels = {}

    with path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter="\t")

        for row in reader:
            query_id = row["query_id"]
            doc_id = str(row["doc_id"])
            score = int(row["score"])

            if query_id not in qrels:
                qrels[query_id] = {}

            qrels[query_id][doc_id] = score

    return qrels


def dcg(relevances: list[int]) -> float:
    value = 0.0

    for index, relevance in enumerate(relevances, start=1):
        value += relevance / math.log2(index + 1)

    return value


def precision_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
    top_k = retrieved_doc_ids[:k]

    if not top_k:
        return 0.0

    true_positive = sum(doc_id in relevant_doc_ids for doc_id in top_k)

    return true_positive / k


def recall_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
    if not relevant_doc_ids:
        return 0.0

    top_k = retrieved_doc_ids[:k]
    true_positive = sum(doc_id in relevant_doc_ids for doc_id in top_k)

    return true_positive / len(relevant_doc_ids)


def mrr_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
    top_k = retrieved_doc_ids[:k]

    for rank, doc_id in enumerate(top_k, start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
    top_k = retrieved_doc_ids[:k]

    relevances = [
        1 if doc_id in relevant_doc_ids else 0
        for doc_id in top_k
    ]

    ideal_relevances = [1] * min(len(relevant_doc_ids), k)
    ideal_relevances += [0] * (k - len(ideal_relevances))

    actual_dcg = dcg(relevances)
    ideal_dcg = dcg(ideal_relevances)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


def save_run(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["query_id", "doc_id", "score", "rank"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def save_metrics(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)


def evaluate(kb_path: Path, queries_path: Path, qrels_path: Path, run_output_path: Path, metrics_output_path: Path, top_k: int) -> None:
    queries = load_queries(queries_path)
    qrels = load_qrels(qrels_path)
    retriever = LexicalRetriever(kb_path=kb_path)

    run_rows = []
    precision_values = []
    recall_values = []
    mrr_values = []
    ndcg_values = []

    for query in queries:
        query_id = query["query_id"]
        query_text = query["text"]

        relevant_doc_ids = {
            doc_id
            for doc_id, score in qrels.get(query_id, {}).items()
            if score > 0
        }

        retrieved_docs = retriever.search(query_text, top_k=top_k)
        retrieved_doc_ids = [str(doc["id"]) for doc in retrieved_docs]

        for rank, doc in enumerate(retrieved_docs, start=1):
            run_rows.append({
                "query_id": query_id,
                "doc_id": str(doc["id"]),
                "score": "",
                "rank": rank,
            })

        precision_values.append(precision_at_k(retrieved_doc_ids, relevant_doc_ids, top_k))
        recall_values.append(recall_at_k(retrieved_doc_ids, relevant_doc_ids, top_k))
        mrr_values.append(mrr_at_k(retrieved_doc_ids, relevant_doc_ids, top_k))
        ndcg_values.append(ndcg_at_k(retrieved_doc_ids, relevant_doc_ids, top_k))

    num_queries = len(queries)

    metrics = {
        "retriever": "v1_lexical",
        "top_k": top_k,
        "num_queries": num_queries,
        f"precision@{top_k}": round(sum(precision_values) / num_queries, 6),
        f"recall@{top_k}": round(sum(recall_values) / num_queries, 6),
        f"mrr@{top_k}": round(sum(mrr_values) / num_queries, 6),
        f"ndcg@{top_k}": round(sum(ndcg_values) / num_queries, 6),
    }

    save_run(run_output_path, run_rows)
    save_metrics(metrics_output_path, metrics)

    print(f"Run saved to: {run_output_path}")
    print(f"Metrics saved to: {metrics_output_path}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LexicalRetriever on benchmark dataset")

    parser.add_argument("--kb", default=str(DEFAULT_KB_PATH))
    parser.add_argument("--queries", default=str(DEFAULT_QUERIES_PATH))
    parser.add_argument("--qrels", default=str(DEFAULT_QRELS_PATH))
    parser.add_argument("--run-output", default=str(DEFAULT_RUN_OUTPUT_PATH))
    parser.add_argument("--metrics-output", default=str(DEFAULT_METRICS_OUTPUT_PATH))
    parser.add_argument("--top-k", type=int, default=3)

    args = parser.parse_args()

    evaluate(
        kb_path=Path(args.kb),
        queries_path=Path(args.queries),
        qrels_path=Path(args.qrels),
        run_output_path=Path(args.run_output),
        metrics_output_path=Path(args.metrics_output),
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
