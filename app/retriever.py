import json
import math
import re
from collections import Counter
from pathlib import Path


KB_PATH = Path("knowledge_base.json")


def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r"[a-zA-Zа-яА-Я0-9]+", text)
    return tokens


class LexicalRetriever:
    def __init__(self, kb_path: Path = KB_PATH):
        if not kb_path.exists():
            raise FileNotFoundError(f"Knowledge base file not found: {kb_path}")

        with open(kb_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)

        if not isinstance(self.documents, list):
            raise ValueError("knowledge_base.json must contain a list of documents")

        self.doc_tokens = []
        self.df = Counter()
        self.N = len(self.documents)

        for doc in self.documents:
            text = " ".join([
                str(doc.get("topic", "")),
                str(doc.get("subtopic", "")),
                str(doc.get("difficulty", "")),
                str(doc.get("problem_type", "")),
                str(doc.get("content", "")),
            ])
            tokens = tokenize(text)
            self.doc_tokens.append(tokens)
            for token in set(tokens):
                self.df[token] += 1

    def _tf_idf_score(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        if not doc_tokens:
            return 0.0

        tf = Counter(doc_tokens)
        score = 0.0

        for token in query_tokens:
            if token not in tf:
                continue
            idf = math.log((self.N + 1) / (self.df[token] + 1)) + 1.0
            score += tf[token] * idf

        return score

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        query_tokens = tokenize(query)
        scored = []

        for doc, tokens in zip(self.documents, self.doc_tokens):
            score = self._tf_idf_score(query_tokens, tokens)
            if score > 0:
                scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]
    