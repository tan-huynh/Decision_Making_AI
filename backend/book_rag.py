from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
BOOK_TEXT = ROOT_DIR / "decision_book.txt"

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were", "been",
    "have", "has", "had", "not", "can", "will", "into", "onto", "than", "then", "when",
    "where", "which", "what", "why", "how", "cua", "cho", "voi", "mot", "cac", "nhung",
    "trong", "ngoai", "neu", "thi", "la", "co", "khong",
}


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-zA-ZÀ-ỹ0-9_]{3,}", text.lower())
        if token not in STOPWORDS
    ]


def infer_section(chunk: str) -> str:
    for line in chunk.splitlines()[:18]:
        cleaned = line.strip()
        if re.match(r"^\d+(\.\d+)*\s+\S+", cleaned):
            return cleaned[:120]
    return "Algorithms for Decision Making"


@lru_cache(maxsize=1)
def load_book_chunks() -> list[dict[str, Any]]:
    if not BOOK_TEXT.exists():
        return []
    pages = BOOK_TEXT.read_text(encoding="utf-8", errors="ignore").split("\f")
    chunks: list[dict[str, Any]] = []
    for page_index, page in enumerate(pages, start=1):
        lines = [line.rstrip() for line in page.splitlines() if line.strip()]
        if not lines:
            continue
        paragraphs: list[str] = []
        current: list[str] = []
        for line in lines:
            current.append(line)
            if len(" ".join(current)) > 1200:
                paragraphs.append(" ".join(current))
                current = []
        if current:
            paragraphs.append(" ".join(current))
        for paragraph_index, text in enumerate(paragraphs):
            terms = tokenize(text)
            if len(terms) < 12:
                continue
            chunks.append(
                {
                    "id": f"p{page_index}-{paragraph_index}",
                    "page": page_index,
                    "section": infer_section(text),
                    "text": text[:1600],
                    "terms": terms,
                    "term_set": set(terms),
                }
            )
    return chunks


def search_book(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    query_terms = tokenize(query)
    if not query_terms:
        return []
    query_set = set(query_terms)
    scored = []
    for chunk in load_book_chunks():
        overlap = query_set & chunk["term_set"]
        if not overlap:
            continue
        tf = sum(1 for term in chunk["terms"] if term in query_set)
        density = tf / math.sqrt(len(chunk["terms"]))
        section_bonus = 0.4 if any(term in chunk["section"].lower() for term in query_set) else 0
        scored.append((density + section_bonus, chunk, overlap))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        {
            "id": chunk["id"],
            "page": chunk["page"],
            "section": chunk["section"],
            "score": round(score, 3),
            "matched_terms": sorted(overlap)[:12],
            "excerpt": chunk["text"][:900],
        }
        for score, chunk, overlap in scored[:top_k]
    ]


def book_context_for_decision(payload: dict[str, Any], top_k: int = 5) -> dict[str, Any]:
    query = " ".join(
        str(payload.get(key, ""))
        for key in ["question", "domain", "objective", "context"]
    )
    results = search_book(query, top_k=top_k)
    summary = "\n".join(
        f"- p.{item['page']} {item['section']}: {item['excerpt'][:280]}"
        for item in results
    )
    return {"results": results, "summary": summary}
