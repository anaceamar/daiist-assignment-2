"""
Retrieval over the saved index: dense (Gemini embeddings), BM25, hybrid and a
transport-mode metadata filter. Used by server.py (search tool) and evaluate.py.

Search designs compared in evaluate.py:
    fixed            dense search over 250-word windows
    structure        dense search over legal-unit chunks
    structure-hybrid dense + BM25 over legal-unit chunks, fused with RRF
    structure-hybrid-mode   the same, filtered to the transport mode the query names

Only the first two need their own embeddings; the others reuse the
'structure' vectors, so they cost no extra quota.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from functools import lru_cache

import numpy as np

import index as pipeline

SEARCH_DESIGNS = ("fixed", "structure", "structure-hybrid", "structure-hybrid-mode")
DEFAULT_DESIGN = "structure-hybrid"
QUERIES_FILE = pipeline.INDEX_DIR / "queries.jsonl"

STOPWORDS = set("""a an and are as at be by can do does for from has have how i if in into is it its
my of on or should that the their them there these they this to was what when where which who will
with would you your me am our any""".split())

MODE_WORDS = {
    "air": r"\b(flights?|fly|flying|plane|aircraft|airlines?|air carriers?|airports?|boarding|air)\b",
    "rail": r"\b(trains?|rail|railways?|stations?)\b",
    "bus": r"\b(bus|buses|coach|coaches)\b",
    "sea": r"\b(ships?|ferry|ferries|cruises?|boats?|ports?|sea|waterways?|vessels?)\b",
}


def chosen_design() -> str:
    try:
        return json.loads(pipeline.CONFIG_FILE.read_text(encoding="utf-8")).get("search_design", DEFAULT_DESIGN)
    except FileNotFoundError:
        return DEFAULT_DESIGN


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def query_mode(query: str) -> str | None:
    """The single transport mode a query names, or None (none or several)."""
    found = [m for m, pattern in MODE_WORDS.items() if re.search(pattern, query.lower())]
    return found[0] if len(found) == 1 else None


class Index:
    """One chunk design loaded into memory: chunk dicts, a vector matrix, BM25 stats."""

    def __init__(self, design: str):
        self.design = design
        self.chunks = pipeline.load_chunks(design)
        vectors = pipeline.load_vectors()
        missing = [c["id"] for c in self.chunks if c["hash"] not in vectors]
        if missing:
            pipeline.log(f"[{design}] {len(missing)} chunks have no vector yet and are skipped. "
                         f"Run `uv run python main.py index` to embed them.")
            self.chunks = [c for c in self.chunks if c["hash"] in vectors]
        if not self.chunks:
            raise RuntimeError(f"The '{design}' index is empty. Run `uv run python main.py index` first.")
        self.matrix = np.array([vectors[c["hash"]] for c in self.chunks], dtype=np.float32)
        # BM25 statistics
        self.docs = [tokens(c["embed_text"]) for c in self.chunks]
        self.lengths = np.array([len(d) for d in self.docs], dtype=np.float32)
        self.avg_len = float(self.lengths.mean())
        self.tf = [Counter(d) for d in self.docs]
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def dense_scores(self, query_vector: np.ndarray) -> np.ndarray:
        return self.matrix @ query_vector

    def bm25_scores(self, query: str, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        for term in set(tokens(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(term)
                if f:
                    scores[i] += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * self.lengths[i] / self.avg_len))
        return scores


@lru_cache(maxsize=4)
def load_index(design: str) -> Index:
    return Index(design)


def reset_cache() -> None:
    """Forget loaded indexes, e.g. after add_document changed the chunk files."""
    load_index.cache_clear()


# --------------------------------------------------------------------------- query embeddings

def _load_query_cache() -> dict[str, list[float]]:
    cache = {}
    if QUERIES_FILE.exists():
        for line in QUERIES_FILE.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                cache[row["h"]] = row["v"]
            except (json.JSONDecodeError, KeyError):
                continue
    return cache


_query_cache: dict[str, list[float]] | None = None


def embed_query(query: str) -> np.ndarray:
    """Embed a query once and keep it: re-running the evaluation costs no quota."""
    global _query_cache
    if _query_cache is None:
        _query_cache = _load_query_cache()
    key = pipeline.text_hash("query|" + query.strip())
    if key not in _query_cache:
        vector = pipeline.embed_texts([query.strip()], "RETRIEVAL_QUERY")[0]
        _query_cache[key] = vector
        QUERIES_FILE.parent.mkdir(parents=True, exist_ok=True)
        with QUERIES_FILE.open("a", encoding="utf-8") as out:
            out.write(json.dumps({"h": key, "q": query.strip(), "v": vector}) + "\n")
    return np.array(_query_cache[key], dtype=np.float32)


# --------------------------------------------------------------------------- search

def search(query: str, k: int = 5, design: str | None = None, include_added: bool = True,
           mode: str | None = None) -> list[dict]:
    """Top-k chunks for a query. Each result is the chunk dict plus 'score' and 'rank'."""
    design = design or chosen_design()
    base = "fixed" if design.startswith("fixed") else "structure"
    idx = load_index(base)
    query_vector = embed_query(query)

    dense = idx.dense_scores(query_vector)
    if design.endswith("hybrid") or design.endswith("hybrid-mode"):
        # Reciprocal rank fusion of the dense and the BM25 rankings.
        bm25 = idx.bm25_scores(query)
        scores = np.zeros_like(dense)
        for ranking in (np.argsort(-dense), np.argsort(-bm25)):
            for rank, i in enumerate(ranking[:100]):
                scores[i] += 1.0 / (60 + rank + 1)
    else:
        scores = dense

    allowed = np.ones(len(idx.chunks), dtype=bool)
    if not include_added:
        allowed &= np.array([not c.get("added") for c in idx.chunks])
    if mode is None and design.endswith("-mode"):
        mode = query_mode(query)
    if mode:
        allowed &= np.array([c["mode"] == mode for c in idx.chunks])
    scores = np.where(allowed, scores, -np.inf)

    results = []
    for rank, i in enumerate(np.argsort(-scores)[:k], 1):
        if not np.isfinite(scores[i]):
            break
        results.append({**idx.chunks[i], "score": float(scores[i]), "rank": rank})
    return results


def cite(chunk: dict) -> str:
    return f"[{chunk['source']}, {chunk['location']}]"


def format_results(results: list[dict]) -> str:
    """The text the search tool returns: each chunk with how to cite it."""
    if not results:
        return "No passages found."
    blocks = []
    for r in results:
        blocks.append(
            f"--- Result {r['rank']} | cite as {cite(r)}\n"
            f"Document: {r['title']}\nMode: {r['mode']}\n\n{r['text']}"
        )
    return "\n\n".join(blocks)


CITATION = re.compile(r"\[([a-z0-9][a-z0-9-]+),\s*([^\]]+?)\]")


def parse_citations(text: str) -> list[tuple[str, str]]:
    return [(s.strip(), l.strip()) for s, l in CITATION.findall(text)]


def citation_matches(cited: tuple[str, str], retrieved: tuple[str, str]) -> bool:
    """A citation is valid if it names a retrieved chunk's source and location,
    allowing a paragraph or point: 'Article 7(1)(b)' matches 'Article 7'."""
    (cs, cl), (rs, rl) = cited, retrieved
    if cs != rs:
        return False
    cl = re.sub(r"\s*\(part \d+/\d+\)", "", cl).strip().rstrip(".")
    return cl == rl or (cl.startswith(rl) and not cl[len(rl)].isalnum() and cl[len(rl)] != ".")


def retrieved_from_tool_output(text: str) -> list[tuple[str, str]]:
    return [(s.strip(), l.strip()) for s, l in re.findall(r"cite as \[([^,\]]+),\s*([^\]]+)\]", text)]


def location_hit(chunk: dict, source: str, location: str) -> bool:
    """Does this chunk contain the expected passage? 'Section 4.4' matches
    'Section 4.4.2'; 'Article 7' matches 'Article 7' but not 'Article 70'."""
    if chunk["source"] != source:
        return False
    location = location.strip()
    for unit in chunk["units"]:
        if unit == location or unit.startswith(location + ".") or unit.startswith(location + "("):
            return True
    return False
