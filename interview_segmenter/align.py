"""Locating questionnaire items in the interviewer's transcript.

Two stages: local matching proposes candidate unit runs per item (embedding
similarity or an LLM), global matching (`solve`) picks at most one candidate
per item so that the picked runs follow the questionnaire order without overlap.

    align(items, transcript) -> list[Match]      (unit indices; times via core.attach_times)
"""

import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

import networkx as nx

from .core import Item, Match, Transcript

EMBEDDING_MODEL = "T-Systems-onsite/cross-en-de-roberta-sentence-transformer"   # German/English
_models = {}


def embed(texts: list[str], model: str = EMBEDDING_MODEL):
    if model not in _models:
        from sentence_transformers import SentenceTransformer
        _models[model] = SentenceTransformer(model)
    return _models[model].encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                                 show_progress_bar=False)


def solve(items: list[Item], cands: dict[str, list[Match]], floor: float = 0.6) -> list[Match]:
    """Global matching as a longest path in a DAG: nodes are candidates worth score - floor,
    an edge A -> B exists when B's item comes later in the questionnaire AND B starts
    after A ends. Items with no node on the path count as not asked."""
    order = {it.item_id: k for k, it in enumerate(items)}
    nodes = [m for ms in cands.values() for m in ms if m.score > floor]
    G = nx.DiGraph()
    G.add_node("start")
    for m in nodes:
        G.add_edge("start", id(m), weight=m.score - floor)
    for a, b in combinations(sorted(nodes, key=lambda m: m.from_unit), 2):
        if order[b.item_id] > order[a.item_id] and b.from_unit >= a.to_unit:
            G.add_edge(id(a), id(b), weight=b.score - floor)
    by_id = {id(m): m for m in nodes}
    return [by_id[k] for k in nx.dag_longest_path(G, weight="weight") if k != "start"]


# --- embedding ----------------------------------------------------------------------

def embedding_candidates(items: list[Item], tr: Transcript, top_n: int = 20, max_span: int = 4,
                         model: str = EMBEDDING_MODEL) -> dict[str, list[Match]]:
    """Local matching: per item, the top_n most similar runs of 1..max_span units (cosine).
    Every run is its own candidate, so a long run cannot hide a shorter one inside it."""
    n = len(tr.units)
    runs = [(a, b) for a in range(n) for b in range(a + 1, min(a + max_span, n) + 1)]
    sim = embed([it.text for it in items], model) @ embed(
        [" ".join(u.text for u in tr.units[a:b]) for a, b in runs], model).T
    return {it.item_id: [Match(it.item_id, *runs[r], score=float(sim[i, r]))
                         for r in sim[i].argsort()[::-1][:top_n]]
            for i, it in enumerate(items)}


def embedding_align(items: list[Item], tr: Transcript, top_n: int = 20, max_span: int = 4,
                    floor: float = 0.6, model: str = EMBEDDING_MODEL) -> list[Match]:
    return solve(items, embedding_candidates(items, tr, top_n, max_span, model), floor)


# --- LLM ------------------------------------------------------------------------------

LLM_URL = os.environ.get("LLM_URL", "http://localhost:8000/v1/chat/completions")   # any OpenAI-compatible server
LLM_MODEL = os.environ.get("LLM_MODEL", "")

LLM_SYSTEM = (
    "You are given the transcript of an interview, one numbered line per utterance of "
    "the interviewer. The interviewer reads scripted questionnaire items aloud, sometimes "
    "paraphrased, shortened, or split across several lines, and sometimes not at all "
    "(conditional questions). Find the lines where a given item is read.")

LLM_SCHEMA = {
    "type": "object",
    "properties": {"occurrences": {"type": "array", "maxItems": 5, "items": {
        "type": "object",
        "properties": {"first_line": {"type": "integer"}, "last_line": {"type": "integer"}},
        "required": ["first_line", "last_line"]}}},
    "required": ["occurrences"],
}


def ask_llm(transcript: str, item: Item, url: str = LLM_URL, model: str = LLM_MODEL) -> list[tuple[int, int]]:
    """One item, one request. The transcript comes first so all requests share a prefix
    (prefix caching); needs a server that supports JSON-schema output (e.g. vLLM)."""
    body = {"model": model, "temperature": 0, "max_tokens": 400,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "occurrences", "schema": LLM_SCHEMA}},
            "messages": [
                {"role": "system", "content": LLM_SYSTEM},
                {"role": "user", "content":
                    f"Transcript:\n{transcript}\n\nScripted item:\n«{item.text}»\n\n"
                    "Return every line range where the interviewer reads this item. "
                    "Return an empty list if it is never read."}]}
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        out = json.loads(json.loads(r.read())["choices"][0]["message"]["content"])
    return [(o["first_line"], o["last_line"]) for o in out["occurrences"]]


def llm_align(items: list[Item], tr: Transcript, url: str = LLM_URL, model: str = LLM_MODEL,
              workers: int = 16) -> list[Match]:
    """Local matching by the LLM (every proposed range worth 1), global matching by `solve`."""
    n = len(tr.units)
    transcript = "\n".join(f"[{k}] {u.text}" for k, u in enumerate(tr.units))
    with ThreadPoolExecutor(workers) as pool:
        answers = pool.map(lambda it: ask_llm(transcript, it, url, model), items)
    cands = {it.item_id: [Match(it.item_id, a, b + 1, score=1.0) for a, b in spans if 0 <= a <= b < n]
             for it, spans in zip(items, answers)}
    return solve(items, cands, floor=0.0)


ALIGNERS = {"embedding": embedding_align, "llm": llm_align}
