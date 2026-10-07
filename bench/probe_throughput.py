#!/usr/bin/env python
"""ODYS-11 indexing-throughput probe — the metric that decides the embedder.

Same 300-chunk sample, same batch size, same machine, one model at a time, so
the comparison is not distorted by two servers competing for the GPU. Reports
tokens/s and the projected wall-clock for the real corpus size in the index.

    python probe_throughput.py            # both models
    python probe_throughput.py gemma      # one model
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import bench_retrieval as B

HERE = Path(__file__).resolve().parent
SAMPLE = 300
REPEATS = 3


def run_model(which: str) -> dict:
    corpus = json.loads(B.CORPUS_CACHE.read_text(encoding="utf-8"))
    chunks = corpus["chunks"]
    rng = np.random.default_rng(11)
    idx = np.sort(rng.choice(len(chunks), size=min(SAMPLE, len(chunks)), replace=False))
    sample = [chunks[int(i)] for i in idx]

    name = "gemma-768" if which.startswith("gemma") else "nomic-768"
    cfg = B.CONFIGS[name]
    texts = [B.prefix_doc(cfg, c["title"], c["text"]) for c in sample]
    tokens = sum(len(t) for t in texts) / 4.0      # ~4 chars/token, rough but consistent

    times = []
    for _ in range(REPEATS):
        _, secs = B.embed_texts(cfg, texts, raw=True)
        times.append(secs)
    best = min(times)
    mean = sum(times) / len(times)
    res = {
        "model": cfg["model"], "backend": cfg["backend"], "docs": len(texts),
        "approx_tokens": int(tokens), "repeats": REPEATS,
        "seconds_best": round(best, 2), "seconds_mean": round(mean, 2),
        "docs_per_s": round(len(texts) / best, 1),
        "tokens_per_s": round(tokens / best, 1),
        "full_corpus_projection_s": round(len(chunks) / (len(texts) / best), 1),
    }
    print(f"[throughput] {which:6s} {res['docs_per_s']:6.1f} doc/s  "
          f"{res['tokens_per_s']:7.1f} tok/s  "
          f"(full {len(chunks)}-chunk corpus takes about "
          f"{res['full_corpus_projection_s'] / 60:.1f} min)",
          flush=True)
    return res


def main() -> None:
    which = sys.argv[1:] or ["gemma", "nomic"]
    out = [run_model(w) for w in which]
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "throughput.json").write_text(
        json.dumps({"sample_size": SAMPLE, "repeats": REPEATS, "runs": out}, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
