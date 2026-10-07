#!/usr/bin/env python
"""ODYS-11 batch-size sweep for the Gemma transport.

llama-server logs one task per slot per request, and Gemma measured 2.1 doc/s at
batch 16 while nomic on Ollama measured 66 doc/s on the same machine. This sweep
tests whether the gap is a batching artefact of the llama.cpp transport (which
would be fixable in the indexer) rather than a property of the model.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import bench_retrieval as B

HERE = Path(__file__).resolve().parent
SAMPLE = 48


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "gemma"
    name = "gemma-768" if model.startswith("gemma") else "nomic-768"
    cfg = B.CONFIGS[name]
    corpus = json.loads(B.CORPUS_CACHE.read_text(encoding="utf-8"))
    chunks = corpus["chunks"]
    rng = np.random.default_rng(7)
    idx = np.sort(rng.choice(len(chunks), size=SAMPLE, replace=False))
    texts = [B.prefix_doc(cfg, chunks[int(i)]["title"], chunks[int(i)]["text"]) for i in idx]

    out = []
    for batch in (1, 2, 4, 8, 16, 32):
        times = []
        for _ in range(2):
            _, secs = B.embed_texts(cfg, texts, batch=batch, raw=True)
            times.append(secs)
        best = min(times)
        row = {"batch": batch, "docs_per_s": round(SAMPLE / best, 2),
               "ms_per_doc": round(best / SAMPLE * 1000, 1)}
        out.append(row)
        print(f"[sweep] {model} batch={batch:3d}  {row['docs_per_s']:6.2f} doc/s  "
              f"{row['ms_per_doc']:7.1f} ms/doc", flush=True)

    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"batch_sweep_{model}.json").write_text(
        json.dumps({"model": cfg["model"], "endpoint": cfg["url"], "sample": SAMPLE,
                    "rows": out}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
