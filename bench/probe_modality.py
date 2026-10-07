#!/usr/bin/env python
"""ODYS-11 modality probe: what happens to image / scanned-PDF queries.

The main benchmark corpus is text-only (PDF extraction and Windows OCR are both
unavailable to the agent sandbox). This probe asks the two models the image and
scan questions anyway and records what they retrieve, so the failure mode is
documented rather than hidden: the shared-vector-space claim for images/audio
cannot be tested on this machine at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import bench_retrieval as B

HERE = Path(__file__).resolve().parent


def main() -> None:
    corpus = json.loads(B.CORPUS_CACHE.read_text(encoding="utf-8"))
    chunks = corpus["chunks"]
    qset = json.loads(B.QUERIES.read_text(encoding="utf-8"))
    probes = qset["missing_modality_probe"]

    out = {"note": __doc__.strip(), "targets_indexed": {}, "results": []}
    for p in probes:
        out["targets_indexed"][p["id"]] = [
            {"target": t, "in_index": any(c["file"].lower() == t.lower() for c in chunks)}
            for t in p["targets"]
        ]

    for name in ("gemma-768", "nomic-768"):
        cfg = B.CONFIGS[name]
        m = B.load_matrix(name)
        for p in probes:
            qvec, _ = B.embed_texts(cfg, [B.prefix_query(cfg, p["query"])], batch=1)
            scores = m @ qvec[0]
            order = scores.argsort()[::-1]
            top = [{"rank": n, "score": round(float(scores[int(i)]), 4),
                    "file": chunks[int(i)]["file"]}
                   for n, i in enumerate(order[:5], 1)]
            gold = [t.lower() for t in p["targets"]]
            hit = any(chunks[int(i)]["file"].lower() in gold for i in order[:10])
            out["results"].append({"config": name, "probe": p["id"], "query": p["query"],
                                   "target_in_index": out["targets_indexed"][p["id"]],
                                   "hit@10": hit, "top5": top})
            print(f"[probe] {name} {p['id']}: hit@10={hit}  top1={Path(top[0]['file']).name} "
                  f"({top[0]['score']:.3f})")

    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "modality_probe.json").write_text(json.dumps(out, indent=2),
                                                          encoding="utf-8")


if __name__ == "__main__":
    main()
