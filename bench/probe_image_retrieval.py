#!/usr/bin/env python
"""Cross-modal probe: can EmbeddingGemma-2 (via llama.cpp) retrieve an image
from a TEXT query, and vice versa?

EmbeddingGemma-2 is embedding+vision+audio in one 768-d space, so the same
server that embeds documents also embeds images — provided the request uses the
multimodal `content` array shape that llama-server accepts:

    {"input": {"content": [{"type": "image_url",
                            "image_url": {"url": "data:image/png;base64,..."}}]}}

(index into /v1/embeddings, NOT a bare data-URI string — a bare string is
tokenized as ~1M chars of text and silently "works" with garbage vectors.)

    python probe_image_retrieval.py                 # default sample image folder
    python probe_image_retrieval.py <image_dir>     # any folder of .png/.jpg

Writes bench/results/image_retrieval.json and caches image vectors next to it
so re-ranking against new queries costs no image re-encoding.
"""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_DIR = os.environ.get("GOBBLE_IMAGE_DIR", "<library>/cards")
URL = os.environ.get("LOCALSEARCH_URL", "http://127.0.0.1:11436/v1/embeddings")
CACHE = HERE / "results" / "image_index.json"
EXTS = {".png", ".jpg", ".jpeg", ".webp"}

# text-side queries keyed to the card they *should* retrieve
QUERIES = {
    "2.2 - Loop diuretics": "task: search result | query: loop diuretics furosemide",
    "1.2 - ACE inhibitors, ARBs, Aliskiren": "task: search result | query: ACE inhibitor ramipril ARB losartan renin inhibitor aliskiren",
    "2.1 - Acetazolamide, mannitol": "task: search result | query: carbonic anhydrase inhibitor acetazolamide mannitol osmotic diuretic",
    "2.3 - Thiazides": "task: search result | query: thiazide diuretic hydrochlorothiazide",
    "2.4 - K+ sparing diuretics": "task: search result | query: potassium sparing diuretic spironolactone aldosterone antagonist",
    "3.1 - Calcium channel blockers": "task: search result | query: calcium channel blocker dihydropyridine verapamil diltiazem",
    "3.2 - Primary hypertension & hypertensive emergency": "task: search result | query: primary hypertension hypertensive emergency management",
    "4.1 - Class I A-C": "task: search result | query: class IA IB IC sodium channel blocker antiarrhythmic",
    "4.2 - Class II": "task: search result | query: class II beta blocker antiarrhythmic",
    "4.3 - Class III": "task: search result | query: class III potassium channel blocker amiodarone sotalol",
    "4.4 - Class IV": "task: search result | query: class IV calcium channel blocker antiarrhythmic",
    "1.1 - Digoxin, milrinone, nesiritide": "task: search result | query: digoxin milrinone nesiritide positive inotrope heart failure",
}


def post(payload: dict) -> dict:
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError:
            raise
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


def embed_images(paths: list[Path]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    t0 = time.time()
    for i, p in enumerate(paths, 1):
        mime = mimetypes.guess_type(p.name)[0] or "image/png"
        uri = f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()
        item = {"type": "image_url", "image_url": {"url": uri}}
        r = post({"model": "gemma", "input": {"content": [item]}})
        out[p.stem] = r["data"][0]["embedding"]
        print(f"  [{i}/{len(paths)}] {p.stem[:42]:44s} dim={len(out[p.stem])}", flush=True)
    dt = time.time() - t0
    print(f"[index] {len(paths)} images in {dt:.1f}s -> {len(paths)/dt:.2f} img/s", flush=True)
    return out


def embed_texts(texts: list[str]) -> list[list[float]]:
    r = post({"model": "gemma", "input": texts})
    return [d["embedding"] for d in r["data"]]


def cos(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d else 0.0


def main() -> None:
    img_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_DIR)
    paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in EXTS)

    if CACHE.exists():
        cache = json.loads(CACHE.read_text(encoding="utf-8"))
    else:
        cache = {"dir": str(img_dir), "vectors": {}}
    known = cache.get("vectors", {})
    todo = [p for p in paths if p.stem not in known]
    if todo:
        print(f"[index] embedding {len(todo)} new image(s) via {URL}", flush=True)
        known.update(embed_images(todo))
    cache.update({"dir": str(img_dir), "vectors": known})
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_text(json.dumps(cache), encoding="utf-8")

    keys = [p.stem for p in paths if p.stem in known]
    print(f"\n[index] {len(keys)} images cached\n", flush=True)

    qs = {k: v for k, v in QUERIES.items() if k in known}
    tvecs = embed_texts(list(qs.values()))

    rows, hits, mrr = [], 0, 0.0
    print("text query -> image ranking")
    for (want, q), qv in zip(qs.items(), tvecs):
        ranked = sorted(((cos(qv, known[k]), k) for k in keys), reverse=True)
        rank = next(i for i, (_, k) in enumerate(ranked, 1) if k == want)
        hits += rank == 1
        mrr += 1.0 / rank
        flag = "OK  " if rank == 1 else ("top3" if rank <= 3 else "MISS")
        print(f"  {flag} rank={rank:2d} cos={ranked[0][0]:+.3f} | want={want[:34]:36s}"
              f" got={ranked[0][1][:34]:36s}", flush=True)
        rows.append({"query": want, "rank": rank,
                     "top5": [{"card": k, "cos": round(s, 4)} for s, k in ranked[:5]]})

    n = len(qs)
    summary = {"images": keys, "n_queries": n, "top1": hits,
               "top1_acc": round(hits / n, 3), "mrr": round(mrr / n, 3),
               "results": rows}
    (HERE / "results" / "image_retrieval.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[summary] top1={hits}/{n} ({hits/n:.0%})  MRR={mrr/n:.3f}"
          f"  -> results/image_retrieval.json")


if __name__ == "__main__":
    main()
