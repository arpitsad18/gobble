#!/usr/bin/env python
"""ODYS-11 — frozen benchmark: embeddinggemma-2 vs nomic-embed-text.

Answers the issue's question with a reproducible harness:

    python bench_retrieval.py build                 # extract + chunk corpus -> corpus.json
    python bench_retrieval.py embed                 # embed every config -> embeds-<cfg>.npz
    python bench_retrieval.py run                   # metrics -> results/ (json + csv + md)
    python bench_retrieval.py all                   # build + embed + run

Corpus is frozen to roots.txt-style scope: Obsidian vault, Hermes skills,
AI-ECG research output, UCS case-report audit.
Query set is frozen in queries.json (hand-labelled targets, never tuning on it).

Prompt formats (per model family, as published):
  gemma  doc  -> "title: {t} | text: {c}"      query -> "task: search result | query: {q}"
  nomic  doc  -> "search_document: {c}"        query -> "search_query: {q}"
Vectors are L2-normalised after MRL truncation to the config dim, so cosine == dot.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CORPUS_CACHE = HERE / "corpus.json"
QUERIES = HERE / "queries.json"
RESULTS = HERE / "results"

# ---------------------------------------------------------------- corpus scope

ROOTS = [
    "E:/Obsidian Vault",
    "E:/Hermes/Data/skills",
    "E:/Medical/Research",
    "E:/Research/UCS_case_reports",
]

TEXT_EXT = {
    ".md", ".txt", ".markdown", ".rst", ".log", ".csv", ".tsv", ".json", ".yaml",
    ".yml", ".toml", ".ini", ".cfg", ".py", ".js", ".ts", ".html", ".css", ".sql",
    ".sh", ".bat", ".ps1", ".tex", ".bib",
}
OOXML = {".docx", ".xlsx", ".pptx"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", ".cache", "site-packages",
    "$RECYCLE.BIN", "System Volume Information", ".obsidian", ".trash", ".stfolder",
    "dist", "build", ".mypy_cache", ".pytest_cache", ".hub", ".locks",
}
SKIP_EXT = {
    ".exe", ".dll", ".so", ".dylib", ".pyc", ".zip", ".7z", ".rar", ".gz", ".iso",
    ".mp4", ".mkv", ".avi", ".mp3", ".wav", ".flac", ".ttf", ".otf", ".woff", ".woff2",
    ".db", ".sqlite", ".npy", ".npz", ".safetensors", ".gguf", ".bin", ".pak", ".asar",
}
MAX_FILE_BYTES = 4_000_000

CHUNK_CHARS = 1800
CHUNK_OVERLAP = 240
BATCH = 16

# ---------------------------------------------------------------- model configs

CONFIGS = {
    # name -> backend, endpoint, model id, dim, prompt family
    "gemma-768": dict(backend="openai", url=os.environ.get("GEMMA_URL", "http://127.0.0.1:11436/v1/embeddings"),
                      model="embeddinggemma-2-BF16", dim=768, prompts="gemma"),
    "gemma-256": dict(backend="openai", url=os.environ.get("GEMMA_URL", "http://127.0.0.1:11436/v1/embeddings"),
                      model="embeddinggemma-2-BF16", dim=256, prompts="gemma"),
    "nomic-768": dict(backend="ollama", url="http://127.0.0.1:11434/api/embed",
                      model="nomic-embed-text:latest", dim=768, prompts="nomic"),
    "nomic-256": dict(backend="ollama", url="http://127.0.0.1:11434/api/embed",
                      model="nomic-embed-text:latest", dim=256, prompts="nomic"),
}


# ---------------------------------------------------------------- extraction

def _ooxml_text(path: Path) -> str | None:
    import html
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            if path.suffix.lower() == ".docx":
                targets = [n for n in names if n == "word/document.xml"]
            elif path.suffix.lower() == ".xlsx":
                targets = [n for n in names if n == "xl/sharedStrings.xml"]
            else:
                targets = sorted(n for n in names if n.startswith("ppt/slides/slide"))
            out = []
            for n in targets:
                xml = z.read(n).decode("utf-8", "replace")
                xml = xml.replace("</w:p>", "\n").replace("</a:p>", "\n").replace("><", "> <")
                out.append(re.sub(r"\s+\n", "\n", re.sub(r"<[^>]+>", " ", xml)))
            text = html.unescape("\n".join(out))
            return text if text.strip() else None
    except (zipfile.BadZipFile, OSError, KeyError):
        return None


def extract(path: Path) -> tuple[str | None, str]:
    """Return (text, method). method is recorded so gaps are visible in the report."""
    suffix = path.suffix.lower()
    try:
        if suffix in TEXT_EXT:
            return path.read_text(encoding="utf-8", errors="replace"), "read"
        if suffix in OOXML:
            t = _ooxml_text(path)
            return t, "ooxml"
        if suffix == ".pdf":
            return None, "pdf-unsupported"      # pdftotext absent in this sandbox
        if suffix in IMAGE_EXT:
            return None, "image-ocr-unavailable"
    except OSError:
        return None, "error"
    return None, "skipped"


def chunk(text: str) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    if len(text) <= CHUNK_CHARS:
        return [text]
    out, start = [], 0
    while start < len(text):
        end = start + CHUNK_CHARS
        if end < len(text):
            window = text[start:end]
            for sep in ("\n\n", "\n", ". "):
                cut = window.rfind(sep)
                if cut > CHUNK_CHARS * 0.55:
                    end = start + cut + len(sep)
                    break
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return [c for c in out if c]


def walk(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix.lower() in SKIP_EXT or name.startswith("~$"):
                continue
            yield p


def cmd_build(args) -> None:
    roots = [Path(r) for r in (args.roots or ROOTS)]
    chunks: list[dict] = []
    files: list[dict] = []
    methods: dict[str, int] = {}
    for root in roots:
        n_file = n_chunk = 0
        for p in walk(root):
            try:
                st = p.stat()
            except OSError:
                continue
            if st.st_size > MAX_FILE_BYTES:
                methods["too-large"] = methods.get("too-large", 0) + 1
                continue
            text, method = extract(p)
            methods[method] = methods.get(method, 0) + 1
            if not text or not text.strip():
                continue
            pieces = chunk(text)
            if not pieces:
                continue
            title = p.stem
            for i, c in enumerate(pieces):
                chunks.append({"file": str(p), "title": title, "ordinal": i, "text": c})
            files.append({"path": str(p), "root": str(root), "chars": len(text),
                          "chunks": len(pieces)})
            n_file += 1
            n_chunk += len(pieces)
        print(f"[build] {root}: {n_file} files, {n_chunk} chunks")
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "roots": [str(r) for r in roots],
        "chunk_chars": CHUNK_CHARS, "chunk_overlap": CHUNK_OVERLAP,
        "max_file_bytes": MAX_FILE_BYTES,
        "extraction_methods": methods,
        "files": files,
        "chunks": chunks,
    }
    CORPUS_CACHE.write_text(json.dumps(payload), encoding="utf-8")
    print(f"[build] files={len(files)} chunks={len(chunks)} "
          f"chars={sum(f['chars'] for f in files):,} -> {CORPUS_CACHE.name}")
    print(f"[build] extraction methods: {json.dumps(methods, sort_keys=True)}")


# ---------------------------------------------------------------- embedding

def _post(url: str, payload: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def prefix_doc(cfg: dict, title: str, text: str) -> str:
    if cfg["prompts"] == "gemma":
        return f"title: {title or 'none'} | text: {text}"
    if cfg["prompts"] == "nomic":
        return f"search_document: {text}"
    return text


def prefix_query(cfg: dict, text: str) -> str:
    if cfg["prompts"] == "gemma":
        return f"task: search result | query: {text}"
    if cfg["prompts"] == "nomic":
        return f"search_query: {text}"
    return text


def embed_texts(cfg: dict, texts: list[str], batch: int = BATCH,
                dim: int | None = None, raw: bool = False) -> tuple[np.ndarray, float]:
    """Embed (optional MRL-truncate + renormalise) -> (matrix float32, seconds).

    raw=True keeps the model's native dimensionality un-normalised, so a 256d
    row can later be derived by slicing the same vector — MRL truncation and
    re-normalisation commute, and slicing a stored vector avoids embedding the
    whole corpus a second time just to change the output width.
    """
    t0 = time.time()
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        part = texts[i:i + batch]
        if cfg["backend"] == "openai":
            resp = _post(cfg["url"], {"model": cfg["model"], "input": part})
            out.extend(d["embedding"] for d in resp["data"])
        else:
            resp = _post(cfg["url"], {"model": cfg["model"], "input": part})
            out.extend(resp["embeddings"])
    a = np.asarray(out, dtype=np.float32)
    if raw:
        return a, time.time() - t0
    a = a[:, :(dim or cfg["dim"])]
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return a / norms, time.time() - t0


def load_matrix(name: str) -> np.ndarray:
    """Vectors for a config, sliced to its dim and L2-normalised on the fly."""
    z = np.load(HERE / f"embeds-{native_of(name)}.npz", allow_pickle=False)
    raw = z["raw"] if "raw" in z.files else z["vecs"]
    a = np.asarray(raw, dtype=np.float32)[:, :CONFIGS[name]["dim"]]
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return a / norms


def native_of(name: str) -> str:
    """embs are cached per model, not per (model, dim)."""
    return "gemma-native" if name.startswith("gemma") else "nomic-native"


def cmd_embed(args) -> None:
    corpus = json.loads(CORPUS_CACHE.read_text(encoding="utf-8"))
    chunks = corpus["chunks"]
    targets = args.models or ["gemma", "nomic"]
    for which in targets:
        name = "gemma-768" if which.startswith("gemma") else "nomic-768"
        cfg = CONFIGS[name]
        texts = [prefix_doc(cfg, c["title"], c["text"]) for c in chunks]
        m, secs = embed_texts(cfg, texts, raw=True)
        np.savez_compressed(HERE / f"embeds-{native_of(name)}.npz", raw=m,
                            files=np.array([c["file"] for c in chunks]),
                            model=cfg["model"])
        ndoc = len(texts)
        print(f"[embed] {which:6s} native_dim={m.shape[1]:4d} docs={ndoc:4d} "
              f"{secs:7.2f}s  {ndoc / secs:6.1f} doc/s", flush=True)


# ---------------------------------------------------------------- queries

_TOKEN = re.compile(r"[a-z0-9]+")


def norm_path(p: str) -> str:
    """Compare paths the same way on both sides: slashes normalised, lowercase."""
    return p.replace("\\", "/").lower()


def lexical_baseline(query: str, titles: list[str], idf: dict[str, float]) -> int:
    """Where a filename-only scorer would rank the query (rank of best-matching title).

    Reference point, not a vector config: shows how much of the result a plain
    filename matcher already explains, which is exactly what a Spotlight-style
    user already has.
    """
    qt = set(_TOKEN.findall(query.lower()))
    if not qt:
        return 999
    scored = []
    for i, t in enumerate(titles):
        tt = set(_TOKEN.findall(Path(t).stem.lower()))
        overlap = sum(idf.get(w, 1.0) for w in (qt & tt))
        scored.append((overlap, i))
    scored.sort(key=lambda s: (-s[0], s[1]))
    return scored[0][1] + 1 if scored and scored[0][0] > 0 else 999


def cmd_run(args) -> None:
    corpus = json.loads(CORPUS_CACHE.read_text(encoding="utf-8"))
    chunks = corpus["chunks"]
    qset = json.loads(QUERIES.read_text(encoding="utf-8"))
    names = args.configs or list(CONFIGS)

    # chunk index by file for recall@10 chunk-level scoring
    per_file: dict[str, list[int]] = {}
    for i, c in enumerate(chunks):
        per_file.setdefault(norm_path(c["file"]), []).append(i)

    # per-file title vectors are just chunk 0 of each file (one entry per file)
    file_first_chunk: dict[str, int] = {}
    for i, c in enumerate(chunks):
        file_first_chunk.setdefault(norm_path(c["file"]), i)
    file_list = list(file_first_chunk)
    titles = [chunks[file_first_chunk[f]]["title"] for f in file_list]
    df: dict[str, int] = {}
    for t in titles:
        for w in set(_TOKEN.findall(Path(t).stem.lower())):
            df[w] = df.get(w, 0) + 1
    n_docs = max(len(titles), 1)
    idf = {w: np.log(n_docs / c) for w, c in df.items()}

    queries = qset["queries"]
    # queries that are a verbatim file title: a filename matcher already solves
    # these, so they are reported separately from the semantic headline number.
    trivial_ids = {q["id"] for q in queries
                   if any(q["query"].strip().lower() == Path(t).stem.lower()
                          for t in q["targets"])}
    verdict_ids = [q["id"] for q in queries
                   if q["kind"] == "positive" and q["id"] not in trivial_ids]
    print(f"[run] {len(verdict_ids)} semantic positives, "
          f"{len(trivial_ids)} verbatim-title queries ({sorted(trivial_ids)})")
    REPORT = [f"# ODYS-11 retrieval benchmark — {len(chunks)} chunks, "
              f"{len(queries)} frozen queries\n",
              f"Corpus: {', '.join(corpus['roots'])}  ",
              f"chunking: {CHUNK_CHARS} chars / {CHUNK_OVERLAP} overlap  ",
              f"built: {corpus['generated_at']}\n"]

    summary = {"corpus": {"chunks": len(chunks), "files": len(corpus["files"]),
                          "chars": sum(f["chars"] for f in corpus["files"]),
                          "extraction_methods": corpus["extraction_methods"]},
               "query_set": qset.get("name"),
               "trivial_title_queries": sorted(trivial_ids),
               "semantic_positive_ids": verdict_ids,
               "configs": {}}
    per_query_rows: list[dict] = []

    for name in names:
        cfg = CONFIGS[name]
        m = load_matrix(name)
        assert len(m) == len(chunks), f"{name}: embed/corpus length mismatch"

        rows = []
        for q in queries:
            qvec, qsecs = embed_texts(cfg, [prefix_query(cfg, q["query"])], batch=1)
            t0 = time.perf_counter()
            scores = m @ qvec[0]
            order = np.argsort(-scores)
            search_ms = (time.perf_counter() - t0) * 1000

            positives = [norm_path(p) for p in q.get("targets", [])]
            ranked = [norm_path(chunks[int(i)]["file"]) for i in order]
            top1 = chunks[int(order[0])]["file"]

            # path-level (document) ranking — dedup, keep first occurrence
            seen, path_rank = set(), []
            for f in ranked:
                if f not in seen:
                    seen.add(f)
                    path_rank.append(f)
            first_path = next((i + 1 for i, f in enumerate(path_rank) if f in positives), None)
            first_chunk = next((i + 1 for i, f in enumerate(ranked) if f in positives), None)

            hit1 = int(bool(positives) and first_path == 1)
            hit5 = int(first_path is not None and first_path <= 5)
            hit10 = int(first_path is not None and first_path <= 10)
            r1 = 1.0 / first_path if first_path else 0.0
            r5 = (1.0 / first_path) if (first_path and first_path <= 5) else 0.0
            r10 = (1.0 / first_path) if (first_path and first_path <= 10) else 0.0
            # chunk-level recall@10: fraction of gold chunks in the top-10
            gold_idx = {i for p in positives for i in per_file.get(p, [])}
            top10_idx = {int(i) for i in order[:10]}
            crec = (len(gold_idx & top10_idx) / len(gold_idx)) if gold_idx else 0.0

            # negative controls: what is on top when nothing is right?
            neg_top1 = float(scores[int(order[0])]) if q["kind"] == "negative" else None

            rows.append({
                "config": name, "query_id": q["id"], "query": q["query"], "kind": q["kind"],
                "category": q.get("category", ""), "n_targets": len(positives),
                "first_target_rank_path": first_path, "first_target_rank_chunk": first_chunk,
                "hit@1_path": hit1, "hit@5_path": hit5, "hit@10_path": hit10,
                "rr_path": r1, "rr@5": r5, "rr@10": r10,
                "recall@10_chunk": round(crec, 4),
                "top1_path": top1, "top1_score": round(float(scores[int(order[0])]), 4),
                "neg_top1_score": None if neg_top1 is None else round(neg_top1, 4),
                "embed_ms": round(qsecs * 1000, 1), "search_ms": round(search_ms, 2),
                "top5_paths": json.dumps([chunks[int(i)]["file"] for i in order[:5]]),
                "top10": [
                    {"rank": n, "score": round(float(scores[int(i)]), 4),
                     "file": chunks[int(i)]["file"], "ordinal": chunks[int(i)]["ordinal"]}
                    for n, i in enumerate(order[:10], 1)
                ],
            })

        pos = [r for r in rows if r["kind"] == "positive"]
        neg = [r for r in rows if r["kind"] == "negative"]
        ver = [r for r in pos if r["query_id"] in verdict_ids]

        def m(rs, key):
            return round(float(np.mean([r[key] for r in rs])), 4) if rs else None

        agg = {
            "dim": cfg["dim"], "backend": cfg["backend"], "model": cfg["model"],
            "n_positive": len(pos), "n_negative": len(neg),
            "n_semantic_positive": len(ver),
            # headline numbers exclude verbatim-title queries
            "semantic_hit@1": m(ver, "hit@1_path"), "semantic_hit@5": m(ver, "hit@5_path"),
            "semantic_hit@10": m(ver, "hit@10_path"),
            "semantic_recall@10_chunk": m(ver, "recall@10_chunk"),
            "semantic_mrr": m(ver, "rr_path"),
            # full set including the two trivially-matched titles, for transparency
            "all_hit@1": m(pos, "hit@1_path"), "all_hit@5": m(pos, "hit@5_path"),
            "all_hit@10": m(pos, "hit@10_path"),
            "all_recall@10_chunk": m(pos, "recall@10_chunk"),
            "all_mrr": m(pos, "rr_path"),
            "mrr_by_category": {
                c: m([r for r in ver if r["category"] == c], "rr_path")
                for c in sorted({r["category"] for r in ver})
            },
            "hit@10_by_category": {
                c: m([r for r in ver if r["category"] == c], "hit@10_path")
                for c in sorted({r["category"] for r in ver})
            },
            "median_path_rank": float(np.median([r["first_target_rank_path"] or 999
                                                 for r in ver])) if ver else None,
            "embed_ms_median": round(float(np.median([r["embed_ms"] for r in rows])), 1),
            "search_ms_median": round(float(np.median([r["search_ms"] for r in rows])), 2),
            "neg_top1_mean": round(float(np.mean([r["neg_top1_score"] for r in neg])), 4) if neg else None,
            "neg_top1_max": round(float(np.max([r["neg_top1_score"] for r in neg])), 4) if neg else None,
            "pos_top1_mean": round(float(np.mean([r["top1_score"] for r in ver])), 4) if ver else None,
            "pos_top1_min": round(float(np.min([r["top1_score"] for r in ver])), 4) if ver else None,
        }
        summary["configs"][name] = agg
        per_query_rows.extend(rows)
        print(f"[run] {name:10s} semantic hit@1={agg['semantic_hit@1']:.3f} "
              f"hit@5={agg['semantic_hit@5']:.3f} hit@10={agg['semantic_hit@10']:.3f} "
              f"MRR={agg['semantic_mrr']:.3f} "
              f"chunk-recall@10={agg['semantic_recall@10_chunk']:.3f} "
              f"embed={agg['embed_ms_median']:.0f}ms search={agg['search_ms_median']:.2f}ms")

    # ---------------- filename-only reference point
    lex_rows = []
    for q in queries:
        if q["kind"] != "positive":
            continue
        rank = lexical_baseline(q["query"], titles, idf)
        target_file = None
        if rank != 999:
            target_file = file_list[rank - 1]
        ok = bool(target_file and target_file in [norm_path(t) for t in q["targets"]])
        lex_rows.append({"query_id": q["id"], "query": q["query"], "rank": rank if ok else None,
                         "path": target_file})
    lex_hits = [r for r in lex_rows if r["rank"]]
    lex_mrr = float(np.mean([1.0 / r["rank"] for r in lex_hits])) if lex_hits else 0.0
    summary["filename_only_baseline"] = {
        "note": "IDF-weighted filename match, no embeddings; reference point only",
        "hit@10": round(len(lex_hits) / len(lex_rows), 4),
        "mrr": round(lex_mrr, 4),
        "solved": {r["query_id"]: Path(r["path"]).name for r in lex_hits},
    }
    print(f"[run] filename-only baseline hit@10={len(lex_hits)}/{len(lex_rows)} "
          f"MRR={lex_mrr:.3f}")

    # ---------------- paired bootstrap over per-query RR (gemma vs nomic)
    def rr_vector(cfg_name: str) -> np.ndarray:
        return np.array([r["rr_path"] for r in per_query_rows
                         if r["config"] == cfg_name and r["query_id"] in verdict_ids])

    if "gemma-768" in summary["configs"] and "nomic-768" in summary["configs"]:
        a, b = rr_vector("gemma-768"), rr_vector("nomic-768")
        rng = np.random.default_rng(11)
        diffs = []
        n = len(a)
        for _ in range(10000):
            idx = rng.integers(0, n, n)
            diffs.append(a[idx].mean() - b[idx].mean())
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        summary["paired_bootstrap_mrr_gemma768_minus_nomic768"] = {
            "delta": round(float(a.mean() - b.mean()), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "p_delta_le_0": round(float(np.mean(np.array(diffs) <= 0)), 4),
        }
        print("[run] MRR delta (gemma-768 - nomic-768) = "
              f"{a.mean() - b.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")

    RESULTS.mkdir(exist_ok=True)

    # ---------------- report
    lines = list(REPORT)
    lines.append(f"## Headline — {len(verdict_ids)} semantic positives "
                 f"(the {len(trivial_ids)} verbatim-title queries are excluded here)\n")
    lines.append("| config | dim | hit@1 | hit@5 | recall@10 (path) | recall@10 (chunk) | MRR | query emb ms | search ms |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for name, a in summary["configs"].items():
        lines.append(f"| {name} | {a['dim']} | {a['semantic_hit@1']:.3f} | "
                     f"{a['semantic_hit@5']:.3f} | {a['semantic_hit@10']:.3f} | "
                     f"{a['semantic_recall@10_chunk']:.3f} | {a['semantic_mrr']:.3f} | "
                     f"{a['embed_ms_median']:.0f} | {a['search_ms_median']:.2f} |")
    lines.append("\n## By category (MRR / hit@10)\n")
    lines.append("| config | " + " | ".join(sorted(
        {c for a in summary["configs"].values() for c in a["mrr_by_category"]})) + " |")
    lines.append("|---|" + "---|" * len(
        {c for a in summary["configs"].values() for c in a["mrr_by_category"]}))
    for name, a in summary["configs"].items():
        cells = [f"{a['mrr_by_category'].get(c, float('nan')):.2f} / "
                 f"{a['hit@10_by_category'].get(c, float('nan')):.2f}"
                 for c in sorted(a["mrr_by_category"])]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("\n## All positives, including the verbatim-title queries\n")
    lines.append("| config | hit@1 | hit@5 | hit@10 | recall@10 (chunk) | MRR |")
    lines.append("|---|---|---|---|---|---|")
    for name, a in summary["configs"].items():
        lines.append(f"| {name} | {a['all_hit@1']:.3f} | {a['all_hit@5']:.3f} | "
                     f"{a['all_hit@10']:.3f} | {a['all_recall@10_chunk']:.3f} | "
                     f"{a['all_mrr']:.3f} |")
    if "paired_bootstrap_mrr_gemma768_minus_nomic768" in summary:
        bs = summary["paired_bootstrap_mrr_gemma768_minus_nomic768"]
        lines.append(f"\nPaired bootstrap (MRR gemma-768 − nomic-768): "
                     f"{bs['delta']:+.4f}  95% CI [{bs['ci95'][0]:+.4f}, {bs['ci95'][1]:+.4f}]  "
                     f"P(delta ≤ 0) = {bs['p_delta_le_0']:.3f}\n")
    fb = summary["filename_only_baseline"]
    lines.append(f"\nFilename-only reference point (no embeddings): "
                 f"hit@10 = {fb['hit@10']:.3f}, MRR = {fb['mrr']:.3f} — "
                 f"solved {len(fb['solved'])} of the queries by filename alone.\n")
    lines.append("\n## Is a raw cosine threshold usable for the web-search fallback?\n")
    lines.append("Top-1 score ranges per config (positives vs negative controls). "
                 "A usable threshold needs the negative band strictly below the positive band.\n")
    lines.append("| config | positives min / median / max | negatives min / max | overlap |")
    lines.append("|---|---|---|---|")
    threshold_diag = {}
    for n in names:
        pos = sorted(r["top1_score"] for r in per_query_rows
                     if r["config"] == n and r["kind"] == "positive")
        neg = sorted(r["neg_top1_score"] for r in per_query_rows
                     if r["config"] == n and r["kind"] == "negative")
        overlap = max(neg) - min(pos)
        threshold_diag[n] = {"pos_min": min(pos), "pos_median": pos[len(pos) // 2],
                             "pos_max": max(pos), "neg_min": min(neg), "neg_max": max(neg),
                             "overlap": round(overlap, 4)}
        lines.append(f"| {n} | {min(pos):.3f} / {pos[len(pos) // 2]:.3f} / {max(pos):.3f} | "
                     f"{min(neg):.3f} / {max(neg):.3f} | "
                     f"{'YES' if overlap > 0 else 'no'} ({overlap:+.3f}) |")
    summary["threshold_diagnostics"] = threshold_diag
    lines.append("")

    lines.append("\n## Negative controls (nothing in the corpus answers these)\n")
    lines.append("| config | query | top1 score | top1 path |")
    lines.append("|---|---|---|---|")
    for r in per_query_rows:
        if r["kind"] == "negative":
            lines.append(f"| {r['config']} | {r['query']} | {r['neg_top1_score']:.3f} | "
                         f"{Path(r['top1_path']).name} |")
    lines.append("\n## Per-query detail\n")
    lines.append("| query | category | " + " | ".join(f"{n} rank" for n in names) + " |")
    lines.append("|---|---|" + "---|" * len(names))
    for q in queries:
        cells = []
        for n in names:
            row = next((r for r in per_query_rows
                        if r["config"] == n and r["query_id"] == q["id"]), None)
            cells.append("-" if not row or row["first_target_rank_path"] is None
                         else str(row["first_target_rank_path"]))
        lines.append(f"| {q['query']} | {q.get('category', '')} | " + " | ".join(cells) + " |")
    (RESULTS / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RESULTS / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (RESULTS / "per_query.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = [k for k in per_query_rows[0] if k != "top10"]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in per_query_rows:
            w.writerow({k: v for k, v in r.items() if k != "top10"})
    (RESULTS / "per_query_top10.json").write_text(
        json.dumps([{k: v for k, v in r.items() if k in
                     ("config", "query_id", "query", "kind", "top10")}
                    for r in per_query_rows], indent=2), encoding="utf-8")
    print(f"[run] wrote {RESULTS / 'metrics.json'}, per_query.csv, per_query_top10.json, report.md")


def main() -> None:
    ap = argparse.ArgumentParser(prog="bench_retrieval", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pb = sub.add_parser("build"); pb.add_argument("--roots", nargs="*"); pb.set_defaults(func=cmd_build)
    pe = sub.add_parser("embed"); pe.add_argument("--models", nargs="*"); pe.set_defaults(func=cmd_embed)
    pr = sub.add_parser("run"); pr.add_argument("--configs", nargs="*"); pr.set_defaults(func=cmd_run)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
