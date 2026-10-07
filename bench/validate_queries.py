#!/usr/bin/env python
"""Validate the frozen query set against the frozen corpus (no GPU, no network).

Per query it checks:
  * every labelled target resolves inside the frozen corpus (typo / scope guard)
  * how many corpus files share that exact title (bare-title queries)
  * lexical leakage: is the query a literal substring of its target, and what
    share of its content words also occur in the target
A query that is literally contained in one file is not a semantic test, so those
are flagged and their contribution to the headline MRR is reported separately.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOKEN = re.compile(r"[a-z0-9]{3,}")
STOP = {"the", "and", "for", "with", "that", "this", "how", "does", "from", "into",
        "when", "what", "which", "there", "their", "your", "you", "are", "not", "but",
        "have", "has", "can", "must", "any", "one", "two", "its", "out", "who", "why"}


def norm(p: str) -> str:
    return p.replace("\\", "/").lower()


def words(text: str) -> set[str]:
    return {w for w in TOKEN.findall(text.lower()) if w not in STOP}


def main() -> None:
    corpus = json.loads((HERE / "corpus.json").read_text(encoding="utf-8"))
    qset = json.loads((HERE / "queries.json").read_text(encoding="utf-8"))
    chunks = corpus["chunks"]
    by_file: dict[str, list[str]] = {}
    for c in chunks:
        by_file.setdefault(norm(c["file"]), []).append(c["text"])
    titles = [c["title"].lower() for c in chunks]

    problems, flagged = [], []
    print(f"{'id':8s} {'kind':9s} {'targets':>7s} {'found':>5s} {'title#':>6s} "
          f"{'substr':>6s} {'wordcov':>7s}  flags")
    for q in qset["queries"] + qset["missing_modality_probe"]:
        targets = [norm(t) for t in q["targets"]]
        present = [t for t in targets if t in by_file]
        missing = [t for t in targets if t not in by_file]
        if missing and q["kind"] != "modality-probe":
            problems.append(f"{q['id']}: target not in corpus: {missing}")
        if missing and q["kind"] == "modality-probe":
            flagged.append(f"{q['id']}: modality target absent from the text index (expected)")
        ql = q["query"].lower()
        substr_hit = any(ql in t.lower() for t in [x for t in present for x in by_file[t]])
        qw = words(q["query"])
        tgt_words: set[str] = set()
        for t in present:
            tgt_words |= words(" ".join(by_file[t]))
        cov = len(qw & tgt_words) / len(qw) if qw else 0.0
        title_n = titles.count(ql.strip())
        flags = []
        if substr_hit:
            flags.append("LITERAL-SUBSTRING")
            flagged.append(f"{q['id']}: query text appears verbatim in the target")
        if cov >= 0.9:
            flags.append("HIGH-WORD-OVERLAP")
        print(f"{q['id']:8s} {q['kind']:9s} {len(targets):7d} {len(present):5d} "
              f"{title_n:6d} {str(substr_hit):>6s} {cov:7.2f}  {' '.join(flags)}")

    print()
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print("  -", p)
    else:
        print("OK: every non-probe target resolves inside the frozen corpus")
    if flagged:
        print("\nNOTES:")
        for f in flagged:
            print("  -", f)

    print("\nnegative controls — closest files sharing >=2 content words:")
    for q in qset["queries"]:
        if q["kind"] != "negative":
            continue
        qw = words(q["query"])
        scored = sorted(((len(qw & words(" ".join(t))), f) for f, t in by_file.items()),
                        key=lambda s: -s[0])[:3]
        print(f"  {q['id']} ({len(qw)} content words): "
              + ", ".join(f"{Path(f).name}({n})" for n, f in scored))


if __name__ == "__main__":
    main()
