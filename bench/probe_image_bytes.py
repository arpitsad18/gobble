#!/usr/bin/env python
"""Is the 200 on a base64 data URI real image understanding, or the text tower
happily embedding a long string of base64? Compare three vectors."""

from __future__ import annotations

import base64
import json
import urllib.request

import numpy as np


def embed(texts, url="http://127.0.0.1:11434/api/embed",
          model="nomic-embed-text:latest", openai=False):
    payload = {"model": "local" if openai else model, "input": texts}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        body = json.loads(r.read().decode())
    vecs = [d["embedding"] for d in body["data"]] if openai else body["embeddings"]
    a = np.asarray(vecs, dtype=np.float32)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    return a


img = open("E:/BOOKS/Academia/Motherlode/Sketchy/Pharm/Cardiovascular & Renal/"
           "Picture Drop CV & Renal/2.2 - Loop diuretics.png", "rb").read()
uri = "data:image/png;base64," + base64.b64encode(img).decode()
shuffled = "data:image/png;base64," + base64.b64encode(img[::-1]).decode()

for label, kw in (("ollama/nomic", dict()), ):
    v = embed(["a photo of a handwritten pharmacology diagram", uri, shuffled], **kw)
    cos_text_uri = float(v[0] @ v[1])
    cos_text_shuffled = float(v[0] @ v[2])
    cos_uri_shuffled = float(v[1] @ v[2])
    print(f"{label}: cos(text, real-uri)={cos_text_uri:.4f}  "
          f"cos(text, byte-reversed-uri)={cos_text_shuffled:.4f}  "
          f"cos(real, reversed)={cos_uri_shuffled:.4f}")
    print("  -> if the two data URIs are near-identical, the server is hashing/"
          "embedding the base64 text, not looking at the picture")

v = embed(["a photo of a handwritten pharmacology diagram", uri], openai=True,
          url="http://127.0.0.1:11435/v1/embeddings")
print(f"llama/gemma: cos(text, uri)={float(v[0] @ v[1]):.4f}")
