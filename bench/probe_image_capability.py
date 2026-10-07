#!/usr/bin/env python
"""ODYS-11: can this embedding transport see an image at all?

EmbeddingGemma's headline advantage for this project is one shared vector space
for text + image + audio. That needs a multimodal projection, not just the text
tower. This probe asks both transports directly.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request

IMG = ("E:/BOOKS/Academia/Motherlode/Sketchy/Pharm/Cardiovascular & Renal/"
       "Picture Drop CV & Renal/2.2 - Loop diuretics.png")


def post(url: str, payload: dict, timeout: int = 120):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:400]


def main() -> None:
    raw = open(IMG, "rb").read()
    data_uri = "data:image/png;base64," + base64.b64encode(raw).decode()

    results = {}
    for label, payload in (
        ("text_control", {"model": "local", "input": ["loop diuretics act on the thick ascending limb"]}),
        ("data_uri_image", {"model": "local", "input": [data_uri]}),
        ("openai_content_parts", {"model": "local", "input": [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_uri}}]}]}),
    ):
        code, body = post("http://127.0.0.1:11435/v1/embeddings", payload)
        dim = (len(body["data"][0]["embedding"])
               if code == 200 and isinstance(body, dict) and body.get("data") else None)
        results[f"llama-server/{label}"] = {"http": code, "dim": dim,
                                            "body": body if dim is None else "ok"}
        print(f"[img-probe] llama-server {label:22s} http={code} dim={dim}")

    code, body = post("http://127.0.0.1:11434/api/embed",
                      {"model": "nomic-embed-text:latest", "input": [data_uri]})
    dim = len(body["embeddings"][0]) if code == 200 and isinstance(body, dict) else None
    results["ollama/data_uri_image"] = {"http": code, "dim": dim,
                                        "body": body if dim is None else "ok"}
    print(f"[img-probe] ollama nomic data_uri_image  http={code} dim={dim}")

    with open("results/image_capability.json", "w", encoding="utf-8") as fh:
        json.dump({"image": IMG, "image_bytes": len(raw), "results": results}, fh, indent=2)


if __name__ == "__main__":
    main()
