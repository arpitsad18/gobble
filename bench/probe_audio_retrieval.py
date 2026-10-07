"""Audio retrieval probe for the EmbeddingGemma-2 llama-server (CUDA build + mmproj).

Question it answers: does the multimodal tower *read audio*, or just emit a vector
for anything you hand it?  The discriminator matters because synthetic tones are
out-of-distribution and collapse together (tone-vs-noise 0.825, tone-vs-byte-
reversed-tone 0.901), which looks like a dead tower but is really bad input.
Real speech separates cleanly, so this probe uses speech with known text.

Method
------
1. Synthesize three clips with Windows SAPI so the ground truth text is in-code:
   cardio / cardio2 (cardiology) and cooking (unrelated control).
2. Embed each clip via the multimodal `content` array contract:
       {"type": "input_audio", "input_audio": {"data": <b64 wav>, "format": "wav"}}
   (`{"type": "audio_url", ...}` is rejected: HTTP 400 "unsupported content[].type".)
3. Score text->audio and audio->text rankings. Correct top-1 with a wide margin
   on the cooking distractor is the pass condition.

Known-good result on this box (2026-10-07, llama.cpp b11463 CUDA, 768-d):
    cos(cardio, cardio2) = 0.7704   cos(cardio, cooking) = 0.6102
    text "atrial fibrillation anticoagulation stroke prevention"
        -> cardio 0.855 | cardio2 0.755 | cooking 0.569   (top-1 OK)
    text "...cooking pasta with salt" -> cooking 0.828 | cardio 0.492 (top-1 OK)
    audio(cardio) -> "atrial fibrillation..." 0.762 vs "boiling pasta" 0.491

Usage:  python bench/probe_audio_retrieval.py [--url http://127.0.0.1:11436/v1/embeddings]
Writes: bench/results/audio_retrieval.json
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import subprocess
import sys
import urllib.error
import urllib.request

DEFAULT_URL = os.environ.get("GEMMA_URL", "http://127.0.0.1:11436/v1/embeddings")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "audio_retrieval.json")

# Ground truth: (key, spoken text)
CLIPS = [
    ("cardio", "Atrial fibrillation is treated with rate control and anticoagulation for stroke prevention."),
    ("cardio2", "Heart failure with reduced ejection fraction benefits from beta blockers and ACE inhibitors."),
    ("cooking", "Boil two litres of water and add three hundred grams of pasta with a pinch of salt."),
]

TEXT_QUERIES = [
    ("cardio", "atrial fibrillation anticoagulation stroke prevention"),
    ("cooking", "task: search result | query: boiling water and cooking pasta with salt"),
]

# Synthetic-tone control: shows non-speech input is out-of-distribution, not that
# the tower is dead.  Reported, never used as the pass condition.
TONE_CONTROLS = True

PS_SCRIPT = """Add-Type -AssemblyName System.Speech
$out = "{out}"
New-Item -ItemType Directory -Force -Path $out | Out-Null
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.Rate = 0
{lines}
$s.Dispose()
"""


def synth_clips(out_dir: str) -> dict[str, str]:
    """Render CLIPS to wav via Windows SAPI. Returns {key: path}."""
    body = []
    for key, text in CLIPS:
        body.append(f'$s.SetOutputToWaveFile("{out_dir}\\{key}.wav")')
        body.append(f'$s.Speak("{text}")')
    script = PS_SCRIPT.format(out=out_dir.replace("/", "\\"), lines="\n".join(body))
    script_path = os.path.join(out_dir, "synth.ps1")
    os.makedirs(out_dir, exist_ok=True)
    with open(script_path, "w", encoding="utf-8") as fh:
        fh.write(script)
    subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script_path],
        check=True, capture_output=True,
    )
    return {key: os.path.join(out_dir, f"{key}.wav") for key, _ in CLIPS}


class Embedder:
    def __init__(self, url: str) -> None:
        self.url = url

    def _post(self, content: list) -> tuple[int, object]:
        payload = {"model": "gemma", "input": {"content": content}}
        req = urllib.request.Request(
            self.url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                return resp.status, json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode()[:200]

    def embed(self, content: list):
        status, body = self._post(content)
        if status != 200:
            return None, f"HTTP {status} {body}"
        vec = body["data"][0]["embedding"]
        return vec, None

    def audio(self, path: str, fmt: str = "wav"):
        b64 = base64.b64encode(open(path, "rb").read()).decode()
        return self.embed([{"type": "input_audio", "input_audio": {"data": b64, "format": fmt}}])

    def text(self, s: str):
        return self.embed([{"type": "text", "text": s}])


def cos(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--workdir", default=os.path.join(os.environ.get("TMPDIR", "/tmp"), "sapi_audio_probe"))
    ap.add_argument("--no-synth", action="store_true", help="reuse clips already in --workdir")
    args = ap.parse_args()

    if args.no_synth:
        audio_paths = {key: os.path.join(args.workdir, f"{key}.wav") for key, _ in CLIPS}
    else:
        audio_paths = synth_clips(args.workdir)

    emb = Embedder(args.url)
    report: dict = {"url": args.url, "clips": {}, "text_to_audio": {}, "audio_to_text": {}, "controls": {}}

    print("embedding clips ...")
    A: dict[str, list] = {}
    for key, path in audio_paths.items():
        vec, err = emb.audio(path)
        if vec is None:
            print(f"  [FAIL] {key}: {err}")
            report["clips"][key] = {"error": err}
            continue
        A[key] = vec
        report["clips"][key] = {"dim": len(vec), "norm": round(math.sqrt(sum(x * x for x in vec)), 6)}
        print(f"  {key:8s} dim={len(vec)} norm={report['clips'][key]['norm']:.6f}")

    keys = list(A)
    report["audio_to_audio"] = {}
    print("\naudio <-> audio:")
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1:]:
            c = cos(A[k1], A[k2])
            report["audio_to_audio"][f"{k1}|{k2}"] = round(c, 4)
            print(f"  cos({k1:8s}, {k2:8s}) = {c:+.4f}")

    print("\ntext -> audio ranking (top-1 should match the query's topic):")
    hits = 0
    for expect, query in TEXT_QUERIES:
        qv, err = emb.text(query)
        if qv is None:
            print(f"  [FAIL] {query}: {err}")
            continue
        ranked = sorted(((cos(qv, A[k]), k) for k in keys), reverse=True)
        ok = ranked[0][1] == expect
        hits += ok
        report["text_to_audio"][query] = {
            "expect": expect,
            "top1": ranked[0][1],
            "ok": bool(ok),
            "ranking": [[k, round(v, 4)] for v, k in ranked],
        }
        print(f"  {'OK  ' if ok else 'MISS'} {query[:58]:60s} -> "
              + " | ".join(f"{k}:{v:.3f}" for v, k in ranked))

    if keys:
        probe_key = keys[0]
        print(f"\naudio({probe_key}) -> text ranking:")
        for cand in [
            "task: search result | query: atrial fibrillation anticoagulation",
            "task: search result | query: boiling pasta and salt in water",
        ]:
            tv, err = emb.text(cand)
            if tv is None:
                continue
            c = cos(A[probe_key], tv)
            report["audio_to_text"][cand] = round(c, 4)
            print(f"  cos={c:+.4f}  {cand[:62]}")

    if TONE_CONTROLS:
        import io
        import struct
        import wave

        def tone(freq: int, secs: float = 2.0, sr: int = 16000, amp: int = 12000) -> bytes:
            buf = io.BytesIO()
            w = wave.open(buf, "wb")
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(b"".join(
                struct.pack("<h", int(amp * math.sin(2 * math.pi * freq * i / sr))) for i in range(int(sr * secs))
            ))
            w.close()
            return buf.getvalue()

        raw = tone(440)
        a440, _ = emb.embed([{"type": "input_audio",
                              "input_audio": {"data": base64.b64encode(raw).decode(), "format": "wav"}}])
        a880, _ = emb.embed([{"type": "input_audio",
                              "input_audio": {"data": base64.b64encode(tone(880)).decode(), "format": "wav"}}])
        rev = bytearray(raw)
        rev[44:] = reversed(raw[44:])
        arev, _ = emb.embed([{"type": "input_audio",
                              "input_audio": {"data": base64.b64encode(bytes(rev)).decode(), "format": "wav"}}])
        if a440 and a880 and arev:
            report["controls"] = {
                "tone440_vs_tone880": round(cos(a440, a880), 4),
                "tone440_vs_byte_reversed": round(cos(a440, arev), 4),
            }
            print("\nsynthetic-tone controls (out-of-distribution; do not use as a pass condition):")
            for k, v in report["controls"].items():
                print(f"  {k} = {v:+.4f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {OUT}")
    print(f"text->audio top-1: {hits}/{len(TEXT_QUERIES)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
