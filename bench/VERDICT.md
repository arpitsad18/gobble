# Verdict — Gemma now works (text + images + audio) on this box; the embedder call is a ~10% question, not a 30× one

Frozen benchmark, 2026-10-07. Corpus: 421 files / 2,579 chunks / 3.38 M chars
(Obsidian vault, Hermes skills, AI-ECG research output, UCS case-report audit).
Query set: 28 queries frozen in `queries.json` before any run — 22 scored
semantics, 2 verbatim-title queries reported separately, 4 negative controls.
Harness: `bench_retrieval.py`; raw output: `results/`.

**Revision note (2026-10-07, later same day).** §2 and §3 as first written were
wrong on three points, all traced to one cause: Gemma was being run through a
CPU-only llama.cpp build with no multimodal projector. EmbeddingGemma-2 was then
re-run on a CUDA build (b11463) with its `mmproj`, and the numbers moved: the
throughput gap is ~10%, not 30×; images embed (top-1 5/12, MRR 0.613); and audio
embeds too — the earlier "no audio path exists" was an inference from the file
names in the GGUF repo, and it was wrong (the checkpoint's audio tower loads with
the projector; `llama-server-gemma-cuda.log` shows `init_audio` on startup).
What follows is the corrected document; superseded measurements are preserved in
§2.1, `results/throughput_cpu_gemma.json`, and the tone-control paragraph in §3.

## 1. Retrieval quality is a tie

| config | recall@10 (path) | MRR | hit@1 | recall@10 (chunk) |
|---|---|---|---|---|
| embeddinggemma-2 768d | 0.955 | 0.625 | 0.455 | 0.512 |
| embeddinggemma-2 256d | 0.909 | 0.634 | 0.500 | 0.483 |
| nomic-embed-text 768d | 0.909 | 0.643 | 0.455 | 0.480 |
| nomic-embed-text 256d | 0.909 | 0.683 | 0.545 | 0.460 |

Paired bootstrap on per-query reciprocal rank (gemma-768 − nomic-768):
**−0.018, 95% CI [−0.180, +0.137]**, P(Δ ≤ 0) = 0.58. The interval crosses zero
and is wide enough to hide a medium effect in either direction — at 22 queries
this benchmark can only rule out large differences, not rank the two models.
A filename-only matcher gets hit@10 = 0.21 / MRR = 0.10 on the same questions, so
the vector layer is doing real work; it is just not distinguishable between the
two embedders. **Unchanged by the CUDA re-run** — the quality pass does the same
arithmetic on the same BF16 weights, only faster.

## 2. Indexing throughput — 30.8 vs 34.0 doc/s, i.e. nomic is ~10% faster

Same machine, same 300-chunk sample, same batch, one server at a time, best of 3
(`results/throughput.json`, harness `bench/probe_throughput.py`):

| transport | docs/s | tok/s | projected time for the 2,579-chunk corpus |
|---|---|---|---|
| embeddinggemma-2 768d, llama.cpp b11463 CUDA `-ngl 99` on :11436 | 30.8 | 11,877 | ~1.4 min |
| nomic-embed-text 768d, Ollama on :11434 | 34.0 | 13,048 | ~1.3 min |

Both models are now on the GPU, so this is model-vs-model, not device-vs-device.
The gap is ~10% and lands on a pass that takes about a minute and a half either
way; query-time embedding is comparable (~32 ms vs ~24 ms) and brute-force scoring
is ~0.4 ms. **Indexing throughput no longer decides anything.**

### 2.1 The superseded measurement (kept deliberately)

The first pass reported gemma at ~2 doc/s vs nomic at ~70 doc/s ("~30×") and
concluded the embedder swap was 30× more expensive to index. Both figures are in
`results/` and they were not a model comparison. The 300-doc head-to-head
(`results/throughput_cpu_gemma.json`) is gemma **2.1** doc/s / 791 tok/s vs nomic
**66.5** doc/s / 25,502 tok/s — a 32× gap; the matched batch-16 point of the
sweeps is gemma **1.95** vs nomic **73.04** (`results/batch_sweep_*.json`). The
cause is visible in the same files: gemma was served by
`/e/Hermes/Data/tools/llama.cpp-b11460/`, a CPU-only build (`-ngl 0`; ships
`ggml-cpu-*.dll` and no CUDA/CuBLAS DLLs; the binary contains no `cuBLAS` strings;
the log shows `llama threadpool init, n_threads = 8` and no CUDA init) at
`http://127.0.0.1:11435`, while nomic ran on the RTX 4060 through Ollama. The
batch signature is diagnostic: gemma crawls 1.72 → 2.16 doc/s across batch 1→32 (8
threads saturated; the sweep's own JSON lists `:11435` as the endpoint) while
nomic scales 27.2 → 74.4 doc/s (GPU). Corrected run:
`results/throughput_run_cuda.log`.

## 3. Gemma reads images *and audio* — text, images and speech land in one 768-d space

The reason Gemma was chosen was "text+image+audio share one vector space". That
requires a multimodal projection, not just the text tower — and the original test
(§3.1) was run against a CPU build **without** `--mmproj`. With the CUDA build
(b11463) and the projector loaded, images embed. Launch from
`/e/Hermes/Data/tools/llama.cpp-b11463-cuda/` (that dir holds `cublas64_12.dll`,
`cublasLt64_12.dll`, `cudart64_12.dll`, `ggml-cuda.dll` — without them the server
silently falls back to CPU, which is what poisoned §2.1):

```
./llama-server.exe -m <embeddinggemma-2-BF16.gguf blob> \
  --mmproj E:/Hermes/Data/models/embeddinggemma-2/mmproj-embeddinggemma-2-BF16.gguf \
  --embeddings --pooling mean -ngl 99 -c 2048 -b 2048 -ub 2048 \
  --host 127.0.0.1 --port 11436
```

Measured request contract, all against the live server:

| request shape | result |
|---|---|
| `{"input": {"content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,…"}}]}}` | **200, one 768-d unit-norm vector** |
| `{"input": [{"type":"image_url",…}]}` (flat array) | 400 — `"prompt" elements must be a string, a list of tokens, …` |
| `{"input": "data:image/png;base64,…"}` (bare data URI) | 500 — `input (940180 tokens) is too large to process`; the URI is tokenized as text and, when it does fit, silently yields a garbage vector |
| `{"input": {"content":[img, img]}}` or `[img, text]` | 200 but **one** vector — parts are concatenated into one prompt, so it is one image per request |
| `{"input": ["text a", "text b"]}` | 200, two vectors — text still batches normally |

Evidence the vision tower is really in the path (not a silently dropped input):
the same image twice gives cosine **1.000000**; two different cards give
**0.7528**; a matching text query beats a mismatched one (0.5776 vs 0.5438); every
vector comes back L2-normalized, so cosine = dot product.

Cross-modal retrieval — 12 Sketchy cardiology cards (hand-drawn diagrams with drug
names as stylized text), text query → image, `bench/probe_image_retrieval.py`
(`results/image_retrieval.json`; cached image vectors in `results/image_index.json`):

* **top-1 = 5/12 (42%), MRR = 0.613** against a 1/12 = 8% chance floor.
* Indexing cost **3.35 img/s** (3.6 s for 12 images) — one request per image.
* Errors are class-level confusions (thiazide → K⁺-sparing, Class II/III → Class I),
  consistent with EmbeddingGemma-2's published multimodal numbers
  (`google/embeddinggemma-2`: MMEB-v2 image Hit@1 57.28; 20-task image/audio mean
  64.64) on material *harder* than the benchmark's photos.

Caveats, plainly: one 12-item set, cartoon diagrams rather than photos; the model
card says text prefixes apply to text only, so images are passed **without** a
`task:` prefix (which is what the probe does).

**Audio works too** — and getting that right needed a control, because the first
probe said the opposite. Synthetic 440 Hz/880 Hz tones come back *nearly
identical* (cos 0.9520), and a tone versus its own **byte-reversed** self scores
0.9013 — i.e. non-speech input is out of distribution and collapses into one
region, which looks exactly like a dead tower. Real speech separates cleanly, so
`bench/probe_audio_retrieval.py` synthesizes three SAPI clips with the ground-truth
text in-code and measures both directions (`results/audio_retrieval.json`):

| test | result |
|---|---|
| audio ↔ audio | cardio↔cardio2 **0.7704** vs cardio↔cooking **0.6102**, cardio2↔cooking **0.5664** |
| text → audio, cardiac query | **cardio 0.855** \| cardio2 0.755 \| cooking 0.569 — top-1 OK |
| text → audio, cooking query (with `task:` prefix) | **cooking 0.828** \| cardio 0.492 \| cardio2 0.488 — top-1 OK, widest margin |
| audio(cardio) → text | "atrial fibrillation…" **0.7624** vs "boiling pasta…" 0.4907 |

**top-1 = 2/2 text→audio, 2/2 audio→text, no cross-topic confusion.** Contract:
`{"input":{"content":[{"type":"input_audio","input_audio":{"data":<b64>,"format":"wav"}}]}}`
→ 200, one 768-d unit-norm vector; `{"type":"audio_url",…}` → 400
`unsupported content[].type`. Same one-part-per-request rule as images. Three clips
is a smoke test, not a benchmark — but the direction and the margin are unambiguous,
and the wrong first reading is preserved here rather than deleted.

Two practical cautions from the cross-modal probes: precision depends on *how the
query is phrased*, not just the corpus (hand-written class descriptions scored
2/7 on the same 12 cards where the card-title queries scored 5/12 — see
`results/crossmodal_gemma_cuda.json`), and there is no fixed cosine cutoff
separating positives (0.601–0.773) from negatives (0.547–0.702) — rank, don't
threshold.

### 3.1 The superseded measurement

The first pass concluded "neither candidate reads pixels". Two claims backed it,
and both were artefacts of the CPU build:

* llama.cpp `/v1/embeddings` returned HTTP 500 for both a data URI and an OpenAI
  `content: [{"type": "image_url"}]` part (`results/image_capability.json`) —
  true *without* `--mmproj`; that build had no vision tower to route the part to.
* Ollama returns a 768-d vector for a base64 data URI, but it is the text tower
  embedding base64 characters: the real image URI and a byte-reversed copy of the
  same file score 0.944 against each other (`probe_image_bytes.py`). **This one
  still stands** — Ollama's path never looks at the picture.
* Ollama 0.40 refuses the checkpoint outright: `unknown model architecture:
  'gemma-embedding2'` (reproduced by building a local model from the same GGUF
  blob). **Also still stands** — Gemma does need llama.cpp alongside Ollama, which
  is why §6 prices the second server rather than calling it free.

## 4. A hybrid is not justified by this data

Errors are largely uncorrelated, and each model wins questions the other misses —
gemma-768 puts "time for research during fellowship" at rank 1 where nomic-768
ranks it 10; nomic-768 puts "process that keeps dying on Windows" at rank 1 where
gemma-768 ranks it 7. Two models would therefore beat either alone on this set.
But the whole set is 22 questions and both models are already in the same
overlapping band, so "run two embedders, two indexes, and a fusion step" is a
large permanent complexity for an effect this benchmark cannot size. Revisit only
if a 100+ query set still shows complementary wins. (New asymmetry to note: a
hybrid is no longer the *only* route to images — Gemma alone now covers text+image.)

## 5. What actually limits retrieval quality here (not the embedder)

* **Cardiology is the weakest category for every config** (MRR 0.34–0.51 vs
  0.65–0.83 elsewhere) — and it is the project's main use case. The clinical
  material is not in the index: PDF extraction is unavailable in the agent
  sandbox (`pdftotext` is not on PATH and the venv has no PDF library), and
  Windows OCR fails under the sandbox, so `EKGs.pdf`, scanned PDFs and all
  images were never indexed. The indexer's PDF path must not depend on a
  `pdftotext` binary that only sometimes exists — the `pdf` skill already ships
  `pymupdf`-based extraction that would cover it.
* **Per-query detail** (`results/report.md`): the escalate-only safety question
  ranks ~10–12 everywhere; the two multi-target protocol questions land at 3–9.
* **256d truncation costs chunk granularity**: recall@10 at chunk level falls
  from 0.51 / 0.48 (768d) to 0.48 / 0.46 (256d) while document-level recall stays
  similar — the right file is still found, the right paragraph inside it less
  often. 256d also buys nothing at this scale (0.4 ms per query either way), so
  if storage is not the binding constraint, store 768d.
* **A raw cosine threshold cannot drive the web-search fallback.** Negative
  controls are answered with top-1 scores of 0.55–0.77, fully inside the positive
  band (nomic-768: positives 0.601–0.773, negatives 0.547–0.702 — overlap +0.100).
  Every config conflates the two bands, so "no local hits → go to the web" needs a
  different rule: a top-1 vs top-5 margin, a lexical confirmation of the dense
  hit, or the answer LLM judging whether the retrieved chunks actually address
  the question.
* **The Japanese query** scored 0.55–0.76 against English documents — cross-
  lingual queries retrieve confidently and wrongly, which is a second reason a
  score threshold alone is unsafe.

## 6. Decision

1. **Text search: keep nomic-embed-text at 768d as the default embedder.** Quality
   is a statistical tie, it needs no second server, and it is ~10% faster on the
   index pass. Gemma is no longer *expensive*, but it is still *extra*.
2. **Image *and audio* search: Gemma is now the only candidate that works, and it
   is cheap.** If image/scan or speech retrieval is wanted, run the b11463 CUDA
   server on :11436 (EmbeddingGemma-2 + `mmproj`) alongside Ollama and point those
   paths at it. Budget: ~3 GB VRAM resident, 3.35 img/s indexing, 768-d vectors
   shared with Gemma's text *and* speech vectors in one space — which is exactly
   the original rationale, now delivered for both modalities (§3). This replaces
   the earlier "drop the image/audio argument" item.
3. **Do not build a text hybrid now.** Re-test after the query set grows past 100
   labelled queries. (A text+image pipeline is a separate, justified thing, and
   §2 no longer prices it out.)
4. **Wire the media paths through the documented payload contract** (`input.content`
   array; never a bare data URI; one part per request; `input_audio` for speech,
   `image_url` for images) — `bench/probe_image_retrieval.py` and
   `bench/probe_audio_retrieval.py` are the reference implementations.
5. **Fix the corpus, not the model.** Landing PDF/OCR extraction will move the
   cardiology numbers far more than any embedder swap — and images entering the
   index is now unblocked (see §3).
6. **Design the fallback on evidence, not on a cosine cutoff.**

## How to reproduce

```
PY="E:/Python/python"                # NOT the bare `python` — the Hermes tool
                                     # pythons ship without numpy
# text/tie-break benchmark (unchanged)
$PY bench_retrieval.py build         # freeze the corpus  -> corpus.json
python bench_retrieval.py embed      # one pass per model -> embeds-*-native.npz
python bench_retrieval.py run        # metrics            -> results/
python validate_queries.py           # label + leakage audit
python probe_modality.py             # image/scan questions against a text index

# throughput — a fair run needs BOTH models on the GPU
#   start first: llama.cpp b11463 CUDA + embeddinggemma-2-BF16 + mmproj on 127.0.0.1:11436
python probe_throughput.py           # indexing throughput, both transports
python probe_batch.py gemma|nomic    # batch-size sweep

# images
$PY probe_image_capability.py        # does any transport read an image at all
$PY probe_image_retrieval.py         # text -> image ranking -> results/image_retrieval.json

# audio (speech in, speech out; ground-truth text is synthesized in-code; stdlib only)
$PY probe_audio_retrieval.py         # -> results/audio_retrieval.json
```

Env overrides: `GEMMA_URL` retargets the gemma configs in `bench_retrieval.py`
(default `http://127.0.0.1:11436/v1/embeddings`); `LOCALSEARCH_URL` does the same
for `localsrch.py`, which already speaks both transports (`LOCALSEARCH_BACKEND=openai`
→ llama.cpp, `ollama` → `/api/embed`).

Interpreter note: the bench scripts import `numpy`, which the Hermes tool pythons
do not carry — run them with `E:/Python/python` (3.12, numpy 2.5.3), i.e. `PY` as
set above. Symptom of getting this wrong: `ModuleNotFoundError: No module named
'numpy'` on line 1, and any `| tee results/…log` in the same command short-circuits
unless the `&&`-chain has already failed. `probe_audio_retrieval.py` is
stdlib-only and runs under either interpreter.

`results/metrics.json` holds every per-query rank and top-10 list; nothing in
this document is a hand transcription of a terminal.

## 9. Multimodal test index over `D:\search test` (2026-10-07)

First run of `localsrch.py` against a real folder of mixed media, not a curated
probe set. Source: `D:\search test` — 46 photos, 11 videos, 5 PDFs (two of them
medical atlases); **no standalone audio files, the audio lives inside the videos**.

```
ltest.cmd index "D:/search test"      # LOCALSEARCH_DB=index\test-search.db
ltest.cmd query "amyloidosis nephrotic syndrome apple green birefringence" -k 8
ltest.cmd stats
```

Result: `scanned=62 indexed=61 chunks=732 media=56f/564c unchanged=0 no_text=1
errors=0 213.9s`, split **video_audio=262, video_frame=256, text=168, image=46**.
The one `no_text` file is `.temp-20250702_154501090_BACK_SEAMLESS_ZOOM.mp4`, a
`.temp-` failname from some downloader, not a media file — correctly skipped.
All 46 images produced a chunk and 10 of 11 videos produced frame + audio rows.

Note `frames=0 audio=1` on `20250211_081230.mp4`: it is a 9.8 s HEVC clip, and the
sampler takes one frame every 45 s, so a 10-second video legitimately yields no
keyframe. Short clips need a shorter `interval` to be visible to the image tower.

Text queries rank against the media correctly, with timestamps, at ~400 ms:

| query | top-1 hit |
|---|---|
| `amyloidosis nephrotic syndrome apple green birefringence` | `1.5 Amyloidosis.mp4` audio **0:01:30** (0.752), frames 0:01:30 / 0:00:45 in the next two slots |
| `streptococcus pyogenes group A beta hemolytic bacitracin` | top 3 all `B1.3 - Strep pyogenes.mp4` audio (0.756 / 0.724 / 0.715) |
| `staphylococcus aureus gram positive cocci in clusters` | `B1.1 - Staph Aureus.mp4` 0:00:00 (0.722) |
| `reperfusion injury free radical damage to cell membrane` | `1.4 Free Radical Injury.mp4` frames 0:21:00 / 0:21:45 / 0:20:15 |

Text→image is real but only when the query describes the photo. Taking one image,
`20250206_152709.jpg`, and describing what is actually in it ("a man leaning over
a counter holding an OPD patient clinical record folder in a hospital corridor")
puts that image **rank 1 of 46 at cos 0.795** against a next-best 0.635; a blander
phrasing ("hospital OPD reception desk with patient record file") still ranks it
1st at 0.696 but with only 0.035 over the runner-up. Vague queries do not separate
images — the 46 images sit in a narrow band (pairwise cos min 0.557, mean 0.681)
and generic prompts like "a laboratory scene" match everything at 0.58–0.71.

Two silent-corruption traps found here, both now fixed in `localsrch.py`:

1. **A broken Pillow shadowing a working one.** An inherited `PYTHONPATH` pointed
   at a venv whose Pillow is missing `_imaging`; the image path raised `ImportError`,
   the ingest swallowed it, and every photo *and every video keyframe* indexed as
   nothing while the run still exited 0. Both launchers now `set "PYTHONPATH="`, and
   `_image_b64` returns `None` with `_ffprobe_dims` as an ffmpeg fallback, so one bad
   decoder can no longer drop media silently. Diagnose with
   `env -u PYTHONPATH E:/Python/python -c "import PIL;from PIL import Image;print(PIL.__file__)"`.
2. **`pdftotext` is not on `PATH`.** It ships inside the git-bash tree
   (`<git>/mingw64/bin/`), so `shutil.which` misses it and PDF text arrives as
   `�` mojibake. Resolving it by absolute path fixed it: the atlas PDFs now index
   with **0** U+FFFD characters.

Corollary rule: never trust the exit code of a media index. Count rows per modality
and assert each is non-zero — `SELECT modality, COUNT(*) FROM chunks GROUP BY 1`.
