# GOBBLE

Local multimodal search over your own files. One sqlite index, one vector space, for
**text, PDFs, images, audio and video** — nothing leaves the machine, no account, no API key.

Ask in plain words ("cellular injury apoptosis", "rural ECG deployment screening") and get
back the file, the **approximate timestamp** for a video/audio, or the **page number** for a PDF.

```
GOBBLE — local search
+---------------------------------------------------------------+
| [ cellular injury ]                          [ Search ]  Light |
| types: (any) text pdf image video audio      path ~ ______     |
+---------------------------------------------------------------+
| VIDEO AUDIO   ~0:33:00   1.2 Example Clip.mp4            0.79    |
| PDF           PAGE 236   Example Textbook (2021).pdf    0.82    |
+---------------------------------------------------------------+
```

## Quick start

1. Serve an embedding model (llama-server started **with `--mmproj`** — the multimodal
   projector is what lets one model embed text, images and audio into the same space):

   ```
   llama-server -m embeddinggemma-2-bf16.gguf --mmproj mmproj-embeddinggemma-2-bf16.gguf ^
                --embeddings --host 127.0.0.1 --port 8080
   ```

2. Run the web UI:

   ```
   gobble-serve.cmd        :: real index, http://127.0.0.1:8765
   gobble-test.cmd         :: throw-away demo index, http://127.0.0.1:8766
   ```

   Both clear `PYTHONPATH` first (a foreign site-packages can shadow a working Pillow and
   silently drop every image) and pick a `python`/`py` that can `import numpy`; set `PY` to
   your interpreter first if you want to force one. Note: `py` on Windows honours a script's
   `#!/usr/bin/env python` shebang and would then re-resolve `python` from `PATH`, so
   `gobble.py` ships without a shebang and the launchers invoke the interpreter by full path.

3. Or use the CLI:

   ```
   python gobble.py index                       # every root listed in roots.txt
   python gobble.py index "E:/some/dir"         # one folder … or one single file
   #   (edit roots.txt first — it ships with example paths, not real ones)
   python gobble.py query "ecg rhythm strip findings" -k 8
   python gobble.py stats
   python gobble.py serve
   ```

## The page (`gobble.html`)

One static HTML file, served by the tiny `gobble_web.py` JSON API.

* **Light and dark themes**, toggled in the header, remembered in `localStorage`; light is
  the default when the system has no preference. Every colour is a CSS custom property, so
  the whole page re-paints from one `<html data-theme>` attribute.
* **Dynamic results** — cards are built per search; matches are highlighted in the snippet.
* **Approximate timestamps** for video/audio hits (`~0:33:00` — the nearest sampled keyframe
  or audio segment) and **page numbers** for PDF hits (`PAGE 236`, or `PAGE 4–5` when a chunk
  spans a page break). Both are keyword-proximate, not exact.
* **Filters** — type chips (text / pdf / image / video / audio), a free-text `path ~` filter,
  and a root filter when your index holds more than one root.
* **Add to index** — paste a folder or a single file path and hit *Add & index*; the run
  streams its own log (`scanned / indexed / chunks / unchanged / errors`).
* **Preview** — video and audio play inline, images open full size, PDFs open at the matched
  page (`…pdf#page=236`), text opens scrolled to the match.

## PDFs

PDFs are indexed **page by page** (`pdftotext -layout`, split on form feeds), so a hit carries
the page it came from and the viewer jumps straight there. If `pdftotext` is missing the PDF
is still indexed as plain text, just without page attribution.

## Requirements

* Windows, Python 3.12 with `numpy`
* `pdftotext` (poppler) for PDF page numbers, `ffmpeg` for video/audio
* A llama-server exposing an EmbeddingGemma-2 model with `--mmproj`

## Why it works

EmbeddingGemma-2 shares one 768-d space across text, image and audio, so a text query can
return a photo, a keyframe or a lecture timestamp from the same index. The model has no video
tower, so a video is indexed as sampled keyframes (image vectors) **plus** audio segments
(audio vectors), each carrying its own timestamp — which is why timestamps are approximate.

No vector database: vectors are float32 BLOBs in sqlite, scored with a single numpy matrix
multiply (exact cosine, fine at tens of thousands of chunks).

`bench/` holds the retrieval and throughput probes and their frozen results.

## Your files stay yours

`roots.txt` is a template: it ships with example paths, not the author's. Nothing in this
repo points at a real folder, and the committed `bench/` results have their corpus paths
replaced by placeholders (`<vault>/doc-014.md`, `<library>/doc-001.pdf`). The index itself
(`index/*.db`) and model weights (`*.gguf`) are git-ignored, so a clone contains code only.

## Licence

MIT — see `LICENSE`.
