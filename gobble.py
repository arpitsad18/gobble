"""GOBBLE - local multimodal search over your own files.

One sqlite index, one vector space, for text, PDFs, images, audio and video.
EmbeddingGemma-2 runs in a local llama-server (CUDA + --mmproj), so nothing
leaves the machine and no account or API key is involved.

    python gobble.py index                  # index every root in roots.txt
    python gobble.py index "E:/some/dir"    # index one folder
    python gobble.py query "ecg rhythm strip findings" [-k 8]
    python gobble.py stats
    python gobble.py serve                  # web UI on http://127.0.0.1:8765

Text, images, audio and video all live in ONE index and ONE vector space:
EmbeddingGemma-2 shares its 768-d space across text / image / audio, so a text
query can return a photo, a keyframe or a lecture timestamp. It has no video
tower, so a video is indexed as sampled keyframes (image vectors) plus audio
segments (audio vectors) - each carrying its timestamp.

  - text  -> pdftotext / WinRT OCR / raw read
  - image -> embedded visually (+ optional --ocr text chunks)
  - audio -> 16 kHz mono WAV segments -> input_audio
  - video -> ffmpeg keyframes every --frame-interval + audio every --audio-segment

Requires BACKEND=openai against a llama-server started WITH --mmproj; the
Ollama backend cannot embed anything but text.

No vector database: vectors are stored as float32 BLOBs in sqlite and scored
with a single numpy matrix multiply (exact cosine, fine at thousands of chunks).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# Two backends: 'ollama' (/api/embed) and 'openai' (llama-server /v1/embeddings).
BACKEND = os.environ.get("LOCALSEARCH_BACKEND", "ollama")
OPENAI_URL = os.environ.get("LOCALSEARCH_URL", "http://127.0.0.1:11436/v1/embeddings")
OLLAMA = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL = os.environ.get("LOCALSEARCH_MODEL", "hf.co/ggml-org/embeddinggemma-2-GGUF:BF16")
DIM = int(os.environ.get("LOCALSEARCH_DIM", "256"))  # MRL truncation of 768d

# Prompt format switch — vectors from different prompt modes are NOT comparable,
# so the same mode must be used for indexing and querying (re-index when changing).
PROMPT_MODE = os.environ.get("LOCALSEARCH_PROMPTS", "gemma")

# Vectors from different models / dims / prompt formats are NOT comparable, so
# each configuration gets its own index file.
_slug = re.sub(r"[^a-z0-9]+", "-", f"{MODEL}-{BACKEND}-{DIM}-{PROMPT_MODE}".lower()).strip("-")
_db_env = os.environ.get("LOCALSEARCH_DB")
DB_PATH = Path(_db_env) if _db_env else HERE / "index" / f"search-{_slug[-60:]}.db"
if not DB_PATH.is_absolute():
    DB_PATH = HERE / DB_PATH
ROOTS_FILE = HERE / "roots.txt"
CHUNK_CHARS = 1800
CHUNK_OVERLAP = 240
BATCH = 16

TEXT_EXT = {
    ".md", ".txt", ".markdown", ".rst", ".log", ".csv", ".tsv", ".json", ".yaml",
    ".yml", ".toml", ".ini", ".cfg", ".py", ".js", ".ts", ".html", ".css", ".sql",
    ".sh", ".bat", ".ps1", ".tex", ".bib",
}
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", ".cache", "site-packages",
    "$RECYCLE.BIN", "System Volume Information", ".obsidian", ".trash", ".stfolder",
    "dist", "build", ".mypy_cache", ".pytest_cache",
}
SKIP_EXT = {
    ".exe", ".dll", ".so", ".dylib", ".pyc", ".zip", ".7z", ".rar", ".gz", ".iso",
    ".ttf", ".otf", ".woff", ".woff2",
    ".db", ".sqlite", ".npy", ".npz", ".safetensors", ".gguf", ".bin", ".pak", ".asar",
}

# ---------------------------------------------------------------- media config
IMAGE_MEDIA_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".heic"}
AUDIO_MEDIA_EXT = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".opus", ".aac", ".wma", ".aiff", ".aif"}
VIDEO_MEDIA_EXT = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".wmv", ".m4v", ".mpg", ".mpeg",
                   ".3gp", ".flv"}
MEDIA_EXT = IMAGE_MEDIA_EXT | AUDIO_MEDIA_EXT | VIDEO_MEDIA_EXT

TMP_ROOT = Path(os.environ.get("LOCALSEARCH_TMP", str(HERE / "tmp")))
MEDIA_TOOLS = Path(os.environ.get("LOCALSEARCH_TOOLS", "E:/Hermes/Data/tools"))


def modality_of(path: Path) -> str:
    """'image' | 'audio' | 'video' | 'text' - decides how a file gets embedded."""
    s = path.suffix.lower()
    if s in VIDEO_MEDIA_EXT:
        return "video"
    if s in AUDIO_MEDIA_EXT:
        return "audio"
    if s in IMAGE_MEDIA_EXT:
        return "image"
    return "text"


def _ffmpeg() -> str | None:
    exe = os.environ.get("LOCALSEARCH_FFMPEG")
    if exe and Path(exe).exists():
        return exe
    from shutil import which
    found = which("ffmpeg") or which("ffmpeg.exe")
    if found:
        return found
    for cand in sorted(MEDIA_TOOLS.glob("ffmpeg*/bin/ffmpeg.exe")):
        return str(cand)
    return None


# ---------------------------------------------------------------- embedding

def _post(url: str, payload: dict, timeout: int = 300) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def embed(texts: list[str], attempt: int = 0) -> np.ndarray:
    """Embed a batch of already-prefixed strings -> (n, DIM) float32, L2-normalised."""
    if BACKEND == "openai":
        out = _post(OPENAI_URL, {"model": "local", "input": texts}, timeout=600)
        vecs = [d["embedding"] for d in out["data"]]
    else:
        try:
            out = _post(f"{OLLAMA}/api/embed", {"model": MODEL, "input": texts})
            vecs = out["embeddings"]
        except urllib.error.HTTPError as e:
            if e.code == 404 and attempt == 0:  # older daemon: one-at-a-time endpoint
                vecs = [_post(f"{OLLAMA}/api/embeddings",
                              {"model": MODEL, "prompt": t})["embedding"] for t in texts]
            else:
                raise
    a = np.asarray(vecs, dtype=np.float32)[:, :DIM]
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return a / norms


def embed_part(part: dict) -> np.ndarray:
    """Embed ONE non-text part (image or audio) -> (DIM,) float32 unit vector.

    Payload shape is load-bearing: llama.cpp wants
    {"input": {"content": [ {"type":"image_url", ...} ]}} - a bare list of parts
    is rejected with HTTP 400, and an image must be a data: URI inside image_url
    (a raw data URI as `input` blows the token budget: "input (940180 tokens) is
    too large"). Audio only accepts {"type":"input_audio"}; "audio_url" -> 400
    "unsupported content[].type".
    """
    if BACKEND != "openai":
        raise RuntimeError("images/audio need LOCALSEARCH_BACKEND=openai against a "
                           "llama-server started with --mmproj (Ollama embeds text only)")
    out = _post(OPENAI_URL, {"model": "local", "input": {"content": [part]}}, timeout=900)
    v = np.asarray(out["data"][0]["embedding"], dtype=np.float32)[:DIM]
    n = float(np.linalg.norm(v))
    return v / (n or 1.0)


# Prompt format switch — vectors from different prompt modes are NOT comparable,
# so the same mode must be used for indexing and querying (re-index when changing).
PROMPT_MODE = os.environ.get("LOCALSEARCH_PROMPTS", "gemma")


def as_document(title: str, text: str) -> str:
    if PROMPT_MODE == "gemma":
        return f"title: {title or 'none'} | text: {text}"
    if PROMPT_MODE == "nomic":
        return f"search_document: {text}"
    return text


def as_query(text: str) -> str:
    if PROMPT_MODE == "gemma":
        return f"task: search result | query: {text}"
    if PROMPT_MODE == "nomic":
        return f"search_query: {text}"
    return text


# ---------------------------------------------------------------- extraction

def _which_pdftotext() -> str | None:
    """Locate pdftotext, which on Windows is usually NOT on PATH.

    poppler ships inside the git-bash tree (<git>/mingw64/bin), so a bare
    shutil.which() misses it: PDFs then index as U+FFFD mojibake or not at all.
    Check the env override, then PATH, then the known tool trees.
    """
    exe = os.environ.get("LOCALSEARCH_PDFTOTEXT")
    if exe and Path(exe).exists():
        return exe
    from shutil import which
    found = which("pdftotext") or which("pdftotext.exe")
    if found:
        return found
    cands = [Path("C:/Program Files/Git/mingw64/bin/pdftotext.exe"),
             Path("C:/Program Files (x86)/Git/mingw64/bin/pdftotext.exe")]
    cands += sorted(MEDIA_TOOLS.glob("git*/mingw64/bin/pdftotext.exe"))
    cands += sorted(MEDIA_TOOLS.glob("poppler*/**/pdftotext.exe"))
    cands += sorted(MEDIA_TOOLS.glob("**/bin/pdftotext.exe"))
    for c in cands:
        if c.exists():
            return str(c)
    return None


PDFTOTEXT = _which_pdftotext()


OOXML = {".docx", ".xlsx", ".pptx"}
_OOXML_PARTS = {
    ".docx": ("word/document.xml",),
    ".xlsx": ("xl/sharedStrings.xml",),
    ".pptx": ("ppt/slides/",),
}
_TAG = re.compile(r"<[^>]+>")


def _ooxml_text(path: Path) -> str | None:
    """Text from .docx/.xlsx/.pptx using only the stdlib (zip + XML strip)."""
    import html
    import zipfile
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            targets = [n for n in names if n in _OOXML_PARTS[path.suffix.lower()]]
            if path.suffix.lower() == ".pptx":
                targets = sorted(n for n in names if n.startswith("ppt/slides/slide"))
            out = []
            for n in targets:
                xml = z.read(n).decode("utf-8", "replace")
                xml = xml.replace("</w:p>", "\n").replace("</a:p>", "\n").replace("><", "> <")
                out.append(re.sub(r"\s+\n", "\n", _TAG.sub(" ", xml)))
            text = html.unescape("\n".join(out))
            return text if text.strip() else None
    except (zipfile.BadZipFile, OSError, KeyError):
        return None


IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
WIN_OCR = HERE / "win_ocr.ps1"


def _ocr_text(path: Path, max_pages: int = 20) -> str | None:
    """Extract text from images or scanned PDFs via native Windows OCR (WinRT)."""
    if not WIN_OCR.exists():
        return None
    try:
        cmd = [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", str(WIN_OCR),
            "-Path", str(path),
            "-MaxPages", str(max_pages),
        ]
        p = subprocess.run(cmd, capture_output=True, timeout=120)
        if p.returncode == 0:
            text = p.stdout.decode("utf-8", errors="replace").strip()
            return text if text else None
    except (OSError, subprocess.SubprocessError):
        return None
    return None


def extract_pages(path: Path) -> list[str] | None:
    """Per-page text of a PDF, or None if pdftotext cannot read it.

    pdftotext separates pages with a form feed (unless told to drop page
    breaks), so the page number of any character offset stays recoverable.
    That is what lets a hit say "page 412" instead of only naming the file.
    """
    if not PDFTOTEXT:
        return None
    try:
        p = subprocess.run([PDFTOTEXT, "-q", str(path), "-"],
                           capture_output=True, timeout=900)
    except (OSError, subprocess.SubprocessError):
        return None
    if p.returncode != 0:
        return None
    text = p.stdout.decode("utf-8", errors="replace")
    if not text.strip():
        return None
    pages = text.split("\f")
    while pages and not pages[-1].strip():
        pages.pop()
    return pages or None


def extract(path: Path) -> str | None:
    """Return plain text for a file, or None if it has no extractable text."""
    suffix = path.suffix.lower()
    try:
        if suffix in TEXT_EXT:
            return path.read_text(encoding="utf-8", errors="replace")
        if suffix in OOXML:
            return _ooxml_text(path)
        if suffix == ".pdf":
            pages = extract_pages(path)
            if pages:
                return "\n\n".join(pages)
            # Fallback for scanned PDFs (or when pdftotext is not available)
            return _ocr_text(path)
        if suffix in IMAGE_EXT:
            return _ocr_text(path)
    except (OSError, subprocess.SubprocessError):
        return None
    return None


def chunk_spans(text: str) -> list[tuple[str, int, int]]:
    """Split text into overlapping chunks; return (chunk_text, start, end)."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    if len(text) <= CHUNK_CHARS:
        return [(text, 0, len(text))]
    out, start = [], 0
    while start < len(text):
        end = start + CHUNK_CHARS
        if end < len(text):  # prefer a paragraph/sentence break near the edge
            window = text[start:end]
            for sep in ("\n\n", "\n", ". "):
                cut = window.rfind(sep)
                if cut > CHUNK_CHARS * 0.55:
                    end = start + cut + len(sep)
                    break
        out.append((text[start:end].strip(), start, end))
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return [(t, s, e) for t, s, e in out if t]


def chunk(text: str) -> list[str]:
    return [t for t, _s, _e in chunk_spans(text)]


def paged_text(pages: list[str]) -> tuple[str, list[int]]:
    """Join per-page text -> (normalized_text, char offset where each page starts)."""
    norm_parts = [re.sub(r"\n{3,}", "\n\n", p).strip() for p in pages]
    joined = re.sub(r"\n{3,}", "\n\n", "\n\n".join(norm_parts)).strip()
    starts, cursor = [], 0
    for part in norm_parts:
        if not part:
            starts.append(min(cursor, len(joined)))
            continue
        idx = joined.find(part, cursor)
        if idx < 0:
            idx = cursor
        starts.append(idx)
        cursor = idx + len(part)
    return joined, starts


def page_of(offset: int, starts: list[int]) -> int:
    """1-based page number holding a character offset."""
    lo, hi = 0, len(starts)
    off = max(0, offset)
    while lo < hi:
        mid = (lo + hi) // 2
        if starts[mid] <= off:
            lo = mid + 1
        else:
            hi = mid
    return max(1, lo)


# ---------------------------------------------------------------- media

def _hms(seconds: float) -> str:
    s = int(max(0.0, seconds))
    return f"{s // 3600:d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _ffprobe_dims(path: Path) -> tuple[int, int]:
    ff = _ffmpeg()
    probe = str(Path(ff).with_name("ffprobe.exe")) if ff else "ffprobe"
    try:
        out = subprocess.run([probe, "-v", "error", "-select_streams", "v:0",
                              "-show_entries", "stream=width,height",
                              "-of", "csv=p=0:s=x", str(path)],
                             capture_output=True, text=True, timeout=60).stdout.strip()
        w, h = out.splitlines()[0].split("x")[:2]
        return int(w), int(h)
    except Exception:  # noqa: BLE001 - cosmetic only
        return 0, 0


def _image_bytes(path: Path, max_side: int = 1024) -> tuple[bytes, int, int] | None:
    """(downscaled JPEG bytes, ORIGINAL width, ORIGINAL height) or None.

    Downscaling is not cosmetic: an un-resized phone photo costs ~2k image tokens
    and the server refuses it outright ("input (940180 tokens) is too large").
    Pillow is the fast path; ffmpeg is the fallback so a missing/broken Pillow,
    a HEIC or a camera RAW silently drops nothing. (Seen for real: a PYTHONPATH
    pointing at a venv whose Pillow lacked _imaging made EVERY image and every
    video keyframe vanish with no error.)
    """
    try:
        import io

        from PIL import Image
        with Image.open(path) as im:
            w, h = im.size
            rgb = im.convert("RGB")
            scale = max_side / max(w, h)
            if scale < 1.0:
                rgb = rgb.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                                 Image.LANCZOS)
            buf = io.BytesIO()
            rgb.save(buf, format="JPEG", quality=88)
        return buf.getvalue(), w, h
    except Exception:  # noqa: BLE001 - Pillow is best-effort
        pass

    ff = _ffmpeg()
    if not ff:
        return None
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="lsimg_", dir=str(TMP_ROOT)))
    try:
        dst = tmp / "img.jpg"
        _run_ffmpeg([ff, "-v", "error", "-nostdin", "-i", str(path),
                     "-vf", f"scale='min({max_side},iw)':-2", "-q:v", "4",
                     "-f", "image2", "-y", str(dst)])
        if not dst.exists():
            return None
        w, h = _ffprobe_dims(path)
        return dst.read_bytes(), w, h
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def embed_image(path: Path, max_side: int = 1024) -> np.ndarray | None:
    got = _image_bytes(path, max_side)
    if not got:
        return None
    raw, _w, _h = got
    b64 = base64.b64encode(raw).decode()
    return embed_part({"type": "image_url",
                       "image_url": {"url": "data:image/jpeg;base64," + b64}})


def embed_wav(path: Path) -> np.ndarray:
    """16 kHz mono PCM WAV -> audio vector."""
    b64 = base64.b64encode(path.read_bytes()).decode()
    return embed_part({"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}})


def _run_ffmpeg(cmd: list[str]) -> None:
    subprocess.run(cmd, capture_output=True, timeout=7200, check=False)


def extract_frames(src: Path, out_dir: Path, interval: float,
                   max_side: int) -> list[tuple[Path, float]]:
    """ONE ffmpeg pass -> keyframe JPEG every `interval` seconds: [(file, t0), ...]."""
    ff = _ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg not found (set LOCALSEARCH_FFMPEG or put it on PATH)")
    out_dir.mkdir(parents=True, exist_ok=True)
    vf = f"fps=1/{interval},scale='min({max_side},iw)':-2"
    _run_ffmpeg([ff, "-v", "error", "-nostdin", "-i", str(src), "-vf", vf,
                 "-q:v", "4", "-y", str(out_dir / "f_%06d.jpg")])
    return [(f, i * interval) for i, f in enumerate(sorted(out_dir.glob("f_*.jpg")))]


def extract_audio(src: Path, out_dir: Path, segment: float) -> list[tuple[Path, float]]:
    """ONE ffmpeg pass -> 16 kHz mono WAV segments: [(file, t0), ...]."""
    ff = _ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg not found (set LOCALSEARCH_FFMPEG or put it on PATH)")
    out_dir.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg([ff, "-v", "error", "-nostdin", "-i", str(src), "-vn", "-ac", "1",
                 "-ar", "16000", "-c:a", "pcm_s16le", "-f", "segment",
                 "-segment_time", str(segment), "-y", str(out_dir / "a_%05d.wav")])
    return [(f, i * segment) for i, f in enumerate(sorted(out_dir.glob("a_*.wav")))]


def _write_media(con: sqlite3.Connection, path: Path, root: Path, st,
                 rows: list[tuple[str, np.ndarray, dict]]) -> int:
    """One chunk per media vector: rows = [(label, vec, meta), ...]."""
    sha = hashlib.sha256(f"{path}:{st.st_size}:{int(st.st_mtime)}".encode()).hexdigest()
    with con:
        con.execute("DELETE FROM chunks WHERE file_id IN (SELECT id FROM files WHERE path=?)",
                    (str(path),))
        con.execute("""INSERT INTO files (path, root, mtime, size, sha, chars, n_chunks,
                                          indexed_at, kind)
                       VALUES (?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(path) DO UPDATE SET mtime=excluded.mtime, size=excluded.size,
                         sha=excluded.sha, chars=excluded.chars, n_chunks=excluded.n_chunks,
                         indexed_at=excluded.indexed_at, kind=excluded.kind""",
                    (str(path), str(root), st.st_mtime, st.st_size, sha, 0, len(rows),
                     time.time(), modality_of(path)))
        fid = con.execute("SELECT id FROM files WHERE path=?", (str(path),)).fetchone()[0]
        con.executemany("INSERT INTO chunks (file_id, ordinal, text, vec, modality, meta) "
                        "VALUES (?,?,?,?,?,?)",
                        [(fid, i, label, v.tobytes(), meta.get("kind", "media"), json.dumps(meta))
                         for i, (label, v, meta) in enumerate(rows)])
    return len(rows)


def index_media(path: Path, root: Path, con: sqlite3.Connection, st, stats: dict,
                opts: dict) -> None:
    """Embed an image / audio file / video into the SAME index as the text.

    A video has no embedder of its own: it becomes keyframes (image tower) plus
    audio segments (audio tower), each row tagged with its timestamp so a hit
    reads "video audio 0:12:30 of lecture.mp4".
    """
    kind = modality_of(path)
    max_side = int(opts.get("max_side", 1024))
    interval = float(opts.get("interval", 45.0))
    segment = float(opts.get("segment", 45.0))
    rows: list[tuple[str, np.ndarray, dict]] = []
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="lsmedia_", dir=str(TMP_ROOT)))
    try:
        if kind == "image":
            v = embed_image(path, max_side)
            if v is not None:
                rows.append((f"image: {path.name}", v,
                             {"kind": "image", "t": None, "src": str(path)}))
            if opts.get("ocr"):  # photos of slides are text-searchable too
                text = _ocr_text(path)
                if text and text.strip():
                    pieces = chunk(text)
                    pref = [as_document(path.stem, c) for c in pieces]
                    vecs = np.vstack([embed(pref[i:i + BATCH])
                                      for i in range(0, len(pref), BATCH)])
                    for c, vv in zip(pieces, vecs):
                        rows.append((c, vv, {"kind": "text", "t": None, "src": str(path)}))
        elif kind == "audio":
            for f, t0 in extract_audio(path, tmp / "audio", segment):
                rows.append((f"audio {_hms(t0)} of {path.name}", embed_wav(f),
                             {"kind": "audio", "t": t0, "src": str(path)}))
        else:
            if opts.get("frames", True):
                for f, t0 in extract_frames(path, tmp / "frames", interval, max_side):
                    v = embed_image(f, max_side)
                    if v is not None:
                        rows.append((f"video frame {_hms(t0)} of {path.name}", v,
                                     {"kind": "video_frame", "t": t0, "src": str(path)}))
            if opts.get("audio", True):
                for f, t0 in extract_audio(path, tmp / "audio", segment):
                    rows.append((f"video audio {_hms(t0)} of {path.name}", embed_wav(f),
                                 {"kind": "video_audio", "t": t0, "src": str(path)}))
        if not rows:
            stats["no_text"] += 1
            return
        n = _write_media(con, path, root, st, rows)
        stats["indexed"] += 1
        stats["chunks"] += n
        stats["media_files"] += 1
        stats["media_chunks"] += n
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- store

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
  id INTEGER PRIMARY KEY, path TEXT UNIQUE, root TEXT, mtime REAL, size INTEGER,
  sha TEXT, chars INTEGER, n_chunks INTEGER, indexed_at REAL,
  kind TEXT DEFAULT 'text'
);
CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY, file_id INTEGER, ordinal INTEGER, text TEXT, vec BLOB,
  modality TEXT DEFAULT 'text', meta TEXT
);
CREATE INDEX IF NOT EXISTS chunks_file ON chunks(file_id);
"""


def db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.executescript(SCHEMA)
    # migrate indexes built before multi-modal support (existing rows are text)
    ccols = {r[1] for r in con.execute("PRAGMA table_info(chunks)")}
    if "modality" not in ccols:
        con.execute("ALTER TABLE chunks ADD COLUMN modality TEXT DEFAULT 'text'")
    if "meta" not in ccols:
        con.execute("ALTER TABLE chunks ADD COLUMN meta TEXT")
    fcols = {r[1] for r in con.execute("PRAGMA table_info(files)")}
    if "kind" not in fcols:
        con.execute("ALTER TABLE files ADD COLUMN kind TEXT DEFAULT 'text'")
    con.commit()
    return con


def walk(root: Path):
    # A single file is a valid target too - the web UI lets you paste one path.
    if root.is_file():
        if root.suffix.lower() not in SKIP_EXT and not root.name.startswith("~$"):
            yield root
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix.lower() in SKIP_EXT or name.startswith("~$"):
                continue
            yield p


def index(root: Path, con: sqlite3.Connection, force: bool = False,
          opts: dict | None = None) -> dict:
    opts = opts or {}
    stats = {"root": str(root), "scanned": 0, "skipped_unchanged": 0,
             "no_text": 0, "indexed": 0, "chunks": 0, "errors": 0,
             "media_files": 0, "media_chunks": 0}
    t0 = time.time()
    for p in walk(root):
        stats["scanned"] += 1
        try:
            st = p.stat()
        except OSError:
            stats["errors"] += 1
            continue
        row = con.execute("SELECT id, mtime, size FROM files WHERE path=?", (str(p),)).fetchone()
        if row and not force and abs(row[1] - st.st_mtime) < 1 and row[2] == st.st_size:
            stats["skipped_unchanged"] += 1
            continue
        kind = modality_of(p) if opts.get("media", True) else "text"
        if kind != "text":
            try:
                index_media(p, root, con, st, stats, opts)
            except Exception as e:  # noqa: BLE001 - prototype: record and continue
                print(f"  ! media failed {p}: {e}", file=sys.stderr)
                stats["errors"] += 1
            continue
        pages = extract_pages(p) if p.suffix.lower() == ".pdf" else None
        if pages:
            text, page_starts = paged_text(pages)
            spans = chunk_spans(text)
            pieces = [t for t, _s, _e in spans]
            metas = [{"kind": "pdf", "src": str(p),
                      "page": page_of(s, page_starts),
                      "page_end": page_of(max(e - 1, s), page_starts)}
                     for _t, s, e in spans]
        else:
            text = extract(p)
            if not text or not text.strip():
                stats["no_text"] += 1
                continue
            pieces = chunk(text)
            metas = [{"kind": "text", "src": str(p), "page": None} for _ in pieces]
        if not pieces:
            stats["no_text"] += 1
            continue
        title = p.stem
        prefixed = [as_document(title, c) for c in pieces]
        try:
            vecs = np.vstack([embed(prefixed[i:i + BATCH]) for i in range(0, len(prefixed), BATCH)])
        except Exception as e:  # noqa: BLE001 - prototype: record and continue
            print(f"  ! embed failed {p}: {e}", file=sys.stderr)
            stats["errors"] += 1
            continue
        sha = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()
        with con:
            con.execute("DELETE FROM chunks WHERE file_id IN (SELECT id FROM files WHERE path=?)",
                        (str(p),))
            con.execute("""INSERT INTO files (path, root, mtime, size, sha, chars, n_chunks,
                                              indexed_at, kind)
                           VALUES (?,?,?,?,?,?,?,?,?)
                           ON CONFLICT(path) DO UPDATE SET mtime=excluded.mtime, size=excluded.size,
                             sha=excluded.sha, chars=excluded.chars, n_chunks=excluded.n_chunks,
                             indexed_at=excluded.indexed_at, kind=excluded.kind""",
                        (str(p), str(root), st.st_mtime, st.st_size, sha, len(text),
                         len(pieces), time.time(), "pdf" if pages else "text"))
            fid = con.execute("SELECT id FROM files WHERE path=?", (str(p),)).fetchone()[0]
            con.executemany("INSERT INTO chunks (file_id, ordinal, text, vec, modality, meta) "
                            "VALUES (?,?,?,?,?,?)",
                            [(fid, i, t, v.tobytes(), "text", json.dumps(metas[i]))
                             for i, (t, v) in enumerate(zip(pieces, vecs))])
        stats["indexed"] += 1
        stats["chunks"] += len(pieces)
    stats["seconds"] = round(time.time() - t0, 1)
    return stats


# ---------------------------------------------------------------- query

def matrix(con: sqlite3.Connection):
    rows = con.execute("SELECT id, file_id, ordinal, text, vec, modality, meta "
                       "FROM chunks").fetchall()
    if not rows:
        return rows, np.zeros((0, DIM), dtype=np.float32)
    m = np.frombuffer(b"".join(r[4] for r in rows), dtype=np.float32).reshape(len(rows), DIM)
    return rows, m


def matches_group(chunk_modality: str, file_kind: str, group: str) -> bool:
    """Does a chunk belong to a search-filter group: text / pdf / image / audio / video?"""
    cm, fk = chunk_modality or "text", file_kind or "text"
    if group == "text":
        return cm == "text" and fk == "text"
    if group == "pdf":
        return fk == "pdf"
    if group == "image":
        return cm == "image" or (fk == "image" and cm == "text")
    if group == "audio":
        return cm == "audio"
    if group == "video":
        return cm in ("video_frame", "video_audio")
    return True


def query(text: str, k: int, con: sqlite3.Connection, only: set[str] | None = None,
          path_like: str | None = None, root_like: str | None = None) -> list[dict]:
    """Rank chunks by cosine similarity; filters mask non-matching rows to -inf."""
    rows, m = matrix(con)
    if not rows:
        return []
    files = {r[0]: r for r in con.execute(
        "SELECT id, path, root, kind, size, mtime FROM files")}
    keep = np.ones(len(rows), dtype=bool)
    if only or path_like or root_like:
        pl = path_like.lower() if path_like else None
        rl = root_like.lower() if root_like else None
        for i, r in enumerate(rows):
            f = files.get(r[1])
            if f is None:
                keep[i] = False
            elif only and not any(matches_group(r[5], f[3], g) for g in only):
                keep[i] = False
            elif pl and pl not in f[1].lower():
                keep[i] = False
            elif rl and rl not in (f[2] or "").lower():
                keep[i] = False
    if not keep.any():
        return []
    qv = embed([as_query(text)])[0]
    scores = np.where(keep, m @ qv, -np.inf)
    k = max(1, min(int(k), int(keep.sum())))
    order = np.argsort(-scores)[:k]
    out = []
    for i in order:
        val = float(scores[int(i)])
        if not np.isfinite(val):
            continue
        _cid, fid, ordinal, ctext, _v, modality, meta = rows[int(i)]
        try:
            md = json.loads(meta or "{}")
        except ValueError:
            md = {}
        f = files.get(fid, (None, "?", "", "text", 0, 0.0))
        out.append({"score": val, "path": f[1], "root": f[2] or "", "kind": f[3] or "text",
                    "size": f[4], "mtime": f[5], "chunk": ordinal, "text": ctext,
                    "modality": modality or "text", "t": md.get("t"),
                    "page": md.get("page"), "page_end": md.get("page_end")})
    return out


# ---------------------------------------------------------------- cli

def cmd_index(args) -> None:
    roots = [Path(a) for a in args.paths]
    if not roots and ROOTS_FILE.exists():
        roots = [Path(l.strip()) for l in ROOTS_FILE.read_text().splitlines()
                 if l.strip() and not l.strip().startswith("#")]
    if not roots:
        sys.exit("no roots given and roots.txt is empty")
    con = db()
    opts = {"media": not args.no_media, "frames": not args.no_frames,
            "audio": not args.no_audio, "ocr": args.ocr,
            "interval": args.frame_interval, "segment": args.audio_segment,
            "max_side": args.max_side}
    print(f"[index] media={'on' if opts['media'] else 'off'} frames={opts['frames']} "
          f"audio={opts['audio']} ocr={opts['ocr']} interval={opts['interval']}s "
          f"segment={opts['segment']}s" + ("" if opts["media"] else ""))
    total = {"scanned": 0, "indexed": 0, "chunks": 0, "no_text": 0, "errors": 0,
             "skipped_unchanged": 0, "seconds": 0.0, "media_files": 0, "media_chunks": 0}
    for r in roots:
        print(f"[index] {r}")
        s = index(r, con, force=args.force, opts=opts)
        for kk in total:
            total[kk] += s.get(kk, 0)
        print(f"  scanned={s['scanned']} indexed={s['indexed']} chunks={s['chunks']} "
              f"media={s.get('media_files', 0)}f/{s.get('media_chunks', 0)}c "
              f"unchanged={s['skipped_unchanged']} no_text={s['no_text']} "
              f"errors={s['errors']} {s['seconds']}s")
    print(f"[done] files_indexed={total['indexed']} chunks={total['chunks']} "
          f"scanned={total['scanned']} {round(total['seconds'], 1)}s")


def cmd_query(args) -> None:
    con = db()
    t0 = time.time()
    hits = query(args.text, args.k, con, only=set(args.only) if args.only else None,
                 path_like=args.path, root_like=args.root)
    ms = round((time.time() - t0) * 1000)
    if not hits:
        print("no hits (or the index is empty — run: python gobble.py index)")
        return
    print(f'query: "{args.text}"   ({len(hits)} hits, {ms} ms)\n')
    for n, h in enumerate(hits, 1):
        tag = h.get("modality", "text")
        if h.get("t") is not None:
            tag += f" {_hms(h['t'])}"
        if h.get("page"):
            span = f"-{h['page_end']}" if h.get("page_end") and h["page_end"] != h["page"] else ""
            tag += f" page {h['page']}{span}"
        tag = f" {{{tag}}}" if tag and tag != "text" else ""
        snippet = re.sub(r"\s+", " ", h["text"])[:220]
        print(f"{n:>2}. [{h['score']:.3f}]{tag} {h['path']}  (chunk {h['chunk']})")
        print(f"    {snippet}\n")


def cmd_stats(args) -> None:
    con = db()
    files, chunks = con.execute("SELECT COUNT(*) FROM files").fetchone()[0], \
        con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    size = DB_PATH.stat().st_size / 1e6 if DB_PATH.exists() else 0
    print(f"db       : {DB_PATH}  ({size:.1f} MB)")
    print(f"model    : {MODEL}  dim={DIM}")
    print(f"files    : {files}\nchunks   : {chunks}")
    rows = con.execute("SELECT root, COUNT(*), SUM(n_chunks) FROM files GROUP BY root").fetchall()
    for r, n, c in rows:
        print(f"  {r}  -> {n} files, {c} chunks")
    mod = con.execute("SELECT COALESCE(modality,'text'), COUNT(*) FROM chunks "
                      "GROUP BY 1 ORDER BY 2 DESC").fetchall()
    if mod:
        print("by modality: " + ", ".join(f"{m}={n}" for m, n in mod))


def cmd_serve(args) -> None:
    import gobble_web
    gobble_web.serve(host=args.host, port=args.port, open_browser=args.open)


def main() -> None:
    ap = argparse.ArgumentParser(prog="gobble", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("index", help="index folders (default: roots.txt)")
    pi.add_argument("paths", nargs="*")
    pi.add_argument("--force", action="store_true", help="re-index unchanged files")
    pi.add_argument("--no-media", action="store_true",
                    help="text only: do not embed images / audio / video")
    pi.add_argument("--no-frames", action="store_true", help="video: skip keyframes")
    pi.add_argument("--no-audio", action="store_true", help="video: skip the audio track")
    pi.add_argument("--ocr", action="store_true",
                    help="also OCR images into text chunks (slow, ~2-4 s per image)")
    pi.add_argument("--frame-interval", type=float,
                    default=float(os.environ.get("LOCALSEARCH_FRAME_INTERVAL", "45")),
                    help="seconds between video keyframes (default 45)")
    pi.add_argument("--audio-segment", type=float,
                    default=float(os.environ.get("LOCALSEARCH_AUDIO_SEGMENT", "45")),
                    help="seconds per audio segment (default 45)")
    pi.add_argument("--max-side", type=int,
                    default=int(os.environ.get("LOCALSEARCH_MAX_SIDE", "1024")),
                    help="downscale media to this longest side before embedding")
    pi.set_defaults(func=cmd_index)
    pq = sub.add_parser("query", help="search the index")
    pq.add_argument("text")
    pq.add_argument("-k", type=int, default=8)
    pq.add_argument("--only", action="append",
                    choices=["text", "pdf", "image", "audio", "video"],
                    help="restrict to one kind of content (repeatable)")
    pq.add_argument("--path", help="only files whose path contains this text")
    pq.add_argument("--root", help="only files under an indexed root matching this text")
    pq.set_defaults(func=cmd_query)
    ps = sub.add_parser("stats")
    ps.set_defaults(func=cmd_stats)
    pw = sub.add_parser("serve", help="web UI on http://127.0.0.1:8765")
    pw.add_argument("--port", type=int, default=int(os.environ.get("GOBBLE_PORT", "8765")))
    pw.add_argument("--host", default="127.0.0.1")
    pw.add_argument("--open", action="store_true", help="open a browser window on start")
    pw.set_defaults(func=cmd_serve)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
