"""GOBBLE web UI - a tiny localhost server for gobble.html.

Serves one static page plus a JSON API over the index in gobble.py, so the
browser can search, preview the original file (with HTTP range requests, which
is what makes video seeking work) and add folders to the index.

    python gobble.py serve            # or: python gobble_web.py
    python gobble_web.py --port 8765 --open

Everything binds to 127.0.0.1 - nothing is exposed to the network and no
request leaves the machine except the embedding call to your own llama-server.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gobble as core  # noqa: E402  (sibling module)

PAGE = HERE / "gobble.html"
GROUP_NAMES = ("text", "pdf", "image", "audio", "video")


# ---------------------------------------------------------------- helpers

def _hms(sec: float | None) -> str:
    if sec is None:
        return ""
    s = int(max(0, sec))
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _indexed_paths(con) -> set[str]:
    return {str(r[0]).replace("\\", "/").lower() for r in con.execute("SELECT path FROM files")}


def embed_ok() -> tuple[bool, str]:
    try:
        v = core.embed(["health check"])
        return True, f"dim={len(v[0])}"
    except Exception as e:  # noqa: BLE001 - surfaced to the UI verbatim
        return False, str(e)[:300]


def stats() -> dict:
    con = core.db()
    files = con.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    chunks = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    by_mod = dict(con.execute(
        "SELECT COALESCE(modality,'text'), COUNT(*) FROM chunks GROUP BY 1 ORDER BY 2 DESC"))
    by_kind = dict(con.execute(
        "SELECT COALESCE(kind,'text'), COUNT(*) FROM files GROUP BY 1 ORDER BY 2 DESC"))
    roots = [{"root": r[0], "files": r[1], "chunks": r[2] or 0}
             for r in con.execute("SELECT root, COUNT(*), SUM(n_chunks) FROM files GROUP BY root")]
    size = core.DB_PATH.stat().st_size / 1e6 if core.DB_PATH.exists() else 0.0
    ok, detail = embed_ok()
    return {"db": str(core.DB_PATH), "db_mb": round(size, 1), "model": core.MODEL,
            "dim": core.DIM, "backend": core.BACKEND, "files": files, "chunks": chunks,
            "by_modality": by_mod, "by_kind": by_kind, "roots": roots,
            "embed_ok": ok, "embed_detail": detail, "pdf": core.PDFTOTEXT or "",
            "ffmpeg": core._ffmpeg() or ""}


# ---------------------------------------------------------------- index job

class Job:
    """One indexing run, executed on a worker thread and polled by the page."""

    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.running = False
        self.started = None
        self.finished = None
        self.paths: list[str] = []
        self.log: list[str] = []
        self.result: dict | None = None
        self.error: str | None = None

    def say(self, line: str):
        with self.lock:
            self.log.append(line)
            del self.log[:-400]

    def snapshot(self) -> dict:
        with self.lock:
            return {"running": self.running, "paths": self.paths, "result": self.result,
                    "error": self.error, "log": self.log[-200:],
                    "started": self.started, "finished": self.finished,
                    "seconds": (round((self.finished or time.time()) - self.started, 1)
                                if self.started else None)}

    def start(self, paths: list[str], force: bool, media: bool, ocr: bool) -> bool:
        with self.lock:
            if self.running:
                return False
            self.reset()
            self.running = True
            self.started = time.time()
            self.paths = list(paths)
        t = threading.Thread(target=self._run, args=(paths, force, media, ocr), daemon=True)
        t.start()
        return True

    def _run(self, paths, force, media, ocr):
        total = {"scanned": 0, "indexed": 0, "chunks": 0, "no_text": 0, "errors": 0,
                 "skipped_unchanged": 0, "media_files": 0, "media_chunks": 0, "seconds": 0.0}
        opts = {"media": media, "frames": media, "audio": media, "ocr": ocr}
        try:
            con = core.db()
            for raw in paths:
                root = Path(raw).expanduser()
                if not root.exists():
                    self.say(f"! not found: {root}")
                    continue
                self.say(f"[index] {root}")
                s = core.index(root, con, force=force, opts=opts)
                for kk in total:
                    total[kk] += s.get(kk, 0)
                self.say(f"  scanned={s['scanned']} indexed={s['indexed']} chunks={s['chunks']} "
                         f"media={s.get('media_files', 0)}f/{s.get('media_chunks', 0)}c "
                         f"unchanged={s['skipped_unchanged']} no_text={s['no_text']} "
                         f"errors={s['errors']} {s['seconds']}s")
            with self.lock:
                self.result = total
            self.say(f"[done] files_indexed={total['indexed']} chunks={total['chunks']}")
        except Exception as e:  # noqa: BLE001 - report to the page, keep serving
            with self.lock:
                self.error = f"{type(e).__name__}: {e}"
            self.say(f"! {self.error}")
        finally:
            with self.lock:
                self.running = False
                self.finished = time.time()


JOB = Job()


# ---------------------------------------------------------------- api

def api_search(q: dict) -> dict:
    text = (q.get("q") or [""])[0].strip()
    if not text:
        return {"hits": [], "count": 0, "ms": 0, "q": ""}
    k = int((q.get("k") or ["10"])[0] or 10)
    only = {g for g in (q.get("only") or [""])[0].split(",") if g in GROUP_NAMES} or None
    path_like = (q.get("path") or [""])[0].strip() or None
    root_like = (q.get("root") or [""])[0].strip() or None
    con = core.db()
    t0 = time.time()
    hits = core.query(text, max(1, min(k, 200)), con, only=only,
                      path_like=path_like, root_like=root_like)
    ms = round((time.time() - t0) * 1000)
    files = {r[0]: r for r in con.execute("SELECT path, size, mtime FROM files")}
    for h in hits:
        info = files.get(Path(h["path"]).as_posix()) or files.get(h["path"])
        h["size"] = info[1] if info else h.get("size")
        h["mtime"] = info[2] if info else h.get("mtime")
        h["time"] = _hms(h.get("t")) if h.get("t") is not None else None
        h["name"] = Path(h["path"]).name
        h["preview"] = "/api/raw?path=" + _q(h["path"]) if h.get("size") else None
    return {"hits": hits, "count": len(hits), "ms": ms, "q": text,
            "only": sorted(only or []), "path": path_like, "root": root_like}


def _q(s: str) -> str:
    from urllib.parse import quote
    return quote(str(s), safe="")


def serve_file(handler, raw_path: str):
    """Stream an indexed file, honouring HTTP Range so <video> can seek."""
    want = str(Path(unquote(raw_path)))
    con = core.db()
    allowed = _indexed_paths(con)
    norm = want.replace("\\", "/").lower()
    if norm not in allowed:
        handler.send_error(403, "not in index")
        return
    p = Path(want)
    if not p.is_file():
        handler.send_error(404, "missing on disk")
        return
    size = p.stat().st_size
    ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    start, end = 0, size - 1
    code = 200
    rng = handler.headers.get("Range")
    if rng:
        m = re.match(r"bytes=(\d*)-(\d*)$", rng.strip())
        if m:
            if m.group(1):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
            elif m.group(2):  # suffix range
                start = max(0, size - int(m.group(2)))
            end = min(end, size - 1)
            if start > end or start >= size:
                handler.send_response(416)
                handler.send_header("Content-Range", f"bytes */{size}")
                handler.send_header("Content-Length", "0")
                handler.end_headers()
                return
            code = 206
    length = end - start + 1
    handler.send_response(code)
    handler.send_header("Content-Type", ctype)
    handler.send_header("Accept-Ranges", "bytes")
    handler.send_header("Content-Length", str(length))
    handler.send_header("Cache-Control", "no-store")
    if code == 206:
        handler.send_header("Content-Range", f"bytes {start}-{end}/{size}")
    handler.end_headers()
    try:
        with p.open("rb") as fh:
            fh.seek(start)
            left = length
            while left > 0:
                buf = fh.read(min(262144, left))
                if not buf:
                    break
                handler.wfile.write(buf)
                left -= len(buf)
    except (BrokenPipeError, ConnectionResetError):
        pass


def reveal_file(raw_path: str) -> dict:
    p = Path(unquote(raw_path))
    if not p.exists():
        return {"ok": False, "error": "missing on disk"}
    if os.name == "nt":
        # /select highlights the file itself instead of blinking the folder
        subprocess.Popen(["explorer", "/select,", str(p)])
    elif sys.platform == "darwin":
        subprocess.Popen(["open", "-R", str(p)])
    else:
        subprocess.Popen(["xdg-open", str(p.parent)])
    return {"ok": True, "path": str(p)}


# ---------------------------------------------------------------- http

class Handler(BaseHTTPRequestHandler):
    server_version = "GOBBLE/1.0"

    def log_message(self, fmt, *args):  # keep the console readable
        if os.environ.get("GOBBLE_VERBOSE"):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # -- plumbing ---------------------------------------------------------
    def _json(self, obj, code=200):
        body = json.dumps(obj, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8") or "{}")
        except ValueError:
            return {}

    def do_GET(self):  # noqa: N802 - stdlib naming
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path in ("/", "/index.html", "/gobble.html"):
                if not PAGE.exists():
                    self.send_error(500, "gobble.html is missing next to gobble_web.py")
                    return
                body = PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif u.path == "/api/stats":
                self._json(stats())
            elif u.path == "/api/search":
                self._json(api_search(q))
            elif u.path == "/api/index":
                self._json(JOB.snapshot())
            elif u.path == "/api/raw":
                serve_file(self, (q.get("path") or [""])[0])
            elif u.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
            else:
                self.send_error(404, "no such path")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # noqa: BLE001 - a bad request must not kill the server
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_POST(self):  # noqa: N802
        u = urlparse(self.path)
        try:
            if u.path == "/api/index":
                data = self._body()
                paths = data.get("paths") or []
                if isinstance(paths, str):
                    paths = [paths]
                if not paths and core.ROOTS_FILE.exists():
                    paths = [l.strip() for l in core.ROOTS_FILE.read_text(
                        encoding="utf-8", errors="replace").splitlines()
                        if l.strip() and not l.strip().startswith("#")]
                if not paths:
                    self._json({"ok": False, "error": "no folders given"}, 400)
                    return
                started = JOB.start(paths, bool(data.get("force")), data.get("media", True) is not False,
                                    bool(data.get("ocr")))
                self._json({"ok": started, "started": started, "running": JOB.running,
                            "error": None if started else "an index run is already going"})
            elif u.path == "/api/open":
                self._json(reveal_file((self._body().get("path") or "")))
            else:
                self.send_error(404, "no such path")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # noqa: BLE001
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = False) -> None:
    ok, detail = embed_ok()
    print(f"[gobble] db      : {core.DB_PATH}")
    print(f"[gobble] model   : {core.MODEL}  dim={core.DIM}  backend={core.BACKEND}")
    print(f"[gobble] embedder: {'ready ' + detail if ok else 'NOT READY - ' + detail}")
    print(f"[gobble] pdftotext: {core.PDFTOTEXT or 'not found (PDFs will fall back to OCR)'}")
    print(f"[gobble] ui      : http://{host}:{port}   (ctrl-c to stop)")
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(f"http://{host}:{port}/")).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[gobble] stopped")
    finally:
        httpd.server_close()


def main() -> None:
    ap = argparse.ArgumentParser(prog="gobble_web", description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("GOBBLE_PORT", "8765")))
    ap.add_argument("--open", action="store_true", help="open a browser window on start")
    a = ap.parse_args()
    serve(a.host, a.port, a.open)


if __name__ == "__main__":
    main()
