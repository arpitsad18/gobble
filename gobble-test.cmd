@echo off
REM GOBBLE - same UI, but pointed at the small throw-away test index
REM (D:\search test) on port 8766. Handy for demoing without touching the
REM real index.
setlocal
set "PYTHONPATH="
set "PY=E:\Python\python.exe"
set "LOCALSEARCH_BACKEND=openai"
set "LOCALSEARCH_MODEL=embeddinggemma-2-bf16"
set "LOCALSEARCH_DB=%~dp0index\test-search.db"
set "GOBBLE_PORT=8766"
cd /d "%~dp0"
"%PY%" gobble.py serve --open
endlocal
