@echo off
REM GOBBLE - start the local web UI against the main index (port 8765).
setlocal
set "PYTHONPATH="
set "PY=E:\Python\python.exe"
set "LOCALSEARCH_BACKEND=openai"
set "LOCALSEARCH_MODEL=embeddinggemma-2-bf16"
set "LOCALSEARCH_DB=%~dp0index\search-embeddinggemma-2-bf16-openai-256-gemma.db"
set "GOBBLE_PORT=8765"
cd /d "%~dp0"
"%PY%" gobble.py serve --open
endlocal
