@echo off
REM GOBBLE - start the local web UI against the main index (port 8765).
REM Needs a Python 3 with numpy. Set PY first to force a specific one:
REM   set "PY=C:\Python312\python.exe" && gobble-serve.cmd
setlocal
set "PYTHONPATH="
if not defined PY python -c "import numpy" >nul 2>nul && set "PY=python"
if not defined PY for /f "delims=" %%I in ('py -c "import sys;print(sys.executable)" 2^>nul') do (
  if not defined PY "%%I" -c "import numpy" >nul 2>nul && set "PY=%%I"
)
if not defined PY (
  echo GOBBLE needs a Python 3 with numpy. Set PY to your interpreter, then re-run.
  exit /b 1
)
set "LOCALSEARCH_BACKEND=openai"
set "LOCALSEARCH_MODEL=embeddinggemma-2-bf16"
set "LOCALSEARCH_DB=%~dp0index\search-embeddinggemma-2-bf16-openai-256-gemma.db"
set "GOBBLE_PORT=8765"
cd /d "%~dp0"
echo GOBBLE: port 8765
"%PY%" gobble.py serve --open
endlocal
