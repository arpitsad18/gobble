@echo off
REM GOBBLE - same UI, but against a small throw-away demo index (port 8766), so
REM trying things out never touches your real index.
REM Needs a Python 3 with numpy. Set PY first to force a specific one:
REM   set "PY=C:\Python312\python.exe" && gobble-test.cmd
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
set "LOCALSEARCH_DB=%~dp0index\test-search.db"
set "GOBBLE_PORT=8766"
cd /d "%~dp0"
echo GOBBLE: port 8766
"%PY%" gobble.py serve --open
endlocal
