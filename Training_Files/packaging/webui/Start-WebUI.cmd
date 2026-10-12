@echo off
cd /d "%~dp0"
if not exist ".venv-yolo\Scripts\python.exe" (
  echo Run Install-Dependencies.cmd first.
  pause
  exit /b 1
)
".venv-yolo\Scripts\python.exe" -u c1_webui.py --no-gallery
if errorlevel 1 (
  echo WebUI failed. Read the error above.
  pause
)
