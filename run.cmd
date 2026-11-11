@echo off
setlocal
set "APP_PY=%~dp0.venv\Scripts\pythonw.exe"
if not exist "%APP_PY%" (
    echo Python environment is missing. Follow the setup in README.md.
    pause
    exit /b 1
)
start "" "%APP_PY%" "%~dp0app.py" %*
