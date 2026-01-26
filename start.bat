@echo off
title YOLO Training Studio
cd /d "%~dp0"

echo.
echo ========================================
echo    YOLO Training Studio
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo Install Python 3.10+ from python.org
    pause
    exit /b 1
)

REM Install dependencies if needed
if not exist "venv" (
    echo [INFO] First run - installing dependencies...
    python -m pip install customtkinter ultralytics pillow pyyaml --quiet
)

echo [INFO] Starting application...
python app.py

if errorlevel 1 (
    echo.
    echo [ERROR] Application error. Check above for details.
    pause
)
