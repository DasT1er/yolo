@echo off
title YOLO Training Studio
echo.
echo ========================================
echo    YOLO Training Studio
echo    Professional Object Detection Training
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.10 or higher.
    pause
    exit /b 1
)

REM Check if we're in a virtual environment
if defined VIRTUAL_ENV (
    echo [INFO] Virtual environment detected: %VIRTUAL_ENV%
) else (
    REM Check if venv exists and activate it
    if exist "venv\Scripts\activate.bat" (
        echo [INFO] Activating virtual environment...
        call venv\Scripts\activate.bat
    )
)

echo.
echo [INFO] Starting YOLO Training Studio...
echo [INFO] Open your browser at: http://localhost:7860
echo.
echo Press Ctrl+C to stop the server.
echo.

REM Start the application
python -m src.app

REM If there was an error, pause to show it
if errorlevel 1 (
    echo.
    echo [ERROR] Application crashed or failed to start.
    echo Check the error message above for details.
    echo.
    pause
)
