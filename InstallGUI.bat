@echo off
title AI Toolkit UI Installer

echo ======================================================
echo AI Toolkit Configuration Interface - Installer
echo ======================================================
echo.

REM Check if virtual environment exists
if not exist venv\Scripts\activate.bat (
    echo ERROR: Virtual environment not found at venv\Scripts\activate.bat
    echo Please make sure you're running this from your AI Toolkit root directory.
    echo.
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Installing required packages for the UI...
pip install customtkinter pyyaml

echo.
echo Checking installation...
python -c "import customtkinter, yaml" >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Installation failed. Please check for errors above.
    pause
    exit /b 1
)

echo.
echo ======================================================
echo Installation completed successfully!
echo.
echo You can now start the UI by running:
echo    start_ui.bat
echo ======================================================
echo.

REM Create the start_ui.bat file
echo @echo off > start_ui.bat
echo title AI Toolkit Configuration Interface >> start_ui.bat
echo. >> start_ui.bat
echo echo Starting AI Toolkit Configuration Interface... >> start_ui.bat
echo call venv\Scripts\activate.bat >> start_ui.bat
echo python ai_toolkit_gui.py >> start_ui.bat
echo. >> start_ui.bat

echo The start_ui.bat file has been created.
echo.
pause