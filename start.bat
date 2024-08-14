@echo off

cd "%~dp0"
title python

:: venv check
if not exist "venv\" (
    echo No venv found! Create venv and install dependencies now.
    python -m venv venv
    call .\venv\Scripts\activate.bat
    echo Installing dependencies from requirements.txt
    call pip install -r requirements.txt
    echo Done! Run again to start!
    pause
) else (
    call .\venv\Scripts\activate.bat
    call streamlit run server.py --server.headless true
    pause
)