@echo off

cd "%~dp0"
title python

:: venv check
call .\venv\Scripts\activate.bat
title venv-%venvname%
cmd /K