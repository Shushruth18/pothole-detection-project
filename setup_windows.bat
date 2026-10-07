@echo off
cd /d "%~dp0"
python --version >nul 2>&1
if errorlevel 1 goto launcher
python -m venv .venv
if errorlevel 1 goto fail
goto install
:launcher
py -3 -m venv .venv
if errorlevel 1 goto fail
:install
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto fail
echo Setup complete. Run run_app.bat to open RoadWatch.
pause
exit /b 0
:fail
echo Setup failed. Check the error above. Python 3.11 or 3.12 is recommended.
pause
exit /b 1
