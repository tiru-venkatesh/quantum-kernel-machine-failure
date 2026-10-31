@echo off
setlocal
cd /d "%~dp0"
echo ================================================
echo QML AI4I - Windows setup and quick experiment
echo ================================================
echo.
python --version
if errorlevel 1 (
  echo Python was not found. Install Python 3.11 from https://www.python.org/downloads/ and tick "Add Python to PATH".
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  python -m venv .venv
  if errorlevel 1 goto :fail
)
echo Installing required packages. This needs an internet connection...
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto :fail
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto :fail
echo.
echo Running the QUICK smoke test (30 records per class, 256 shots, 1 repetition)...
.venv\Scripts\python.exe run_experiment.py --per-class 30 --shots 256 --reps 1 --output results_quick
if errorlevel 1 goto :fail
echo.
echo SUCCESS. Open results_quick\run_summary.txt
echo Do not present this quick run as the final full experiment.
pause
exit /b 0
:fail
echo.
echo SETUP OR EXPERIMENT FAILED.
echo Please copy the complete error text above and send it to ChatGPT.
pause
exit /b 1
