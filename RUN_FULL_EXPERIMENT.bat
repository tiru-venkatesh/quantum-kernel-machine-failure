@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo First run START_QML_WINDOWS.bat to install packages.
  pause
  exit /b 1
)
echo Running the FULL experiment. This may take several minutes or longer.
.venv\Scripts\python.exe run_experiment.py --per-class 120 --shots 1024 --reps 2 --output results
if errorlevel 1 (
  echo Experiment failed. Copy the complete error above.
  pause
  exit /b 1
)
echo.
echo Full run complete. Open results\run_summary.txt
pause
