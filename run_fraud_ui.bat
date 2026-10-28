@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE="
if exist "%LocalAppData%\Programs\Python\Python312\python.exe" set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python312\python.exe"
for /f "delims=" %%P in ('where py 2^>nul') do if not defined PYTHON_EXE set "PYTHON_EXE=%%P"
if not defined PYTHON_EXE for /f "delims=" %%P in ('where python 2^>nul') do if not defined PYTHON_EXE set "PYTHON_EXE=%%P"

if not defined PYTHON_EXE (
  echo Python could not be found.
  echo Install Python from https://www.python.org/downloads/ and select "Add Python to PATH".
  pause
  exit /b 1
)

echo Using Python: %PYTHON_EXE%
echo Installing the small set of packages needed by the fraud-detection UI...
"%PYTHON_EXE%" -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo Package installation failed. Read the error shown above.
  pause
  exit /b 1
)
echo.
echo Starting FraudGuard in your browser...
"%PYTHON_EXE%" -m streamlit run fraud_detection_ui.py --browser.gatherUsageStats false
if errorlevel 1 (
  echo.
  echo FraudGuard could not start. Read the error shown above.
  pause
  exit /b 1
)
endlocal
