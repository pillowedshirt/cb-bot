@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM Binance.US Live Level 8 Council Bot Launcher
REM Starts:
REM - bot.py in one command window
REM - viewer.py through one Streamlit server window
REM - one browser window pointed at the viewer
REM ============================================================

cd /d "%~dp0"

set "PYTHON_EXE=.venv\Scripts\python.exe"
set "VIEWER_PORT=8502"
set "VIEWER_URL=http://localhost:%VIEWER_PORT%"

echo.
echo ============================================
echo   Binance.US Live Level 8 Council Bot Launcher
echo ============================================
echo.

REM ------------------------------------------------------------
REM Create virtual environment if missing
REM ------------------------------------------------------------

if not exist "%PYTHON_EXE%" (
    echo [setup] Creating virtual environment...
    python -m venv .venv

    if errorlevel 1 (
        echo.
        echo [error] Failed to create virtual environment.
        echo Make sure Python is installed and available on PATH.
        echo.
        pause
        exit /b 1
    )
)

REM ------------------------------------------------------------
REM Install/update dependencies
REM ------------------------------------------------------------

echo [setup] Updating pip...
"%PYTHON_EXE%" -m pip install --upgrade pip

if errorlevel 1 (
    echo.
    echo [error] Failed to update pip.
    echo.
    pause
    exit /b 1
)

if exist "requirements.txt" (
    echo [setup] Installing requirements.txt...
    "%PYTHON_EXE%" -m pip install -r requirements.txt

    if errorlevel 1 (
        echo.
        echo [error] Failed to install requirements.
        echo.
        pause
        exit /b 1
    )
) else (
    echo [warning] requirements.txt not found. Skipping dependency install.
)

REM ------------------------------------------------------------
REM Check required files
REM ------------------------------------------------------------

if not exist "bot.py" (
    echo.
    echo [error] bot.py not found in this folder.
    echo.
    pause
    exit /b 1
)

if not exist "viewer.py" (
    echo.
    echo [error] viewer.py not found in this folder.
    echo.
    pause
    exit /b 1
)

if not exist "ai_brain.py" (
    echo [warning] ai_brain.py not found.
)

if not exist "level8_council.py" (
    echo.
    echo [error] level8_council.py not found.
    echo Level 8 council mode requires level8_council.py.
    echo.
    pause
    exit /b 1
)

if not exist ".env" (
    echo.
    echo [warning] No .env file found.
    echo Make sure your Binance.US .env credentials are configured before live trading.
    echo.
)

REM ------------------------------------------------------------
REM Stop any old Streamlit server on this viewer port
REM ------------------------------------------------------------

echo [setup] Checking for old Streamlit server on port %VIEWER_PORT%...

for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":%VIEWER_PORT% .*LISTENING"') do (
    echo [setup] Stopping old process on port %VIEWER_PORT% with PID %%P...
    taskkill /PID %%P /F >nul 2>nul
)

REM ------------------------------------------------------------
REM Warn if another bot.py already appears to be running
REM ------------------------------------------------------------

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
 "$self = $PID; $procs = Get-CimInstance Win32_Process | Where-Object { ($_.ProcessId -ne $self) -and ($_.Name -in @('python.exe','pythonw.exe')) -and ($_.CommandLine -match '(^|[\\s])bot\.py($|[\s])') }; if ($procs) { $procs | ForEach-Object { Write-Host ('[lock-check] Existing bot.py PID=' + $_.ProcessId + ' CMD=' + $_.CommandLine) }; exit 2 } else { exit 0 }"

if errorlevel 2 (
    echo.
    echo [ERROR] Another Python bot.py process appears to be running.
    echo Close the existing bot window before launching a new live trading instance.
    echo.
    pause
    exit /b 1
)

REM ------------------------------------------------------------
REM Start bot in its own command window
REM ------------------------------------------------------------

echo [check] Compiling Python files before launch...
"%PYTHON_EXE%" -m py_compile bot.py level8_council.py ai_brain.py backtest_intelligence.py viewer.py price_action_context.py previous_session_volume_profile.py quant_context.py session_liquidity.py

if errorlevel 1 (
    echo.
    echo [error] Python compile check failed. Bot was not started.
    echo.
    pause
    exit /b 1
)

echo [run] Starting Level 8 live bot...
start "LEVEL 8 COUNCIL BOT" cmd /k "cd /d "%~dp0" && "%PYTHON_EXE%" bot.py"

REM Give the bot a moment to start writing or appending CSVs
timeout /t 3 /nobreak >nul

REM ------------------------------------------------------------
REM Start Streamlit viewer in its own command window
REM ------------------------------------------------------------

echo [run] Starting Streamlit viewer on port %VIEWER_PORT%...
start "LEVEL 8 COUNCIL VIEWER SERVER" cmd /k "cd /d "%~dp0" && "%PYTHON_EXE%" -m streamlit run viewer.py --server.port %VIEWER_PORT% --server.headless true --browser.gatherUsageStats false"

REM Give Streamlit a moment to boot
timeout /t 5 /nobreak >nul

REM ------------------------------------------------------------
REM Open viewer in exactly one browser window
REM ------------------------------------------------------------

echo [run] Opening viewer in one browser window...

where msedge >nul 2>nul
if not errorlevel 1 (
    start "LEVEL 8 COUNCIL VIEWER" msedge --new-window "%VIEWER_URL%"
    goto launched
)

where chrome >nul 2>nul
if not errorlevel 1 (
    start "LEVEL 8 COUNCIL VIEWER" chrome --new-window "%VIEWER_URL%"
    goto launched
)

REM Fallback to default browser
start "LEVEL 8 COUNCIL VIEWER" "%VIEWER_URL%"

:launched

echo.
echo ============================================
echo Started:
echo - Bot window: LEVEL 8 COUNCIL BOT
echo - Viewer server window: LEVEL 8 COUNCIL VIEWER SERVER
echo - Viewer URL: %VIEWER_URL%
echo.
echo Notes:
echo - Streamlit is started with --server.headless true, so it will not auto-open a second browser window.
echo - Old CSV files are preserved and reused.
echo - If meaningful old CSV rows existed before this launch, bot.py adopts holdings instead of startup-liquidating them.
echo - If no meaningful old CSV rows existed before this launch, bot.py reconciles existing Binance.US holdings before live trading.
echo ============================================
echo.
echo You can close this launcher window.
echo The bot and viewer will keep running in their own windows.
echo.

endlocal
