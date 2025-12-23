@echo off
setlocal
title HAWKEYE_CONTROL_CENTER

echo ===================================================
echo HAWKEYE SURVEILLANCE // SYSTEM INITIALIZATION
echo ===================================================

echo [1/3] CHECKING PROJECT ROOT...
cd %~dp0

echo [2/3] INITIALIZING BACKEND UPLINK...
if not exist "backend\venv" (
    echo [!] Backend VENV not found. Creating...
    python -m venv backend\venv
    call backend\venv\Scripts\activate
    pip install -r backend\requirements.txt
)
start "HAWKEYE_BACKEND" cmd /k "cd backend && call venv\Scripts\activate && uvicorn app.main:app --host 0.0.0.0 --port 8000"

echo [WAIT] Waiting for Backend to stabilize...
timeout /t 5

echo [3/3] INITIALIZING TACTICAL HUD...
if not exist "frontend\node_modules" (
    echo [!] Frontend node_modules not found. Installing...
    cd frontend && npm install && cd ..
)
start "HAWKEYE_FRONTEND" cmd /k "cd frontend && npm run dev"

echo.
echo ===================================================
echo SYSTEM ONLINE // NEURAL LINK ESTABLISHED
echo ===================================================
echo HUB: http://localhost:5173
echo API: http://localhost:8000/health
echo ===================================================
echo.

:: Automatically launch browser
timeout /t 3 /nobreak > nul
start http://localhost:5173

pause
