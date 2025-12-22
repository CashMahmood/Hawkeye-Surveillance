@echo off
setlocal

echo [1/3] Navigating to Project Root...
cd %~dp0

echo [2/3] Initializing Backend...
start cmd /k "cd backend && call venv\Scripts\activate && uvicorn app.main:app --host 0.0.0.0 --port 8000"

timeout /t 5

echo [3/3] Initializing Frontend...
start cmd /k "cd frontend && npm run dev"

echo.
echo Hawkeye Surveillance Console is launching.
echo Backend: http://localhost:8000
echo Frontend: http://localhost:5173
echo.
pause
