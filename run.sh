#!/bin/bash

echo "[1/3] Navigating to Project Root..."
cd "$(dirname "$0")"

echo "[2/3] Initializing Backend..."
cd backend
if [ ! -d "venv" ]; then
    python3 -m venv venv
    ./venv/bin/pip install -r requirements.txt
fi
source venv/bin/activate
nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
BACKEND_PID=$!
cd ..

echo "[3/3] Initializing Frontend..."
cd frontend
if [ ! -d "node_modules" ]; then
    npm install
fi
npm run dev &
FRONTEND_PID=$!

echo "---------------------------------------------------"
echo "HAWKEYE SURVEILLANCE CONSOLE IS LAUNCHING"
echo "Backend: http://localhost:8000 (PID: $BACKEND_PID)"
echo "Frontend: http://localhost:5173 (PID: $FRONTEND_PID)"
echo "---------------------------------------------------"

trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT TERM
wait
