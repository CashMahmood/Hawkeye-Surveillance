# HAWKEYE Surveillance System

> **Tactical Real-Time Intelligent Monitoring Console**

Hawkeye is a professional-grade surveillance system designed to monitor live camera feeds and automatically detect potential threats using AI-based object detection. It is optimized for CPU-only environments and features a high-performance tactical dashboard.

## 🏗️ Architecture

```text
+---------------------+       +---------------------------+       +------------------------+
|    ESP32-CAM        | ----> |      FastAPI Backend      | <---> |   React Tactical UI    |
| (MJPEG Stream)      | stream| (YOLOv8 Inference Engine)  | WS/API| (Framer Motion HUD)    |
+---------------------+       +---------------------------+       +------------------------+
                                        |             |
                                        v             v
                                [ SQLite DB ]   [ Local JPG Storage ]
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+

### 1-Click Launch (Windows)
Double-click `run.bat` in the root directory. This will:
1. Initialize the backend server (Port 8000).
2. Initialize the frontend console (Port 5173).
3. Open the dashboard in your default browser.

### Manual Setup

#### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

#### Frontend
```bash
cd frontend
npm install
npm run dev
```

## ⚙️ Configuration
Modify `backend/app/config.py` to customize system behavior:
- `ESP_STREAM_URL`: Address of your camera stream.
- `CONF_THRESH`: Detection confidence (0.35 default).
- `FRAME_SKIP`: Frames to skip between AI passes (2 default).
- `GROUP_DISTANCE_PX`: Distance threshold for suspicious group clustering.

## 🛡️ Threat Detection Logic
1. **WEAPON_DETECTED**: Triggered when a firearm, knife, or other weapon is identified.
2. **PERSON_WITH_WEAPON**: Association logic triggered when a weapon is within proximity of an individual.
3. **SUSPICIOUS_GROUP**: Rule-based detection for $N$ individuals clustered in a tight radius for $T$ seconds.

## 🛠️ Tech Stack
- **Inference**: Ultralytics YOLOv8 (Nano)
- **Backend**: FastAPI, OpenCV, SQLite
- **Frontend**: React, Vite, Tailwind CSS, Framer Motion, Lucide Icons

## 📜 Known Limitations
- optimized for CPU; high person counts may increase latency.
- MJPEG stream stability depends on network quality.
- Group detection is distance-based; depth perception is estimated from 2D coordinates.

---
**HAWKEYE SURVEILLANCE // OPERATIONAL**
