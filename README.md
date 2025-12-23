# HAWKEYE: Tactical AI Surveillance System

HAWKEYE is a high-performance, military-style surveillance HUD designed for real-time threat detection using ESP32-S3 hardware and YOLOv8 neural networks.

## 🦅 System Architecture
- **Lens**: ESP32-S3 WROOM CAM (QVGA 320x240 @ 15fps)
- **Neural Core**: FastAPI + Ultralytics YOLOv8 (Optimized for CPU)
- **HUD**: React + Tailwind + Framer Motion (Tactical HUD)
- **Persistence**: SQLite (Event Log) + Local Snapshots (Evidence)

## 📡 Hardware Profile (Verified)
- **Stream URL**: `http://172.20.10.3:81/stream` (MJPEG)
- **Exposure**: AEC Sensor ON / AEC DSP OFF / AGC ON
- **Lens**: CAMERA_MODEL_ESP32S3_EYE

## 🚀 One-Click Launch (Windows)
1. Ensure your ESP32-CAM is powered and on the `172.20.10.3` network.
2. Double-click `run.bat` in the project root.
3. Access the HUD at `http://localhost:5173`.

## 🧠 Threat Intelligence Protocol
- **WEAPON_DETECTED**: Identification of firearms, knives, or blunt objects.
- **PERSON_WITH_WEAPON**: Association logic triggered if weapon <60px from person.
- **SUSPICIOUS_GROUP**: Triggered if 4+ individuals cluster for >4 seconds.

## ✅ Operational Checklist (Production)
- [x] Backend connects to `172.20.10.3:81/stream`
- [x] Ingest rate throttled to ~5-8 FPS (Stable CPU)
- [x] HUD displays "ONLINE // LINKED" status
- [x] Deduplication (3s cooldown) prevents alert spam
- [x] Snapshots saved to `backend/data/events/`

---

## Production Specification (v2.0 Refined)

### 🧬 Intelligence Profile
- **Weapon Threshold**: Strict **0.70 (70%)** confidence.
- **Whitelist**: Guns, knives, bats, and tactical weapons.
- **Gating**: Persistence (3 cycles), Area Gating (<40% frame), and Proximity Association.

### 💾 Data Retention
- **Max Storage**: **100 events** (Rolling).
- **Auto-Purge**: Oldest DB rows and image files are deleted when capacity is exceeded.
- **Live Sync**: Archive auto-refreshes every 10 seconds.

### ⚡ Performance
- **Ingest**: Multi-threaded fetcher (Zero-Lag).
- **Processing**: Target 15 FPS (Inference Decoupled).
- **Scaling**: Precision BBox alignment using dynamic metadata.

**HAWKEYE // PROTECTING PERIPHERAL BOUNDARIES**
