import asyncio
import cv2
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse
import os
import time

from .config import ESP_STREAM_URL, FRAME_SKIP, SAVE_DIR, CONF_THRESH_PERSON, CONF_THRESH_WEAPON
from .db import get_events, get_event_by_id
from .inference import DetectionSystem

app = FastAPI(title="Hawkeye Surveillance System")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Detection System (Lazy loaded to avoid blocking startup)
detector = None

def load_detector_async():
    global detector, latest_detections
    latest_detections["status"] = "MODEL_SYNC"
    print("[SYSTEM] ACQUIRING NEURAL WEIGHTS...")
    detector = DetectionSystem()
    
    m_gen = os.path.basename(detector.model_gen.ckpt_path) if hasattr(detector.model_gen, 'ckpt_path') and detector.model_gen.ckpt_path else "yolov8m"
    m_spec = os.path.basename(detector.model_weapons.ckpt_path) if hasattr(detector.model_weapons, 'ckpt_path') and detector.model_weapons.ckpt_path else "custom"
    
    latest_detections["debug"]["model_used"] = f"Hybrid ({m_gen} + {m_spec})"
    print(f"[SYSTEM] NEURAL CONVERGENCE COMPLETE: {latest_detections['debug']['model_used']}")

# State
is_camera_connected = False
latest_frame = None
last_frame_time = 0
latest_detections = {
    "timestamp": "",
    "fps": 0,
    "counts": {"persons": 0, "weapons": 0},
    "threats": [],
    "boxes": [],
    "status": "INITIALIZING",
    "frame_dims": [0, 0],
    "debug": {"model_used": "yolov8m (pending)"}
}
clients = set()

# Mount static files for events
app.mount("/images", StaticFiles(directory=SAVE_DIR), name="images")

def video_fetcher():
    """Constantly grabs frames from the ESP32-CAM to clear buffers."""
    global latest_frame, is_camera_connected
    while True:
        print(f"[STREAM] ATTEMPTING LINK: {ESP_STREAM_URL}")
        cap = cv2.VideoCapture(ESP_STREAM_URL)
        if not cap.isOpened():
            is_camera_connected = False
            latest_frame = None
            print(f"[STREAM] LINK FAILURE: Destination {ESP_STREAM_URL} unreachable.")
            time.sleep(3)
            continue
            
        is_camera_connected = True
        print(f"[STREAM] UPLINK ESTABLISHED -> {ESP_STREAM_URL}")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                is_camera_connected = False
                latest_frame = None # Clear stale frame
                break
            latest_frame = frame
            last_frame_time = time.time()
            # No sleep here - grab as fast as possible
            
        cap.release()
        time.sleep(1)

def video_processing_loop():
    """Handles inference and broadcasting at a stable rate."""
    global latest_detections, latest_frame
    frame_count = 0
    start_time = time.time()
    
    while True:
        t1 = time.time()
        
        if latest_frame is not None:
            frame = latest_frame.copy()
            
            # Update dims immediately if not set
            if latest_detections["frame_dims"] == [0, 0]:
                h, w = frame.shape[:2]
                latest_detections["frame_dims"] = [w, h]

            if frame_count % FRAME_SKIP == 0 and detector is not None:
                boxes, persons, weapons, dims = detector.detect(frame)
                threats = detector.process_threats(frame, boxes, persons, weapons)
                
                end_time = time.time()
                fps = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0
                start_time = end_time
                
                latest_detections.update({
                    "timestamp": str(time.time()),
                    "fps": round(fps, 1),
                    "counts": {
                        "persons": len(persons),
                        "weapons": len(weapons)
                    },
                    "threats": threats,
                    "boxes": boxes,
                    "status": "CONNECTED",
                    "frame_dims": dims
                })
            elif detector is None:
                latest_detections["status"] = "MODEL_SYNC"
            else:
                # Connected but skipped frame
                latest_detections["status"] = "CONNECTED"
        else:
            # No frame available
            latest_detections["fps"] = 0
            if is_camera_connected:
                # Camera thread says linked, but no frames (possible buffer stall or auth)
                latest_detections["status"] = "MODEL_SYNC" if detector is None else "UPLINK_STALL"
            else:
                latest_detections["status"] = "OFFLINE"
                latest_detections["frame_dims"] = [0, 0]

        # Constant Broadcast (Heartbeat)
        for client in list(clients):
            try:
                asyncio.run_coroutine_threadsafe(client.send_json(latest_detections), loop)
            except Exception:
                pass
        
        frame_count += 1
        elapsed = time.time() - t1
        time.sleep(max(0.01, 0.066 - elapsed))

# Start background threads
loop = asyncio.get_event_loop()
threading.Thread(target=load_detector_async, daemon=True).start()
threading.Thread(target=video_fetcher, daemon=True).start()
threading.Thread(target=video_processing_loop, daemon=True).start()

@app.get("/health")
def health():
    return {
        "status": "ok", 
        "stream": "CONNECTED" if is_camera_connected else "DISCONNECTED",
        "last_frame_age": round(time.time() - last_frame_time, 1) if last_frame_time > 0 else "N/A",
        "model": latest_detections["debug"]["model_used"],
        "clients": len(clients)
    }

@app.get("/events")
def list_events():
    return get_events()

@app.get("/events/{event_id}")
def event_details(event_id: int):
    event = get_event_by_id(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return event

@app.get("/events/{event_id}/image")
def get_event_image(event_id: int):
    event = get_event_by_id(event_id)
    if not event or not event['image_path']:
        raise HTTPException(status_code=404, detail="Image not found")
    
    path = os.path.join(SAVE_DIR, event['image_path'])
    return FileResponse(path)

@app.websocket("/ws/detections")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    try:
        while True:
            await websocket.receive_text() # Keep alive
    except WebSocketDisconnect:
        clients.remove(websocket)

def gen_frames():
    while True:
        if latest_frame is not None:
            ret, buffer = cv2.imencode('.jpg', latest_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            time.sleep(0.01)

@app.get("/video")
def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")
