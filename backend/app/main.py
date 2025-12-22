import asyncio
import cv2
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse
import os
import time

from .config import ESP_STREAM_URL, FRAME_SKIP, SAVE_DIR
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

# Detection System
detector = DetectionSystem()

# State
is_camera_connected = False
latest_frame = None
latest_detections = {
    "timestamp": "",
    "fps": 0,
    "counts": {"persons": 0, "weapons": 0},
    "threats": [],
    "boxes": [],
    "status": "DISCONNECTED",
    "debug": {"model_used": os.path.basename(detector.model.ckpt_path) if hasattr(detector.model, 'ckpt_path') and detector.model.ckpt_path else "yolov8n.pt"}
}
clients = set()

# Mount static files for events
app.mount("/images", StaticFiles(directory=SAVE_DIR), name="images")

def video_ingestion_loop():
    global latest_frame, latest_detections, is_camera_connected
    
    while True:
        cap = cv2.VideoCapture(ESP_STREAM_URL)
        if not cap.isOpened():
            print(f"Failed to connect to stream: {ESP_STREAM_URL}. Retrying in 5s...")
            is_camera_connected = False
            latest_detections["status"] = "DISCONNECTED"
            time.sleep(5)
            continue

        print(f"Connected to stream: {ESP_STREAM_URL}")
        is_camera_connected = True
        latest_detections["status"] = "CONNECTED"
        frame_count = 0
        start_time = time.time()
        
        while True:
            t1 = time.time()
            ret, frame = cap.read()
            
            if not ret:
                print("Stream lost. Reconnecting...")
                is_camera_connected = False
                latest_detections["status"] = "DISCONNECTED"
                break
                
            latest_frame = frame.copy()
            
            if frame_count % FRAME_SKIP == 0:
                boxes, persons, weapons = detector.detect(frame)
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
                    "status": "CONNECTED"
                })
                
                # Broadcast detections
                for client in list(clients):
                    try:
                        asyncio.run_coroutine_threadsafe(client.send_json(latest_detections), loop)
                    except Exception:
                        pass # Ignore send errors for specific clients
            
            frame_count += 1
            
            # FPS Clamping to ~15 FPS max to save CPU
            elapsed = time.time() - t1
            wait = max(0.01, 0.066 - elapsed) # Target ~15 FPS
            time.sleep(wait)

        cap.release()
        time.sleep(1)

# Background Thread for Video
loop = asyncio.get_event_loop()
threading.Thread(target=video_ingestion_loop, daemon=True).start()

@app.get("/health")
def health():
    return {"status": "ok", "model": latest_detections["debug"]["model_used"]}

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
        time.sleep(0.1)

@app.get("/video")
def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")
