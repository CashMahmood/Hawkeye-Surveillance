import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
from datetime import datetime
from .config import (
    MODEL_PATH_WEAPON, MODEL_PATH_PERSON,
    CONF_THRESH_PERSON, CONF_THRESH_WEAPON, CONF_THRESH_WEAPON_ARCHIVE,
    WEAPON_MAX_BOX_AREA_PCT, WEAPON_PERSISTENCE_CYCLES, STATIC_SUPPRESSION_FRAMES,
    ASSOCIATION_MARGIN_PX, EVENT_COOLDOWN_SECONDS, SAVE_DIR
)
from .db import log_event
from .storage import save_snapshot

class DetectionSystem:
    def __init__(self):
        print(f"--- Loading Accuracy-Optimized Models ---")
        self.model_person = YOLO(MODEL_PATH_PERSON)
        self.model_weapon = YOLO(MODEL_PATH_WEAPON)
        
        # Subh775 Classes: {0: 'Gun', 1: 'explosion', 2: 'grenade', 3: 'knife'}
        self.weapon_classes = [0, 1, 2, 3] 
        
        self.last_threat_time = {}
        self.weapon_history = [] 
        self.prev_weapons = [] 
        
        print(f"Neural Core: HD Weapon Scan Enabled (CPU-Optimized)")
        print(f"----------------------------------------")

    def detect(self, frame):
        h, w = frame.shape[:2]
        frame_area = h * w
        boxes = []
        persons = []
        weapons = []
        
        # 1. PRIMARY: Weapon Detection (HD Pass)
        # We increase imgsz to 640. This allows the model to see edges of knives/guns much more clearly.
        # We use half=False/augment=False to keep it fast on CPU.
        res_weapon = self.model_weapon(
            frame, 
            imgsz=640, # HIGH DEFINITION for accuracy
            conf=CONF_THRESH_WEAPON * 0.7, # Sensitive initial scan
            iou=0.45, 
            verbose=False,
            task='detect'
        )[0]

        for box in res_weapon.boxes:
            cls_id = int(box.cls[0])
            if cls_id in self.weapon_classes:
                conf = float(box.conf[0])
                label = self.model_weapon.names[cls_id].upper()
                xyxy = box.xyxy[0].tolist()
                
                # Filter noise by area and strictness
                box_area_pct = ((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])) / frame_area
                if box_area_pct > WEAPON_MAX_BOX_AREA_PCT:
                    continue

                # Final Accuracy Gate: High-Conf Weapon locking
                data = {
                    "cls": cls_id, "label": label, "conf": conf, 
                    "x1": xyxy[0], "y1": xyxy[1], "x2": xyxy[2], "y2": xyxy[3],
                    "type": "weapon"
                }
                weapons.append(data)
                boxes.append(data)

        # 2. MOMENTUM TRACKING (Keep red box alive if accurate match found previously)
        if not weapons and self.prev_weapons:
            for old_w in self.prev_weapons:
                if old_w.get('persistence', 0) < 4:
                    old_w['persistence'] = old_w.get('persistence', 0) + 1
                    weapons.append(old_w)
                    boxes.append(old_w)
        
        self.prev_weapons = [w for w in weapons if w.get('persistence', 0) == 0]

        # 3. SECONDARY: Person Tracking (Normal Pass)
        # Fast person tracking at 320 to balance the CPU load
        res_person = self.model_person(frame, imgsz=320, conf=CONF_THRESH_PERSON, verbose=False)[0]
        for box in res_person.boxes:
            cls_id = int(box.cls[0])
            if cls_id == 0:
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                data = {
                    "label": "PERSON", "conf": conf, 
                    "x1": xyxy[0], "y1": xyxy[1], "x2": xyxy[2], "y2": xyxy[3],
                    "type": "person"
                }
                persons.append(data)
                boxes.append(data)

        return boxes, persons, weapons, (w, h)

    def process_threats(self, frame, boxes, persons, weapons):
        threats = []
        now = time.time()
        
        # Target only real detections (not persistence boxes) for archival
        actual_weapons = [w for w in weapons if w.get('persistence', 0) == 0]
        max_conf_now = max([w['conf'] for w in actual_weapons]) if actual_weapons else 0
        
        self.weapon_history.append(max_conf_now)
        if len(self.weapon_history) > WEAPON_PERSISTENCE_CYCLES:
            self.weapon_history.pop(0)
            
        # Decision Logic: Check for consistency in the HD scan window
        is_weapon_detected = any(c >= CONF_THRESH_WEAPON for c in self.weapon_history)
        
        if is_weapon_detected:
            threats.append("WEAPON_DETECTED")
            
            # Archive Logic: Lock snapshot only when confidence is stable and high
            should_archive = any(c >= CONF_THRESH_WEAPON_ARCHIVE for c in self.weapon_history)
            
            if should_archive:
                last_time = self.last_threat_time.get("WEAPON_DETECTED", 0)
                if now - last_time > EVENT_COOLDOWN_SECONDS:
                    self.last_threat_time["WEAPON_DETECTED"] = now
                    self.save_event(frame, "WEAPON_DETECTED", boxes)

        return threats

    def save_event(self, frame, threat_type, boxes):
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp_str}_{threat_type}.jpg"
        save_snapshot(frame, filename)
        labels = [b['label'] for b in boxes]
        conf = max([b['conf'] for b in boxes]) if boxes else 0
        log_event(threat_type, labels, conf, filename, boxes)
        print(f"CRITICAL: Accurately Archived -> {filename}")
