import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
from datetime import datetime
from .config import (
    MODEL_PATH_PRIMARY, MODEL_PATH_BACKUP, MODEL_PATH_FALLBACK,
    CONF_THRESH, GROUP_MIN_COUNT, GROUP_DISTANCE_PX, GROUP_TIME_SECONDS,
    EVENT_COOLDOWN_SECONDS, SAVE_DIR
)
from .db import log_event
from .storage import save_snapshot

class DetectionSystem:
    def __init__(self):
        self.model = self._load_model()
        self.class_names = self.model.names
        self.last_threat_time = {}
        self.cluster_active_since = None
        
        # Define weapon keywords to match labels from the model
        self.weapon_keywords = ['gun', 'pistol', 'rifle', 'handgun', 'knife', 'stick', 'bat', 'weapon', 'firearm', 'arm']
        self.weapon_class_ids = [
            id for id, name in self.class_names.items() 
            if any(k in name.lower() for k in self.weapon_keywords)
        ]
        self.person_class_ids = [
            id for id, name in self.class_names.items() 
            if 'person' in name.lower()
        ]
        
        print(f"Model loaded with {len(self.class_names)} classes.")
        print(f"Detected person classes: {[self.class_names[i] for i in self.person_class_ids]}")
        print(f"Detected weapon classes: {[self.class_names[i] for i in self.weapon_class_ids]}")

    def _load_model(self):
        if os.path.exists(MODEL_PATH_PRIMARY):
            print(f"Loading primary model: {MODEL_PATH_PRIMARY}")
            return YOLO(MODEL_PATH_PRIMARY)
        elif os.path.exists(MODEL_PATH_BACKUP):
            print(f"Loading backup model: {MODEL_PATH_BACKUP}")
            return YOLO(MODEL_PATH_BACKUP)
        else:
            print(f"Fallback to COCO model: {MODEL_PATH_FALLBACK}")
            return YOLO(MODEL_PATH_FALLBACK)

    def detect(self, frame):
        # Run inference
        results = self.model(frame, conf=CONF_THRESH, verbose=False)[0]
        boxes = []
        persons = []
        weapons = []
        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = self.class_names[cls_id]
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            
            box_data = {
                "cls": cls_id,
                "label": label,
                "conf": conf,
                "x1": xyxy[0],
                "y1": xyxy[1],
                "x2": xyxy[2],
                "y2": xyxy[3]
            }
            boxes.append(box_data)
            
            if cls_id in self.person_class_ids:
                persons.append(box_data)
            elif cls_id in self.weapon_class_ids:
                weapons.append(box_data)
                
        return boxes, persons, weapons

    def process_threats(self, frame, boxes, persons, weapons):
        threats = []
        now = time.time()
        
        # 1. WEAPON_DETECTED
        if weapons:
            threats.append("WEAPON_DETECTED")
            
        # 2. PERSON_WITH_WEAPON (Association)
        person_with_weapon = False
        for w in weapons:
            w_center = np.array([(w['x1'] + w['x2']) / 2, (w['y1'] + w['y2']) / 2])
            for p in persons:
                p_center = np.array([(p['x1'] + p['x2']) / 2, (p['y1'] + p['y2']) / 2])
                dist = np.linalg.norm(w_center - p_center)
                # Check if weapon center is within or near person box (using distance or intersection)
                if dist < 80: # Within 80 pixels of center
                    person_with_weapon = True
                    break
        
        if person_with_weapon:
            threats.append("PERSON_WITH_WEAPON")
            
        # 3. SUSPICIOUS_GROUP (Rule-based)
        if len(persons) >= GROUP_MIN_COUNT:
            # Clustering check using centroid
            person_centers = [np.array([(p['x1'] + p['x2']) / 2, (p['y1'] + p['y2']) / 2]) for p in persons]
            centroid = np.mean(person_centers, axis=0)
            
            clustered_count = sum(1 for c in person_centers if np.linalg.norm(c - centroid) < GROUP_DISTANCE_PX)
            
            if clustered_count >= GROUP_MIN_COUNT:
                if self.cluster_active_since is None:
                    self.cluster_active_since = now
                
                if now - self.cluster_active_since >= GROUP_TIME_SECONDS:
                    threats.append("SUSPICIOUS_GROUP")
            else:
                self.cluster_active_since = None
        else:
            self.cluster_active_since = None
                
        # Deduplication and Persisting
        save_required = False
        primary_threat = None
        
        for threat in threats:
            last_time = self.last_threat_time.get(threat, 0)
            if now - last_time > EVENT_COOLDOWN_SECONDS:
                self.last_threat_time[threat] = now
                save_required = True
                primary_threat = threat # Log the first significant threat found
                
        if save_required:
            self.save_event(frame, primary_threat, boxes, weapons)
                
        return threats

    def save_event(self, frame, threat_type, boxes, weapons):
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp_str}_{threat_type}.jpg"
        
        save_snapshot(frame, filename)
        
        labels = [b['label'] for b in boxes]
        conf = max([b['conf'] for b in boxes]) if boxes else 0
        
        log_event(threat_type, labels, conf, filename, boxes)
        print(f"Event Captured: {threat_type} -> {filename}")
