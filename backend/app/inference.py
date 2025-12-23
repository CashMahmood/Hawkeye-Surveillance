import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
from datetime import datetime
from .config import (
    MODEL_PATH_PRIMARY, MODEL_PATH_BACKUP, MODEL_PATH_FALLBACK,
    CONF_THRESH_PERSON, CONF_THRESH_WEAPON, CONF_THRESH_WEAPON_ARCHIVE,
    WEAPON_MAX_AREA_PCT, WEAPON_MAX_BOX_AREA_PCT, WEAPON_PERSISTENCE_CYCLES, STATIC_SUPPRESSION_FRAMES,
    GROUP_MIN_COUNT, GROUP_DISTANCE_PX, ASSOCIATION_MARGIN_PX, GROUP_TIME_SECONDS,
    EVENT_COOLDOWN_SECONDS, SAVE_DIR
)
from .db import log_event
from .storage import save_snapshot

class DetectionSystem:
    def __init__(self):
        # Initialize Dual Neural Pipeline
        self.model_weapons, self.model_gen = self._load_models()
        self.gen_class_names = self.model_gen.names
        self.weapon_class_names = self.model_weapons.names
        
        self.last_threat_time = {}
        self.cluster_active_since = None
        self.weapon_history = [] # For voting persistence
        self.static_weapon_counts = {} # {bbox_key: frame_count} for static suppression
        
        # Tactical Whitelists
        self.weapon_keywords = [
            'gun', 'pistol', 'rifle', 'handgun', 'firearm', 
            'knife', 'dagger', 'machete', 'sword', 
            'bat', 'baseball bat', 'hockey stick', 'stick', 'weapon', 'club', 'spear'
        ]
        
        # Map IDs across models
        self.weapon_ids_gen = [id for id, name in self.gen_class_names.items() if any(k in name.lower() for k in self.weapon_keywords)]
        self.person_ids_gen = [id for id, name in self.gen_class_names.items() if 'person' in name.lower()]
        
        self.weapon_ids_spec = [id for id, name in self.weapon_class_names.items() if any(k in name.lower() for k in self.weapon_keywords)]
        self.person_ids_spec = [id for id, name in self.weapon_class_names.items() if 'person' in name.lower()]

        print(f"--- Hybrid Neural Architecture Online ---")
        print(f"Model Gen: {os.path.basename(self.model_gen.ckpt_path) if hasattr(self.model_gen, 'ckpt_path') else 'yolov8m'}")
        print(f"Model Spec: {os.path.basename(self.model_weapons.ckpt_path) if hasattr(self.model_weapons, 'ckpt_path') else 'Custom Spec'}")
        print(f"Sensors: General({len(self.person_ids_gen)}P/{len(self.weapon_ids_gen)}W) | Spec({len(self.person_ids_spec)}P/{len(self.weapon_ids_spec)}W)")
        print(f"----------------------------------------")

    def _load_models(self):
        """Loads two models: one for general persons/objects, one for specialized weapons."""
        # 1. Load General Intelligence (YOLOv8m)
        try:
            m_gen = YOLO("yolov8m.pt")
        except:
            m_gen = YOLO(MODEL_PATH_FALLBACK)
            
        # 2. Load Specialized Weapon Model
        if os.path.exists(MODEL_PATH_PRIMARY):
            m_spec = YOLO(MODEL_PATH_PRIMARY)
        elif os.path.exists(MODEL_PATH_BACKUP):
            m_spec = YOLO(MODEL_PATH_BACKUP)
        else:
            m_spec = m_gen # Fallback to same if unique model is missing
            
        return m_spec, m_gen

    def _calculate_iou(self, box1, box2):
        """Helper for Intersection over Union to prevent double-counting."""
        x1 = max(box1['x1'], box2['x1'])
        y1 = max(box1['y1'], box2['y1'])
        x2 = min(box1['x2'], box2['x2'])
        y2 = min(box1['y2'], box2['y2'])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
        area2 = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])
        
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0

    def _is_near(self, weapon, person):
        """Refined association logic: Check if weapon is within proximity of the person's box."""
        # Person box with expanded margin
        px1, py1 = person['x1'] - ASSOCIATION_MARGIN_PX, person['y1'] - ASSOCIATION_MARGIN_PX
        px2, py2 = person['x2'] + ASSOCIATION_MARGIN_PX, person['y2'] + ASSOCIATION_MARGIN_PX
        
        # Weapon center
        wcx = (weapon['x1'] + weapon['x2']) / 2
        wcy = (weapon['y1'] + weapon['y2']) / 2
        
        return px1 <= wcx <= px2 and py1 <= wcy <= py2

    def detect(self, frame):
        h, w = frame.shape[:2]
        frame_area = h * w
        boxes = []
        persons = []
        weapons = []
        
        # Parallel Multi-Head Inference
        res_spec = self.model_weapons(frame, imgsz=640, conf=min(CONF_THRESH_PERSON, CONF_THRESH_WEAPON), verbose=False)[0]
        res_gen = self.model_gen(frame, imgsz=640, conf=min(CONF_THRESH_PERSON, CONF_THRESH_WEAPON), verbose=False)[0]
        
        # Process Specialized Model (Primary for Weapons)
        for box in res_spec.boxes:
            cls_id = int(box.cls[0])
            label = self.weapon_class_names[cls_id]
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            
            is_p = cls_id in self.person_ids_spec and conf >= CONF_THRESH_PERSON
            is_w = cls_id in self.weapon_ids_spec and conf >= CONF_THRESH_WEAPON
            
            # Weapon Area Filter: Ignore huge false positives (furniture/walls)
            if is_w:
                box_area_pct = ((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])) / (frame_area)
                if box_area_pct > WEAPON_MAX_BOX_AREA_PCT:
                    is_w = False # Suppress oversized boxes
            
            if is_p or is_w:
                # Add tactical diagnostic metadata
                diag_label = f"[{label.upper()}/SPEC] {conf:.2f}"
                data = {"cls": cls_id, "label": diag_label, "conf": conf, "x1": xyxy[0], "y1": xyxy[1], "x2": xyxy[2], "y2": xyxy[3], "source": "spec"}
                if is_p: persons.append(data)
                if is_w: weapons.append(data)
                boxes.append(data)

        # Process General Model (Secondary for Weapons, Primary for Persons/Silhouettes/Sticks)
        for box in res_gen.boxes:
            cls_id = int(box.cls[0])
            label = self.gen_class_names[cls_id]
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            
            is_p = cls_id in self.person_ids_gen and conf >= CONF_THRESH_PERSON
            is_w = cls_id in self.weapon_ids_gen and conf >= CONF_THRESH_WEAPON
            
            if is_w:
                box_area_pct = ((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])) / (frame_area)
                if box_area_pct > WEAPON_MAX_BOX_AREA_PCT:
                    is_w = False
            
            if is_p or is_w:
                diag_label = f"[{label.upper()}/GEN] {conf:.2f}"
                data = {"cls": cls_id, "label": diag_label, "conf": conf, "x1": xyxy[0], "y1": xyxy[1], "x2": xyxy[2], "y2": xyxy[3], "source": "gen"}
                
                # IoU Deduplication
                # Check if this object is already detected by the Spec model
                is_duplicate = False
                for existing in boxes:
                    if self._calculate_iou(data, existing) > 0.45: # Standard IoU threshold
                        is_duplicate = True
                        # If the Gen model has higher confidence, update the box (optional, but keeps highest score)
                        if data['conf'] > existing['conf']:
                            existing.update(data)
                        break
                
                if not is_duplicate:
                    if is_p: persons.append(data)
                    if is_w: weapons.append(data)
                    boxes.append(data)
                
        return boxes, persons, weapons, (w, h)

    def process_threats(self, frame, boxes, persons, weapons):
        threats = []
        now = time.time()
        
        # 1. STATIC SUPPRESSION LOGIC (Identifying Furniture/Closets)
        active_weapon_keys = []
        filtered_weapons = []
        
        for w in weapons:
            # Create a spatial key (rounded to ignore tiny jitters)
            box_key = (round(w['x1'], -1), round(w['y1'], -1), round(w['x2'], -1), round(w['y2'], -1))
            active_weapon_keys.append(box_key)
            
            # Increment static count
            self.static_weapon_counts[box_key] = self.static_weapon_counts.get(box_key, 0) + 1
            
            # Check if anyone is near this weapon
            is_being_handled = any(self._is_near(w, p) for p in persons)
            
            # Suppression: If static for long but NO person is near, it's noise
            is_static_noise = (self.static_weapon_counts[box_key] > STATIC_SUPPRESSION_FRAMES) and not is_being_handled
            
            if not is_static_noise:
                filtered_weapons.append(w)
            else:
                # Completely remove from the HUD/Archive boxes list
                if w in boxes:
                    boxes.remove(w)

        # Cleanup static counts for objects no longer present
        all_keys = list(self.static_weapon_counts.keys())
        for k in all_keys:
            if k not in active_weapon_keys:
                # If it's gone for 10 frames, reset (prevents noise from being permanently marked)
                self.static_weapon_counts[k] -= 2 # Fade out
                if self.static_weapon_counts[k] <= 0:
                    del self.static_weapon_counts[k]

        weapons = filtered_weapons # Use suppressed list for alerting
        
        # 1. WEAPON_DETECTED (Voting Persistence + Archival Locking)
        # Store max confidence for the window
        max_conf_now = max([w['conf'] for w in weapons]) if weapons else 0
        self.weapon_history.append(max_conf_now)
        
        if len(self.weapon_history) > WEAPON_PERSISTENCE_CYCLES:
            self.weapon_history.pop(0)
            
        # HUD Trigger (Flicker resistant)
        is_weapon_seen_enough = sum(1 for c in self.weapon_history if c >= CONF_THRESH_WEAPON) >= 2
        
        # Archiving Trigger (Requires at least one locked detection in window)
        is_weapon_locked = any(c >= CONF_THRESH_WEAPON_ARCHIVE for c in self.weapon_history)
        
        if is_weapon_seen_enough:
            # We save the event ONLY if it was "locked" at some point in the cycle
            if is_weapon_locked:
                threats.append("WEAPON_DETECTED")
            else:
                # Still show on HUD for awareness, but don't archive as a confirmed threat yet
                threats.append("WEAPON_DETECTED_UNLOCKED") # Custom internal type for HUD-only
            
        # 2. PERSON_WITH_WEAPON (Box Proximity Association)
        person_with_weapon = False
        for w in weapons:
            for p in persons:
                if self._is_near(w, p):
                    person_with_weapon = True
                    break
        
        if person_with_weapon and is_weapon_locked:
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
        
        for t in threats:
            # HUD-only threats (unlocked) are never saved to database
            if t.endswith("_UNLOCKED"):
                continue

            last_time = self.last_threat_time.get(t, 0)
            if now - last_time > EVENT_COOLDOWN_SECONDS:
                self.last_threat_time[t] = now
                save_required = True
                primary_threat = t # Log the first significant threat found
                
        if save_required:
            self.save_event(frame, primary_threat, boxes, weapons)
                
        if threats:
            print(f"THREAT ANALYSIS: Detected {len(persons)} persons and {len(weapons)} weapons.")
            print(f"ACTIVE THREATS: {threats}")
            # Diagnostic: Show why a weapon might be suppressed
            for w in weapons:
                w_area = ((w['x2'] - w['x1']) * (w['y2'] - w['y1'])) / (frame.shape[0] * frame.shape[1])
                print(f"  > [TARGET] {w['label']} Area:{w_area:.1%} Conf:{w['conf']:.2f}")
        elif len(persons) > 0:
            print(f"SURVEILLANCE: {len(persons)} contacts in sector.")
        elif len(weapons) > 0:
            print(f"SUPPRESSION: {len(weapons)} environmental signals filtered (Static/Oversized).")
            
        return threats

    def save_event(self, frame, threat_type, boxes, weapons):
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp_str}_{threat_type}.jpg"
        
        # Ensure only 100 images requirement: log_event handles the retention
        save_snapshot(frame, filename)
        
        labels = [b['label'] for b in boxes]
        conf = max([b['conf'] for b in boxes]) if boxes else 0
        
        log_event(threat_type, labels, conf, filename, boxes)
        print(f"Event Captured: {threat_type} -> {filename}")
