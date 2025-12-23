import sqlite3
import json
from datetime import datetime
import os
from .config import DB_PATH, MAX_EVENT_COUNT, SAVE_DIR

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            type TEXT NOT NULL,
            labels TEXT,
            confidence REAL,
            image_path TEXT,
            bboxes TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_event(event_type, labels, confidence, image_path, bboxes):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO events (timestamp, type, labels, confidence, image_path, bboxes)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        event_type,
        json.dumps(labels),
        confidence,
        image_path,
        json.dumps(bboxes)
    ))
    conn.commit()
    event_id = cursor.lastrowid
    
    # Retention Policy: Keep only latest MAX_EVENT_COUNT
    cursor.execute("SELECT COUNT(*) FROM events")
    count = cursor.fetchone()[0]
    
    if count > MAX_EVENT_COUNT:
        # Find oldest events to delete
        cursor.execute("SELECT id, image_path FROM events ORDER BY timestamp ASC LIMIT ?", (count - MAX_EVENT_COUNT,))
        to_delete = cursor.fetchall()
        
        for row in to_delete:
            del_id, del_path = row
            # Delete image file
            if del_path:
                full_path = os.path.join(SAVE_DIR, del_path)
                if os.path.exists(full_path):
                    try:
                        os.remove(full_path)
                    except Exception as e:
                        print(f"Error deleting old event image {full_path}: {e}")
            
            # Delete DB row
            cursor.execute("DELETE FROM events WHERE id = ?", (del_id,))
        
        conn.commit()
        
    conn.close()
    return event_id

def get_events(limit=200):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM events ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    
    events = []
    ids_to_purge = []
    
    for row in rows:
        event = dict(row)
        if event['image_path']:
            image_full_path = os.path.join(SAVE_DIR, event['image_path'])
            if os.path.exists(image_full_path):
                events.append(event)
            else:
                ids_to_purge.append(event['id'])
        else:
            events.append(event)
            
    if ids_to_purge:
        try:
            cursor.executemany("DELETE FROM events WHERE id = ?", [(id_val,) for id_val in ids_to_purge])
            conn.commit()
            print(f"Archive Purge: Removed {len(ids_to_purge)} entries with missing source images.")
        except Exception as e:
            print(f"Archive Purge Error: {e}")
            
    conn.close()
    return events

def get_event_by_id(event_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM events WHERE id = ?", (event_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

# Initialize on import
init_db()
