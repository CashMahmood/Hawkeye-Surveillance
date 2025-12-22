import sqlite3
import json
from datetime import datetime
from .config import DB_PATH

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
    conn.close()
    return event_id

def get_events(limit=200):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM events ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    events = [dict(row) for row in rows]
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
