import os

# ESP32-CAM Configuration (Production Verified)
ESP_STREAM_URL = "http://172.20.10.2:81/stream"

# Model Paths
MODEL_PATH_PRIMARY = os.path.abspath("./models/threat_yolov8n/weights/best.pt")
MODEL_PATH_BACKUP = os.path.abspath("./models/firearm_yolov8n/weights/best.pt")
MODEL_PATH_FALLBACK = "yolov8m.pt"

# YOLO Configuration
CONF_THRESH_PERSON = 0.20  # Lowered to 0.20 for tactical drone tracking
CONF_THRESH_WEAPON = 0.40  # Restored to 0.40 for better dummy gun/knife tracking
CONF_THRESH_WEAPON_ARCHIVE = 0.70 # Lowered to 0.70 to ensure incidents are captured
WEAPON_MAX_BOX_AREA_PCT = 0.25 # Ignore boxes > 25% of frame (eliminates furniture false positives)
STATIC_SUPPRESSION_FRAMES = 50 # Frames a weapon must move to be considered dynamic
INFERENCE_IMGSZ = 640      # Higher res for distant objects
FRAME_SKIP = 2

# Archive Retention
MAX_EVENT_COUNT = 100      # Keep only latest 100 images

# Weapon Gating
WEAPON_MAX_AREA_PCT = 0.40  # Ignore if >40% of frame
WEAPON_PERSISTENCE_CYCLES = 5  # Tracking history window (voting logic applied in inference.py)

# Threat Logic Configuration
GROUP_MIN_COUNT = 4
GROUP_DISTANCE_PX = 120
ASSOCIATION_MARGIN_PX = 80 # Max pixel distance from box edge for person-weapon pairing
GROUP_TIME_SECONDS = 2     # Reduced for faster tactical response
EVENT_COOLDOWN_SECONDS = 3

# Storage Configuration
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAVE_DIR = os.path.join(DATA_DIR, "events")
DB_PATH = os.path.join(DATA_DIR, "events.db")

# Ensure directories exist
os.makedirs(SAVE_DIR, exist_ok=True)
