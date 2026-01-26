import os

# ESP32-CAM Configuration (Production Verified)
# Verified IP from user: 172.20.10.2
ESP_STREAM_URL = "http://172.20.10.2:81/stream"

# Storage Configuration (Needed early for BASE_DIR)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAVE_DIR = os.path.join(DATA_DIR, "events")
DB_PATH = os.path.join(DATA_DIR, "events.db")

# Model Paths (NEW TASK: Subh775 Weapon Detection)
# Using BASE_DIR to ensure absolute paths regardless of CWD
MODEL_PATH_WEAPON = os.path.join(BASE_DIR, "..", "models", "subh775_threat.pt")
MODEL_PATH_PERSON = os.path.join(BASE_DIR, "yolov8n.pt") 

# YOLO Configuration
CONF_THRESH_PERSON = 0.40  
CONF_THRESH_WEAPON = 0.45  
CONF_THRESH_WEAPON_ARCHIVE = 0.50 

WEAPON_MAX_BOX_AREA_PCT = 0.30 
STATIC_SUPPRESSION_FRAMES = 30 
INFERENCE_IMGSZ = 640      
FRAME_SKIP = 1 # Higher precision

# Archive Retention
MAX_EVENT_COUNT = 100      

# Weapon Gating
WEAPON_MAX_AREA_PCT = 0.50  
WEAPON_PERSISTENCE_CYCLES = 5  
WEAPON_VOTE_THRESHOLD = 2    

# Threat Logic Configuration
GROUP_MIN_COUNT = 4
GROUP_DISTANCE_PX = 120
ASSOCIATION_MARGIN_PX = 100 
EVENT_COOLDOWN_SECONDS = 5

# Ensure directories exist
os.makedirs(SAVE_DIR, exist_ok=True)
