import os

# ESP32-CAM Configuration
ESP_STREAM_URL = os.getenv("ESP_STREAM_URL", "http://192.168.1.100:81/stream")

# Model Paths
MODEL_PATH_PRIMARY = os.path.abspath("./models/threat_yolov8n/weights/best.pt")
MODEL_PATH_BACKUP = os.path.abspath("./models/firearm_yolov8n/weights/best.pt")
MODEL_PATH_FALLBACK = "yolov8n.pt"

# YOLO Configuration
CONF_THRESH = float(os.getenv("CONF_THRESH", 0.35))
FRAME_SKIP = int(os.getenv("FRAME_SKIP", 2))

# Threat Logic Configuration
GROUP_MIN_COUNT = int(os.getenv("GROUP_MIN_COUNT", 4))
GROUP_DISTANCE_PX = int(os.getenv("GROUP_DISTANCE_PX", 120))
GROUP_TIME_SECONDS = int(os.getenv("GROUP_TIME_SECONDS", 4))
EVENT_COOLDOWN_SECONDS = int(os.getenv("EVENT_COOLDOWN_SECONDS", 3))

# Storage Configuration
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAVE_DIR = os.path.join(DATA_DIR, "events")
DB_PATH = os.path.join(DATA_DIR, "events.db")

# Ensure directories exist
os.makedirs(SAVE_DIR, exist_ok=True)
