import os
import cv2
from .config import SAVE_DIR

def save_snapshot(frame, filename):
    filepath = os.path.join(SAVE_DIR, filename)
    cv2.imwrite(filepath, frame)
    return filepath
