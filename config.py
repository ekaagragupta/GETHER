
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
INTERIM_DATA_DIR = DATA_DIR / 'interim'
MODELS_DIR = PROJECT_ROOT / 'models'
LOGS_DIR = PROJECT_ROOT / 'logs'

# Expected columns in raw data
REQUIRED_COLUMNS = [
    'timestamp',
    'CO2',
    'NO2',
    'SO2',
    'temperature',
    'humidity',
    'wind_speed',
    'wind_direction'
]

# Feature ranges (for validation)
FEATURE_RANGES = {
    'CO2': (50, 500),          # ppm
    'NO2': (0, 200),            # ppb
    'SO2': (0, 100),            # ppb
    'temperature': (-10, 50),   # Celsius
    'humidity': (0, 100),       # percentage
    'wind_speed': (0, 50),      # km/h
    'wind_direction': (0, 360)  # degrees
}