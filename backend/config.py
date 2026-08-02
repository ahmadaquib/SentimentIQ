"""
Configuration & Environment Settings for SentimentIQ

Loads environment variables from .env file and provides central configuration defaults.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")


def load_env_file():
    """Simple parser to load .env file variables into os.environ if present."""
    if os.path.exists(ENV_PATH):
        try:
            with open(ENV_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key and key not in os.environ:
                        os.environ[key] = val
        except Exception as e:
            print(f"[Config] Warning loading .env file: {e}")


# Initialize env loading
load_env_file()

# Configuration variables
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "8501"))

HF_TOKEN = os.getenv("HF_TOKEN", "")
SARCASM_MODEL = os.getenv("SARCASM_MODEL", "helinivan/english-sarcasm-detector")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

MODELS_DIR = os.path.join(BASE_DIR, os.getenv("MODELS_DIR", "models"))
DATA_DIR = os.path.join(BASE_DIR, os.getenv("DATA_DIR", "data"))
PREDICTIONS_CSV = os.path.join(BASE_DIR, os.getenv("PREDICTIONS_CSV", "data/predictions.csv"))

# Specific Model File Paths
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")
SENTIMENT_MODEL_PATH = os.path.join(MODELS_DIR, "sentiment_model.pkl")
EMOTION_MODEL_PATH = os.path.join(MODELS_DIR, "emotion_model.pkl")

# Export token to environment if provided
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
