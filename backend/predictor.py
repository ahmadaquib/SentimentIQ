"""
Predictor Module for SentimentIQ

Loads trained vectorizer and classification models, incorporates sarcasm detection,
calculates confidence scores, and invokes the feeling mapper.
"""

import os
import joblib
import numpy as np
from typing import Dict, Any

from backend.config import VECTORIZER_PATH, SENTIMENT_MODEL_PATH, EMOTION_MODEL_PATH, SARCASM_MODEL
from backend.logger import get_logger
from backend.preprocess import clean_text
from backend.feeling_mapper import map_feeling

logger = get_logger("Predictor")

logger.info(f"Loading vectorizer from {VECTORIZER_PATH}...")
vectorizer = joblib.load(VECTORIZER_PATH)

logger.info(f"Loading sentiment model from {SENTIMENT_MODEL_PATH}...")
sentiment_model = joblib.load(SENTIMENT_MODEL_PATH)

logger.info(f"Loading emotion model from {EMOTION_MODEL_PATH}...")
emotion_model = joblib.load(EMOTION_MODEL_PATH)

# Lazy loading for HuggingFace Sarcasm Pipeline
sarcasm_pipeline = None
_sarcasm_attempted = False


def _get_sarcasm_pipeline():
    """Lazy initializer for HuggingFace sarcasm detection pipeline."""
    global sarcasm_pipeline, _sarcasm_attempted
    if sarcasm_pipeline is None and not _sarcasm_attempted:
        _sarcasm_attempted = True
        try:
            from transformers import pipeline
            logger.info(f"Loading HuggingFace sarcasm model ({SARCASM_MODEL})...")
            sarcasm_pipeline = pipeline(
                "text-classification",
                model=SARCASM_MODEL,
                truncation=True
            )
            logger.info("Sarcasm pipeline loaded successfully!")
        except Exception as e:
            logger.warning(f"Sarcasm pipeline could not be loaded (using fallback): {e}")
            sarcasm_pipeline = None
    return sarcasm_pipeline


def detect_sarcasm(text: str) -> bool:
    """
    Detect whether input text contains sarcasm / irony.
    Falls back to keyword heuristics if transformer pipeline is unavailable.
    """
    pipe = _get_sarcasm_pipeline()
    if pipe is not None:
        try:
            res = pipe(text)
            if res and isinstance(res, list):
                first_pred = res[0]
                if isinstance(first_pred, list) and len(first_pred) > 0:
                    first_pred = first_pred[0]
                label = first_pred.get("label", "").upper()
                return label in ("LABEL_1", "SARCASM", "SARCASTIC")
        except Exception as e:
            logger.error(f"Sarcasm prediction error: {e}")

    # Fallback heuristic: classic sarcasm markers
    text_lower = text.lower()
    sarcasm_markers = ("/s", "oh great", "yeah right", "totally awesome", "just great", "best day ever", "oh sure", "oh fantastic")
    return any(marker in text_lower for marker in sarcasm_markers)


def predict_text(text: str) -> Dict[str, Any]:
    """
    Run full end-to-end prediction pipeline on a single text string.

    Parameters
    ----------
    text : str
        Input raw text.

    Returns
    -------
    dict
        Dictionary with keys: text, sentiment, emotion, feeling, sarcasm, confidence
    """
    cleaned = clean_text(text)
    if not cleaned:
        cleaned = text.lower()

    # Vectorize
    vec = vectorizer.transform([cleaned])

    # Sentiment Prediction & Confidence
    sentiment_pred = str(sentiment_model.predict(vec)[0])
    sentiment_proba = float(np.max(sentiment_model.predict_proba(vec)[0]))

    # Emotion Prediction & Confidence
    emotion_pred = str(emotion_model.predict(vec)[0])
    emotion_proba = float(np.max(emotion_model.predict_proba(vec)[0]))

    # Combined Confidence Score
    confidence = round((sentiment_proba + emotion_proba) / 2.0, 4)

    # Sarcasm Detection
    sarcasm = detect_sarcasm(text)

    # Feeling Mapping
    feeling = map_feeling(emotion_pred, sentiment_pred, text)

    return {
        "text": text,
        "sentiment": sentiment_pred,
        "emotion": emotion_pred,
        "feeling": feeling,
        "sarcasm": sarcasm,
        "confidence": confidence
    }
