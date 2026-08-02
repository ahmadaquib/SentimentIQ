"""
CSV Logger Module for SentimentIQ

Handles persistent logging of predictions to `data/predictions.csv`,
retrieval of historical logs, calculation of summary analytics, and clearing history.
"""

import os
import csv
from datetime import datetime
from typing import List, Dict, Any

from backend.config import DATA_DIR, PREDICTIONS_CSV
from backend.logger import get_logger

logger = get_logger("CSVLogger")

FIELDNAMES = ["id", "timestamp", "text", "sentiment", "emotion", "feeling", "sarcasm", "confidence"]


def _ensure_csv_exists():
    """Ensure data directory and predictions.csv file exist with valid headers."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(PREDICTIONS_CSV) or os.path.getsize(PREDICTIONS_CSV) == 0:
        with open(PREDICTIONS_CSV, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()


def _get_next_id() -> int:
    """Read CSV to determine the next auto-incrementing integer ID."""
    _ensure_csv_exists()
    max_id = 0
    try:
        with open(PREDICTIONS_CSV, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("id"):
                    try:
                        row_id = int(row["id"])
                        if row_id > max_id:
                            max_id = row_id
                    except ValueError:
                        pass
    except Exception as e:
        logger.error(f"Error calculating next ID: {e}")
    return max_id + 1


def log_prediction(prediction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Log a prediction record to CSV with auto-generated ID and ISO timestamp.

    Parameters
    ----------
    prediction : dict
        Dict containing text, sentiment, emotion, feeling, sarcasm, confidence

    Returns
    -------
    dict
        Updated prediction dict containing id and timestamp
    """
    _ensure_csv_exists()
    pred_id = _get_next_id()
    timestamp = datetime.now().isoformat()

    record = {
        "id": pred_id,
        "timestamp": timestamp,
        "text": prediction.get("text", ""),
        "sentiment": prediction.get("sentiment", ""),
        "emotion": prediction.get("emotion", ""),
        "feeling": prediction.get("feeling", ""),
        "sarcasm": str(prediction.get("sarcasm", False)),
        "confidence": prediction.get("confidence", 0.0)
    }

    try:
        with open(PREDICTIONS_CSV, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writerow(record)
        logger.info(f"Logged prediction #{pred_id} for text: '{prediction.get('text', '')[:30]}...'")
    except Exception as e:
        logger.error(f"Error writing prediction to CSV: {e}")

    out = dict(prediction)
    out["id"] = pred_id
    out["timestamp"] = timestamp
    return out


def get_prediction_history(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Retrieve historical predictions from CSV file.

    Parameters
    ----------
    limit : int, optional
        Maximum number of records to return (most recent first).

    Returns
    -------
    list[dict]
        List of prediction records.
    """
    _ensure_csv_exists()
    records = []
    try:
        with open(PREDICTIONS_CSV, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row.get("id"):
                    continue
                try:
                    records.append({
                        "id": int(row["id"]),
                        "timestamp": row["timestamp"],
                        "text": row["text"],
                        "sentiment": row["sentiment"],
                        "emotion": row["emotion"],
                        "feeling": row["feeling"],
                        "sarcasm": row["sarcasm"].lower() == "true",
                        "confidence": float(row["confidence"])
                    })
                except (ValueError, KeyError):
                    continue
    except Exception as e:
        logger.error(f"Error reading prediction history: {e}")

    records.sort(key=lambda x: x["id"], reverse=True)
    return records[:limit]


def get_analytics_summary() -> Dict[str, Any]:
    """
    Compute summary analytics over all logged predictions.

    Returns
    -------
    dict
        Aggregated statistics including total count, average confidence,
        sentiment distribution, emotion distribution, feeling distribution, and sarcasm count.
    """
    history = get_prediction_history(limit=10000)
    total_count = len(history)

    if total_count == 0:
        return {
            "total_count": 0,
            "avg_confidence": 0.0,
            "sentiment_distribution": {},
            "emotion_distribution": {},
            "feeling_distribution": {},
            "sarcasm_count": 0
        }

    confidences = [r["confidence"] for r in history]
    avg_confidence = round(sum(confidences) / total_count, 4)

    sentiment_counts: Dict[str, int] = {}
    emotion_counts: Dict[str, int] = {}
    feeling_counts: Dict[str, int] = {}
    sarcasm_count = 0

    for r in history:
        s = r["sentiment"]
        e = r["emotion"]
        f = r["feeling"]
        if r["sarcasm"]:
            sarcasm_count += 1

        sentiment_counts[s] = sentiment_counts.get(s, 0) + 1
        emotion_counts[e] = emotion_counts.get(e, 0) + 1
        feeling_counts[f] = feeling_counts.get(f, 0) + 1

    return {
        "total_count": total_count,
        "avg_confidence": avg_confidence,
        "sentiment_distribution": sentiment_counts,
        "emotion_distribution": emotion_counts,
        "feeling_distribution": feeling_counts,
        "sarcasm_count": sarcasm_count
    }


def clear_prediction_history() -> bool:
    """Clear all records from predictions.csv while preserving header row."""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(PREDICTIONS_CSV, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
        logger.info("Cleared all prediction logs from CSV.")
        return True
    except Exception as e:
        logger.error(f"Error clearing prediction history: {e}")
        return False
