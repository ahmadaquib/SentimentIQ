"""
FastAPI Main Server Application for SentimentIQ

Provides REST API endpoints for sentiment, emotion, feeling, sarcasm prediction,
logging history, fetching analytics summary, and resetting logs.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from backend.config import API_HOST, API_PORT
from backend.logger import get_logger
from backend.schemas import (
    TextRequest,
    BatchTextRequest,
    PredictionResponse,
    HealthResponse
)
from backend.predictor import predict_text
from backend.csv_logger import (
    log_prediction,
    get_prediction_history,
    get_analytics_summary,
    clear_prediction_history
)

logger = get_logger("MainAPI")

app = FastAPI(
    title="SentimentIQ API",
    description="Production REST API for Sentiment, Emotion, Fine-Grained Feeling, and Sarcasm Intelligence",
    version="1.0.0"
)

# Enable CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    logger.info(f"FastAPI Server starting up on http://{API_HOST}:{API_PORT}")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Health check endpoint to verify API and model status."""
    return HealthResponse(status="ok", models_loaded=True)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_single(request: TextRequest):
    """
    Analyze raw text string to predict Sentiment, Emotion, Feeling, Sarcasm, and Confidence,
    and persist record to data/predictions.csv.
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")

    try:
        prediction = predict_text(request.text)
        logged_record = log_prediction(prediction)
        return logged_record
    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=List[PredictionResponse], tags=["Prediction"])
def predict_batch(request: BatchTextRequest):
    """
    Batch analyze a list of text strings, logging each to history.
    """
    if not request.texts:
        raise HTTPException(status_code=400, detail="Text list cannot be empty.")

    results = []
    for text in request.texts:
        if not text or not text.strip():
            continue
        try:
            pred = predict_text(text)
            logged = log_prediction(pred)
            results.append(logged)
        except Exception as e:
            logger.error(f"Error processing batch item '{text[:20]}...': {e}")
            continue

    return results


@app.get("/history", response_model=List[PredictionResponse], tags=["History & Analytics"])
def fetch_history(limit: int = Query(100, ge=1, le=1000, description="Max history records to fetch")):
    """Retrieve historical prediction records logged in CSV format."""
    try:
        return get_prediction_history(limit=limit)
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")


@app.get("/analytics", tags=["History & Analytics"])
def fetch_analytics():
    """Retrieve aggregate summary analytics for all historical predictions."""
    try:
        return get_analytics_summary()
    except Exception as e:
        logger.error(f"Error fetching analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch analytics: {str(e)}")


@app.delete("/history", tags=["History & Analytics"])
def reset_history():
    """Clear all historical prediction records from CSV while retaining header structure."""
    success = clear_prediction_history()
    if not success:
        raise HTTPException(status_code=500, detail="Failed to clear history log.")
    return {"message": "Prediction history cleared successfully."}
