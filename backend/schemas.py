"""
Pydantic Schemas for SentimentIQ API
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text string to analyze", example="I am so excited for this amazing project!")


class BatchTextRequest(BaseModel):
    texts: List[str] = Field(..., description="List of text strings for batch analysis")


class PredictionResponse(BaseModel):
    id: Optional[int] = Field(None, description="Prediction unique ID")
    timestamp: Optional[str] = Field(None, description="ISO timestamp of prediction")
    text: str = Field(..., description="Original input text")
    sentiment: str = Field(..., description="Predicted sentiment (Negative, Neutral, Positive)")
    emotion: str = Field(..., description="Predicted emotion (Sadness, Joy, Love, Anger, Fear, Surprise)")
    feeling: str = Field(..., description="Synthesized feeling (Excited, Grateful, Frustrated, Bitter, Anxious, Shocked, Peaceful)")
    sarcasm: bool = Field(..., description="Sarcasm / irony detected boolean flag")
    confidence: float = Field(..., description="Prediction confidence score between 0.0 and 1.0")


class HealthResponse(BaseModel):
    status: str = Field("ok", description="API health status")
    models_loaded: bool = Field(True, description="Flag indicating if models are loaded")
