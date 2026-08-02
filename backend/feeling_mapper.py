"""
Feeling Mapper Module for SentimentIQ

This module provides a rule-based mapping function that synthesizes predicted sentiment
and emotion (plus optional text context) into one of 7 fine-grained human feelings:
- Excited
- Grateful
- Frustrated
- Bitter
- Anxious
- Shocked
- Peaceful
"""

import re
from typing import Optional, Tuple


def _has_keyword(text: str, keywords: Tuple[str, ...]) -> bool:
    """
    Check if any keyword appears in text using word-boundary matching.

    Multi-word keywords (e.g. "can't wait", "no way") use simple substring matching.
    Single-word keywords use regex word boundaries to avoid false substring matches
    (e.g. "lie" must not match inside "unbelievable").

    Parameters
    ----------
    text : str
        Lowercased input text.
    keywords : tuple of str
        Keywords to search for.

    Returns
    -------
    bool
        True if any keyword is found in the text.
    """
    for kw in keywords:
        if " " in kw:
            # Multi-word phrase: use simple substring match
            if kw in text:
                return True
        else:
            # Single word: use word-boundary regex to prevent substring false positives
            if re.search(r'\b' + re.escape(kw) + r'\b', text):
                return True
    return False


def map_feeling(emotion: str, sentiment: str, text: Optional[str] = "") -> str:
    """
    Synthesize sentiment and emotion classification predictions into a nuanced feeling category.

    Parameters
    ----------
    emotion : str
        Predicted emotion category (e.g., 'Joy', 'Sadness', 'Anger', 'Fear', 'Surprise', 'Love').
    sentiment : str
        Predicted sentiment class (e.g., 'Positive', 'Neutral', 'Negative').
    text : str, optional
        Original or cleaned text context for keyword-level micro-rules.

    Returns
    -------
    str
        One of 7 target feeling labels:
        - 'Excited'
        - 'Grateful'
        - 'Frustrated'
        - 'Bitter'
        - 'Anxious'
        - 'Shocked'
        - 'Peaceful'
    """
    emotion_lower = (emotion or "").strip().lower()
    sentiment_lower = (sentiment or "").strip().lower()
    text_lower = (text or "").lower()

    # Keyword check helpers
    gratitude_kw = ("thank", "thanks", "grateful", "blessed", "appreciate", "kindness")
    excited_kw = ("wow", "amazing", "thrilled", "can't wait", "hyped", "awesome", "yay", "super", "omg")
    frustrated_kw = ("annoyed", "stuck", "again", "slow", "cancel", "useless", "fail", "broken", "issue", "bug")
    bitter_kw = ("betrayed", "unfair", "cheated", "liar", "lie", "lied", "lying", "hate", "jealous", "ruined", "disgusting")
    anxious_kw = ("worried", "scared", "nervous", "panic", "afraid", "dread", "stress", "tense")
    shocked_kw = ("unbelievable", "wtf", "no way", "shocked", "stunned", "suddenly", "unexpected")
    peaceful_kw = ("calm", "relax", "quiet", "peace", "peaceful", "fine", "content", "tranquil", "serene")

    # High priority keyword overrides (using word-boundary matching)
    if _has_keyword(text_lower, gratitude_kw):
        return "Grateful"

    if _has_keyword(text_lower, excited_kw) and sentiment_lower in ("positive", "neutral"):
        return "Excited"

    # Check shocked before bitter to avoid substring collisions (e.g. "lie" inside "unbelievable")
    if _has_keyword(text_lower, shocked_kw):
        return "Shocked"

    if _has_keyword(text_lower, bitter_kw):
        return "Bitter"

    if _has_keyword(text_lower, anxious_kw):
        return "Anxious"

    if _has_keyword(text_lower, frustrated_kw):
        return "Frustrated"

    if _has_keyword(text_lower, peaceful_kw):
        return "Peaceful"

    # Core Emotion + Sentiment Rules
    # 1. Joy & Love
    if emotion_lower in ("joy", "love"):
        if sentiment_lower == "positive":
            return "Excited"
        elif sentiment_lower == "neutral":
            return "Peaceful"
        else:
            return "Grateful"

    # 2. Surprise
    elif emotion_lower == "surprise":
        if sentiment_lower == "positive":
            return "Excited"
        elif sentiment_lower == "negative":
            return "Shocked"
        else:
            return "Shocked"

    # 3. Anger
    elif emotion_lower == "anger":
        if sentiment_lower == "negative":
            return "Bitter"
        elif sentiment_lower == "neutral":
            return "Frustrated"
        else:
            return "Frustrated"

    # 4. Fear
    elif emotion_lower == "fear":
        if sentiment_lower == "negative":
            return "Anxious"
        elif sentiment_lower == "neutral":
            return "Anxious"
        else:
            return "Shocked"

    # 5. Sadness
    elif emotion_lower == "sadness":
        if sentiment_lower == "negative":
            return "Frustrated"
        elif sentiment_lower == "neutral":
            return "Peaceful"
        else:
            return "Grateful"

    # Fallback rules based solely on sentiment
    if sentiment_lower == "positive":
        return "Excited"
    elif sentiment_lower == "negative":
        return "Frustrated"
    else:
        return "Peaceful"
