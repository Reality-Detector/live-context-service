# src/live_context/core/quality.py
from typing import Dict


def compute_quality(text: str, confidence: float, is_final: bool) -> Dict[str, object]:
    """
    Compute simple quality metrics for an utterance.

    - word_count / char_count are basic size indicators.
    - avg_confidence is just the ASR-provided confidence for now.
    - langchain_ready is a heuristic: only true for final utterances that
      are long and confident enough to be worth processing.
    """
    words = text.strip().split()
    word_count = len(words)
    char_count = len(text)

    # Heuristic thresholds – you can tune these later
    min_words = 8
    min_conf = 0.85

    if is_final and word_count >= min_words and confidence >= min_conf:
        langchain_ready = True
    else:
        langchain_ready = False

    return {
        "word_count": word_count,
        "char_count": char_count,
        "avg_confidence": float(confidence),
        "langchain_ready": langchain_ready,
        "is_final": is_final,
    }
