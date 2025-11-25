# src/live_context/protocol/events_v1.py
from typing import Dict, Any


def session_started(config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "session_started",
        "config": config,
    }


def session_stopped() -> Dict[str, Any]:
    return {
        "type": "session_stopped",
    }


def partial_utterance(
    session_id: str,
    utterance_id: str,
    speaker: str,
    t_start: float,
    t_end: float,
    text: str,
    confidence: float,
) -> Dict[str, Any]:
    return {
        "type": "partial_utterance",
        "session_id": session_id,
        "utterance_id": utterance_id,
        "speaker": speaker,
        "t_start": t_start,
        "t_end": t_end,
        "text": text,
        "confidence": confidence,
    }


def final_utterance(
    session_id: str,
    utterance_id: str,
    speaker: str,
    t_start: float,
    t_end: float,
    raw_text: str,
    confidence: float,
) -> Dict[str, Any]:
    return {
        "type": "final_utterance",
        "session_id": session_id,
        "utterance_id": utterance_id,
        "speaker": speaker,
        "t_start": t_start,
        "t_end": t_end,
        "raw_text": raw_text,
        "confidence": confidence,
    }
