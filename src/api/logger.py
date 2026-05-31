# Logger structuré JSON

import json
import logging
import os
from datetime import datetime, timezone

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "inference.log")

# configuration du logger
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(message)s",
)

logger = logging.getLogger("telemed")

def log_inference(
    payload: dict,
    result: dict,
    duration_ms: float,
    session_id: str | None = None,
    user_id: str | None = None,
) -> None:
    """log structuré d'une inférence"""
    record = {
        "event": "prediction",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "user_id": user_id,
        "duration_ms": duration_ms,
        "input": payload,
        "output": result,
    }
    logger.info(json.dumps(record, ensure_ascii=False))


def log_retrain(status: str, metrics: dict, model_updated: bool) -> None:
    """log structuré d'un réentraînement"""
    record = {
        "event": "retrain",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "model_updated": model_updated,
        "metrics": metrics,
    }
    logger.info(json.dumps(record, ensure_ascii=False))


def log_error(event: str, error: str, payload: dict | None = None) -> None:
    """log structuré d'une erreur"""
    record = {
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "error": error,
        "payload": payload,
    }
    logger.error(json.dumps(record, ensure_ascii=False))