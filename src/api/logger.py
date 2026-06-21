# Logger structuré JSON

import hashlib
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

# Champ(s) contenant du texte libre patient : jamais journalisé en clair.
TEXT_FIELDS_TO_HASH = ["description_symptomes"]
HASH_TRUNCATE_LENGTH = 16  # caractères hex conservés (64 bits, largement suffisant pour de l'audit/dédup)


def _fingerprint_text(text: str | None) -> dict:
    """Remplace un texte libre par une empreinte SHA-256 tronquée et sa longueur.
    Empêche toute reconstruction du verbatim patient à partir des logs,
    tout en gardant un signal exploitable pour le débogage et la détection
    de doublons (même texte -> même empreinte)."""
    if not text:
        return {"hash": None, "length": 0}
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {"hash": digest[:HASH_TRUNCATE_LENGTH], "length": len(text)}


def _sanitize_payload(payload: dict) -> dict:
    """Retourne une copie du payload où les champs de texte libre sont
    remplacés par leur empreinte (hash tronqué + longueur), jamais le texte brut."""
    safe = dict(payload)
    for field in TEXT_FIELDS_TO_HASH:
        if field in safe:
            text = safe.pop(field)
            fp = _fingerprint_text(text)
            safe[f"{field}_hash"] = fp["hash"]
            safe[f"{field}_length"] = fp["length"]
    return safe


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
        "input": _sanitize_payload(payload),
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
        "payload": _sanitize_payload(payload) if payload else None,
    }
    logger.error(json.dumps(record, ensure_ascii=False))