from __future__ import annotations

import os
import time
import joblib
from datetime import datetime, timezone
from typing import Annotated

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from src.api.database import save_feedback, get_feedbacks
from src.api.schemas import (
    PatientInput, PredictionResponse,
    RetrainResponse, HealthResponse
)
from src.api.predict import predict_one, load_artifacts, LABELS
from src.api.logger import log_inference, log_retrain, log_error

# chargement initial des artefacts
try:
    _model, _tab_prep, _tfidf = load_artifacts()
except FileNotFoundError as e:
    raise RuntimeError(f"Impossible de charger les artefacts : {e}")

# app
app = FastAPI(
    title="Telemed Urgence IA",
    description="API de prédiction du niveau d'urgence médicale",
    version="1.0.0",
)

# cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Métriques Prometheus
REQUEST_COUNT = Counter(
    'telemed_requests_total',
    'Nombre total de requêtes',
    ['method', 'endpoint', 'status']
)
REQUEST_LATENCY = Histogram(
    'telemed_request_latency_seconds',
    'Latence des requêtes en secondes',
    ['endpoint']
)
PREDICTION_COUNT = Counter(
    'telemed_predictions_total',
    'Nombre de prédictions par classe',
    ['niveau_urgence']
)

# Clé API retrain
RETRAIN_API_KEY = os.getenv("RETRAIN_API_KEY", "telemed-secret-key")


@app.get("/metrics")
def metrics():
    """Endpoint Prometheus — métriques temps réel"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    REQUEST_COUNT.labels(method="GET", endpoint="/health", status="200").inc()
    return HealthResponse(
        status="ok",
        version="1.0.0",
        model="LogisticRegression",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(
    patient: PatientInput,
    x_session_id: Annotated[str | None, Header()] = None,
    x_user_id: Annotated[str | None, Header()] = None,
) -> PredictionResponse:
    """
    Prédit le niveau d'urgence d'un patient
    Headers optionnels : X-Session-ID, X-User-ID pour la traçabilité
    """
    started_at = time.perf_counter()

    try:
        result = predict_one(patient.model_dump())
    except FileNotFoundError as exc:
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="503").inc()
        log_error("prediction_error", str(exc), patient.model_dump())
        raise HTTPException(status_code=503, detail="Modèle indisponible") from exc
    except Exception as exc:
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="500").inc()
        log_error("prediction_error", str(exc), patient.model_dump())
        raise HTTPException(status_code=500, detail="Erreur pendant l'inférence") from exc

    duration_ms = round((time.perf_counter() - started_at) * 1000, 2)

    # métriques Prometheus
    REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="200").inc()
    REQUEST_LATENCY.labels(endpoint="/predict").observe(duration_ms / 1000)
    PREDICTION_COUNT.labels(niveau_urgence=str(result['niveau_urgence'])).inc()

    # Logging structuré
    log_inference(
        payload=patient.model_dump(),
        result=result,
        duration_ms=duration_ms,
        session_id=x_session_id,
        user_id=x_user_id,
    )

    return PredictionResponse(
        **result,
        timestamp=datetime.now(timezone.utc).isoformat(),
        duration_ms=duration_ms,
    )


@app.post("/retrain", response_model=RetrainResponse)
def retrain(
    background_tasks: BackgroundTasks,
    x_api_key: Annotated[str | None, Header()] = None,
) -> RetrainResponse:
    """
    Réentraîne le modèle de manière sécurisée
    Requiert le header X-API-Key
    """
    # Authentification
    if x_api_key != RETRAIN_API_KEY:
        REQUEST_COUNT.labels(method="POST", endpoint="/retrain", status="403").inc()
        raise HTTPException(status_code=403, detail="Clé API invalide")

    try:
        import joblib as jl
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score, recall_score

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        MODEL_PATH     = os.path.join(BASE_DIR, "models", "best_model.joblib")
        SCENARIOS_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "scenarios.joblib")
        LABELS_PATH    = os.path.join(BASE_DIR, "..", "data", "processed", "labels.joblib")

        scenarios = jl.load(SCENARIOS_PATH)
        y_train, y_test = jl.load(LABELS_PATH)

        X_train = scenarios['S1_multimodal']['X_train']
        X_test  = scenarios['S1_multimodal']['X_test']

        # Nouveau modèle
        new_model = LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42
        )
        new_model.fit(X_train, y_train)

        # métriques nouveau modèle
        y_pred_new = new_model.predict(X_test)
        new_metrics = {
            "accuracy"   : round(float(accuracy_score(y_test, y_pred_new)), 4),
            "f1_weighted": round(float(f1_score(y_test, y_pred_new, average='weighted')), 4),
            "recall_c2"  : round(float(recall_score(y_test, y_pred_new, labels=[2], average=None)[0]), 4),
        }

        # Métriques modèle actuel
        current_model = jl.load(MODEL_PATH)
        y_pred_cur = current_model.predict(X_test)
        current_metrics = {
            "accuracy"   : round(float(accuracy_score(y_test, y_pred_cur)), 4),
            "f1_weighted": round(float(f1_score(y_test, y_pred_cur, average='weighted')), 4),
            "recall_c2"  : round(float(recall_score(y_test, y_pred_cur, labels=[2], average=None)[0]), 4),
        }

        # Décision
        improved = bool(new_metrics['recall_c2'] >= current_metrics['recall_c2'])
        status = "✅ Modèle mis à jour" if improved else "⚠️ Ancien modèle conservé"
        if improved:
            jl.dump(new_model, MODEL_PATH)

        REQUEST_COUNT.labels(method="POST", endpoint="/retrain", status="200").inc()
        background_tasks.add_task(log_retrain, status, new_metrics, improved)

        return RetrainResponse(
            status=status,
            model_updated=improved,
            new_metrics=new_metrics,
            current_metrics=current_metrics,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    except Exception as exc:
        REQUEST_COUNT.labels(method="POST", endpoint="/retrain", status="500").inc()
        log_error("retrain_error", str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/history")
def history(limit: int = 10):
    """Retourne l'historique des inférences."""
    REQUEST_COUNT.labels(method="GET", endpoint="/history", status="200").inc()

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    LOG_FILE = os.path.join(BASE_DIR, "..", "logs", "inference.log")

    if not os.path.exists(LOG_FILE):
        return {"count": 0, "history": []}

    import json
    with open(LOG_FILE, "r") as f:
        lines = f.readlines()

    entries = []
    for line in lines[-limit:]:
        try:
            entry = json.loads(line)
            if entry.get("event") == "prediction":
                entries.append(entry)
        except Exception:
            continue

    return {"count": len(entries), "history": entries}

@app.post("/feedback")
def feedback(
    niveau_predit: int,
    utile: bool,
    niveau_reel: int | None = None,
    commentaire: str | None = None,
    x_session_id: Annotated[str | None, Header()] = None,
):
    """save le feedback utilisateur après une prédiction"""
    try:
        feedback_id = save_feedback(
            niveau_predit=niveau_predit,
            utile=utile,
            niveau_reel=niveau_reel,
            commentaire=commentaire,
            session_id=x_session_id,
        )
        REQUEST_COUNT.labels(method="POST", endpoint="/feedback", status="200").inc()
        return {
            "status": "✅ Feedback enregistré",
            "feedback_id": feedback_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        REQUEST_COUNT.labels(method="POST", endpoint="/feedback", status="500").inc()
        log_error("feedback_error", str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/feedbacks")
def feedbacks(limit: int = 50):
    """retourne l'historique des feedbacks"""
    REQUEST_COUNT.labels(method="GET", endpoint="/feedbacks", status="200").inc()
    data = get_feedbacks(limit=limit)
    return {"count": len(data), "feedbacks": data}
