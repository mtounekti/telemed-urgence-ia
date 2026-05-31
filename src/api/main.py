
# API FastAPI
from __future__ import annotations

import os
import time
import joblib
import copy
from datetime import datetime, timezone
from typing import Annotated

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Clé API retrain
RETRAIN_API_KEY = os.getenv("RETRAIN_API_KEY", "telemed-secret-key")

# Routes

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
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
    prédit le niveau d'urgence d'un patient
    headers optionnels : X-Session-ID, X-User-ID pour la traçabilité
    """
    started_at = time.perf_counter()

    try:
        result = predict_one(patient.model_dump())
    except FileNotFoundError as exc:
        log_error("prediction_error", str(exc), patient.model_dump())
        raise HTTPException(status_code=503, detail="Modèle indisponible") from exc
    except Exception as exc:
        log_error("prediction_error", str(exc), patient.model_dump())
        raise HTTPException(status_code=500, detail="Erreur pendant l'inférence") from exc

    duration_ms = round((time.perf_counter() - started_at) * 1000, 2)

    # logging asynchrone
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
    Réentraîne le modèle de manière sécurisée.
    Requiert le header X-API-Key
    """
    # Authentification
    if x_api_key != RETRAIN_API_KEY:
        raise HTTPException(status_code=403, detail="Clé API invalide")

    try:
        import joblib as jl
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score, recall_score

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.joblib")
        SCENARIOS_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "scenarios.joblib")
        LABELS_PATH    = os.path.join(BASE_DIR, "..", "data", "processed", "labels.joblib")

        scenarios = jl.load(SCENARIOS_PATH)
        y_train, y_test = jl.load(LABELS_PATH)

        X_train = scenarios['S1_multimodal']['X_train']
        X_test  = scenarios['S1_multimodal']['X_test']

        # new modèle
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

        # métriques modèle actuel
        current_model = jl.load(MODEL_PATH)
        y_pred_cur = current_model.predict(X_test)
        current_metrics = {
            "accuracy"   : round(float(accuracy_score(y_test, y_pred_cur)), 4),
            "f1_weighted": round(float(f1_score(y_test, y_pred_cur, average='weighted')), 4),
            "recall_c2"  : round(float(recall_score(y_test, y_pred_cur, labels=[2], average=None)[0]), 4),
        }

        # décision
        improved = bool(new_metrics['recall_c2'] >= current_metrics['recall_c2'])
        if improved:
            jl.dump(new_model, MODEL_PATH)
            status = "✅ Modèle mis à jour"
        else:
            status = "⚠️ Ancien modèle conservé"

        # log en arrière-plan
        background_tasks.add_task(
            log_retrain, status, new_metrics, improved
        )

        return RetrainResponse(
            status=status,
            model_updated=improved,
            new_metrics=new_metrics,
            current_metrics=current_metrics,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    except Exception as exc:
        log_error("retrain_error", str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/history")
def history(limit: int = 10):
    """return l'historique des inférences"""
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