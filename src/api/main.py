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
from src.api.predict import predict_one, load_artifacts, predict_with_threshold, LABELS
from src.api.logger import log_inference, log_retrain, log_error

# chargement initial des artefacts
try:
    _model, _tab_prep, _tfidf, _threshold, _model_name = load_artifacts()
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
        model=f"{_model_name} (seuil_classe_2={_threshold})",
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

    # logs structuré
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
    Réapplique la recherche du seuil optimal sur la classe 2.
    Requiert le header X-API-Key
    """
    if x_api_key != RETRAIN_API_KEY:
        REQUEST_COUNT.labels(method="POST", endpoint="/retrain", status="403").inc()
        raise HTTPException(status_code=403, detail="Clé API invalide")

    try:
        import joblib as jl
        import numpy as np
        from xgboost import XGBClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.impute import SimpleImputer
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics import accuracy_score, f1_score, recall_score
        import scipy.sparse as sp
        import pandas as pd

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        DATA_PATH = os.path.join(BASE_DIR, "..", "data", "raw", "dataset_telemed.csv")
        PROJECT_ROOT = os.path.dirname(BASE_DIR)
        MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "triage_model_optimized.joblib")

        col_num = ['age', 'freq_cardiaque', 'tension_sys', 'temp', 'sat_oxygene', 'antecedents', 'duree_symptomes']
        col_cat = ['sexe', 'zone_vie', 'source']
        col_text = 'description_symptomes'

        df = pd.read_csv(DATA_PATH).drop(columns=['patient_id'])
        X = df.drop(columns=['niveau_urgence'])
        y = df['niveau_urgence']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        preprocessor = ColumnTransformer([
            ('num', numeric_pipeline, col_num),
            ('cat', categorical_pipeline, col_cat),
        ])
        X_train_tab = preprocessor.fit_transform(X_train)
        X_test_tab = preprocessor.transform(X_test)

        tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), sublinear_tf=True)
        X_train_text = tfidf.fit_transform(X_train[col_text].fillna('').tolist())
        X_test_text = tfidf.transform(X_test[col_text].fillna('').tolist())

        X_train_full = sp.hstack([X_train_tab, X_train_text]).tocsr()
        X_test_full = sp.hstack([X_test_tab, X_test_text]).tocsr()

        new_model = XGBClassifier(n_estimators=200, random_state=42, eval_metric='mlogloss', n_jobs=-1)
        new_model.fit(X_train_full, y_train)

        thresholds = [0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03, 0.01]

        def critical_count(model, X, y, threshold):
            probas = model.predict_proba(X)
            y_pred = np.array([
                2 if probas[i, 2] >= threshold else int(np.argmax(probas[i, :2]))
                for i in range(X.shape[0])
            ])
            y_arr = np.asarray(y)
            critical = int(np.sum((y_arr == 2) & (y_pred != 2)))
            return critical, y_pred

        baseline_critical, y_pred_baseline = critical_count(new_model, X_test_full, y_test, 0.5)
        baseline_f1 = f1_score(y_test, y_pred_baseline, average='weighted')

        best_threshold, best_critical = 0.5, baseline_critical
        for t in thresholds:
            c, y_pred_t = critical_count(new_model, X_test_full, y_test, t)
            f1_t = f1_score(y_test, y_pred_t, average='weighted')
            if f1_t >= (baseline_f1 - 0.03) and c < best_critical:
                best_threshold, best_critical = t, c

        _, y_pred_new = critical_count(new_model, X_test_full, y_test, best_threshold)
        new_metrics = {
            "accuracy": round(float(accuracy_score(y_test, y_pred_new)), 4),
            "f1_weighted": round(float(f1_score(y_test, y_pred_new, average='weighted')), 4),
            "recall_c2": round(float(recall_score(y_test, y_pred_new, labels=[2], average=None)[0]), 4),
        }

        current_bundle = jl.load(MODEL_PATH)
        current_model = current_bundle['model']
        current_threshold = current_bundle['threshold_class_2']
        _, y_pred_cur = critical_count(current_model, X_test_full, y_test, current_threshold)
        current_metrics = {
            "accuracy": round(float(accuracy_score(y_test, y_pred_cur)), 4),
            "f1_weighted": round(float(f1_score(y_test, y_pred_cur, average='weighted')), 4),
            "recall_c2": round(float(recall_score(y_test, y_pred_cur, labels=[2], average=None)[0]), 4),
        }

        improved = bool(new_metrics['recall_c2'] >= current_metrics['recall_c2'])
        status = " Modèle mis à jour" if improved else " Ancien modèle conservé"
        if improved:
            jl.dump(
                {'model': new_model, 'threshold_class_2': best_threshold, 'model_name': 'xgboost'},
                MODEL_PATH
            )
            jl.dump(preprocessor, os.path.join(BASE_DIR, "models", "tabular_preprocessor.joblib"))
            jl.dump(tfidf, os.path.join(BASE_DIR, "models", "tfidf_vectorizer.joblib"))

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