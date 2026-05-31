# API FastAPI — Telemed Urgence IA

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import Optional
import joblib
import scipy.sparse as sp
import numpy as np
import json
import os
from datetime import datetime

# chargement des modèles
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH     = os.path.join(BASE_DIR, "models", "best_model.joblib")
TAB_PREP_PATH  = os.path.join(BASE_DIR, "models", "tabular_preprocessor.joblib")
TFIDF_PATH     = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.joblib")
LOG_PATH       = os.path.join(BASE_DIR, "..", "logs", "inference_log.json")

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

model          = joblib.load(MODEL_PATH)
tab_prep       = joblib.load(TAB_PREP_PATH)
tfidf          = joblib.load(TFIDF_PATH)

# application FastAPI
app = FastAPI(
    title="Telemed Urgence IA",
    description="API de prédiction du niveau d'urgence médicale",
    version="1.0.0"
)

# schéma d'entrée (Pydantic)
class PatientInput(BaseModel):
    sexe: str = Field(..., pattern="^[FH]$", description="F ou H")
    age: float = Field(..., ge=0, le=120)
    zone_vie: str = Field(..., pattern="^[UR]$", description="U ou R")
    source: str = Field(..., pattern="^(appel|chat)$")
    freq_cardiaque: float = Field(..., ge=30, le=250)
    tension_sys: float = Field(..., ge=50, le=300)
    temp: float = Field(..., ge=34.0, le=43.0)
    sat_oxygene: float = Field(..., ge=50.0, le=100.0)
    antecedents: float = Field(..., ge=0, le=1)
    duree_symptomes: float = Field(..., ge=0)
    description_symptomes: str = Field(..., min_length=3)

    @field_validator('description_symptomes')
    @classmethod
    def clean_text(cls, v):
        return v.strip()

# labels
LABELS = {
    0: "Non urgent",
    1: "Urgence relative",
    2: "Urgence vitale ⚠️"
}

# fonction de preprocessing
def preprocess(data: PatientInput):
    import pandas as pd
    df = pd.DataFrame([data.model_dump()])
    tab_features = tab_prep.transform(df)
    text_features = tfidf.transform([data.description_symptomes])
    return sp.hstack([tab_features, text_features])

# Route: santé de l'API
@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "1.0.0",
        "model": "LogisticRegression",
        "timestamp": datetime.now().isoformat()
    }

# Route: prédiction
@app.post("/predict")
def predict(patient: PatientInput):
    try:
        X = preprocess(patient)
        prediction = int(model.predict(X)[0])
        probas = model.predict_proba(X)[0].tolist()

        result = {
            "niveau_urgence": prediction,
            "label": LABELS[prediction],
            "probabilites": {
                "non_urgent": round(probas[0], 4),
                "urgence_relative": round(probas[1], 4),
                "urgence_vitale": round(probas[2], 4)
            },
            "timestamp": datetime.now().isoformat()
        }

        # logging de l'inférence
        log_entry = {**patient.model_dump(), **result}
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Route: historique des inférences
@app.get("/history")
def history(limit: int = 10):
    if not os.path.exists(LOG_PATH):
        return {"history": []}
    with open(LOG_PATH, "r") as f:
        lines = f.readlines()
    entries = [json.loads(l) for l in lines[-limit:]]
    return {"count": len(entries), "history": entries}

# Route: réentraînement monitoré
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
import scipy.sparse as sp

# Route: réentraînement monitoré
@app.post("/retrain")
def retrain():
    try:
        import joblib as jl
        from sklearn.linear_model import LogisticRegression as LR
        from sklearn.metrics import accuracy_score, f1_score, recall_score

        scenarios = jl.load(os.path.join(BASE_DIR, "..", "data", "processed", "scenarios.joblib"))
        y_train, y_test = jl.load(os.path.join(BASE_DIR, "..", "data", "processed", "labels.joblib"))

        X_train = scenarios['S1_multimodal']['X_train']
        X_test  = scenarios['S1_multimodal']['X_test']

        # entraînement nouveau modèle
        new_model = LR(max_iter=1000, class_weight='balanced', random_state=42)
        new_model.fit(X_train, y_train)

        # métriques nouveau modèle
        y_pred_new = new_model.predict(X_test)
        new_metrics = {
            "accuracy"   : round(accuracy_score(y_test, y_pred_new), 4),
            "f1_weighted": round(f1_score(y_test, y_pred_new, average='weighted'), 4),
            "recall_c2"  : round(recall_score(y_test, y_pred_new, labels=[2], average=None)[0], 4)
        }

        # métriques modèle actuel
        current_model = jl.load(MODEL_PATH)
        y_pred_cur = current_model.predict(X_test)
        current_metrics = {
            "accuracy"   : round(accuracy_score(y_test, y_pred_cur), 4),
            "f1_weighted": round(f1_score(y_test, y_pred_cur, average='weighted'), 4),
            "recall_c2"  : round(recall_score(y_test, y_pred_cur, labels=[2], average=None)[0], 4)
        }

        # décision
        improved = bool(new_metrics['recall_c2'] >= current_metrics['recall_c2'])
        if improved:
            jl.dump(new_model, MODEL_PATH)
            status = "✅ Modèle mis à jour"
        else:
            status = "⚠️ Ancien modèle conservé"

        # logging
        log_entry = {
            "type"           : "retrain",
            "timestamp"      : datetime.now().isoformat(),
            "status"         : status,
            "new_metrics"    : new_metrics,
            "current_metrics": current_metrics,
            "model_updated"  : improved
        }
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        return {
            "status"         : status,
            "model_updated"  : improved,
            "new_metrics"    : new_metrics,
            "current_metrics": current_metrics,
            "timestamp"      : datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))