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