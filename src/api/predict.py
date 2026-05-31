# Logique de prédiction

import os
import joblib
import scipy.sparse as sp
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH    = os.path.join(BASE_DIR, "models", "best_model.joblib")
TAB_PREP_PATH = os.path.join(BASE_DIR, "models", "tabular_preprocessor.joblib")
TFIDF_PATH    = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.joblib")

LABELS = {
    0: "Non urgent",
    1: "Urgence relative",
    2: "Urgence vitale ⚠️"
}


def load_artifacts():
    """charger les artefacts ML depuis le disque"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")
    model    = joblib.load(MODEL_PATH)
    tab_prep = joblib.load(TAB_PREP_PATH)
    tfidf    = joblib.load(TFIDF_PATH)
    return model, tab_prep, tfidf


def predict_one(data: dict) -> dict:
    """
    effectueer une prédiction pour un patient.
    return le niveau d'urgence + le label + les probabilités
    """
    model, tab_prep, tfidf = load_artifacts()

    df = pd.DataFrame([data])
    text = data.get("description_symptomes", "")

    tab_features  = tab_prep.transform(df)
    text_features = tfidf.transform([text])
    X = sp.hstack([tab_features, text_features])

    prediction = int(model.predict(X)[0])
    probas     = model.predict_proba(X)[0].tolist()

    return {
        "niveau_urgence": prediction,
        "label": LABELS[prediction],
        "probabilites": {
            "non_urgent"       : round(probas[0], 4),
            "urgence_relative" : round(probas[1], 4),
            "urgence_vitale"   : round(probas[2], 4),
        }
    }