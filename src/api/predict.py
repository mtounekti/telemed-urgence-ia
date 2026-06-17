# logic de prédiction
import os
import joblib
import scipy.sparse as sp
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

MODEL_PATH    = os.path.join(PROJECT_ROOT, "models", "triage_model_optimized.joblib")
TAB_PREP_PATH = os.path.join(BASE_DIR, "models", "tabular_preprocessor.joblib")
TFIDF_PATH    = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.joblib")

LABELS = {
    0: "Non urgent",
    1: "Urgence relative",
    2: "Urgence vitale ⚠️"
}


def load_artifacts():
    """Charger les artefacts ML depuis le disque."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")

    bundle = joblib.load(MODEL_PATH)
    model = bundle['model']
    threshold_class_2 = bundle.get('threshold_class_2', 0.5)
    model_name = bundle.get('model_name', 'unknown')

    tab_prep = joblib.load(TAB_PREP_PATH)
    tfidf = joblib.load(TFIDF_PATH)

    return model, tab_prep, tfidf, threshold_class_2, model_name


def predict_with_threshold(model, X, threshold_class2: float) -> int:
    """
    Applique le seuil personnalisé sur la classe 2:
    si proba(classe 2) >= seuil, on prédit 2,
    sinon on prédit argmax parmi les classes 0 et 1
    """
    probas = model.predict_proba(X)[0]
    if probas[2] >= threshold_class2:
        return 2
    return int(np.argmax(probas[:2]))


def predict_one(data: dict) -> dict:
    """
    effectue une prédiction pour un patient
    Retourne le niveau d'urgence, le label et les probabilités,
    en appliquant le seuil de décision optimisé pour la classe 2
    """
    model, tab_prep, tfidf, threshold_class_2, model_name = load_artifacts()

    df = pd.DataFrame([data])
    text = data.get("description_symptomes", "")

    tab_features = tab_prep.transform(df)
    text_features = tfidf.transform([text])
    X = sp.hstack([tab_features, text_features]).tocsr()

    prediction = predict_with_threshold(model, X, threshold_class_2)
    probas = model.predict_proba(X)[0].tolist()

    return {
        "niveau_urgence": prediction,
        "label": LABELS[prediction],
        "probabilites": {
            "non_urgent": round(probas[0], 4),
            "urgence_relative": round(probas[1], 4),
            "urgence_vitale": round(probas[2], 4),
        },
        "model_name": model_name,
        "threshold_class_2": threshold_class_2,
    }