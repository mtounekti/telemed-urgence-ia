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

def explain_prediction(data: dict, niveau_urgence: int, probas: list) -> list:
    """
    Génère une interprétation lisible de la prédiction en français,
    basée sur les signaux cliniques et textuels détectés.
    Ce n'est pas une explication médicale, mais un résumé des facteurs
    qui ont probablement influencé la décision du modèle.
    """
    explications = []
    texte = data.get("description_symptomes", "").lower()

    # Signaux textuels (mots-clés simples, sans NLP avancé)
    signaux_texte = {
        "thoracique": "La description mentionne une douleur thoracique, signal souvent associé à un risque élevé.",
        "respiratoire": "La description mentionne une détresse respiratoire, signal de gravité potentielle.",
        "essoufflement": "La description mentionne un essoufflement, ce qui peut orienter vers une situation plus urgente.",
        "convulsion": "La description mentionne des convulsions, signal d'urgence vitale potentielle.",
        "perte de connaissance": "La description mentionne une perte de connaissance, signal critique.",
        "anaphylactique": "La description mentionne une réaction allergique sévère.",
        "plaie": "La description mentionne une plaie ou blessure ouverte.",
        "saignement": "La description mentionne un saignement.",
    }
    for mot_cle, message in signaux_texte.items():
        if mot_cle in texte:
            explications.append(message)

    # Signaux cliniques (constantes vitales)
    sat = data.get("sat_oxygene")
    if sat is not None and sat < 92:
        explications.append("La saturation en oxygène est basse, ce qui augmente le niveau de vigilance.")

    fc = data.get("freq_cardiaque")
    if fc is not None and (fc > 110 or fc < 50):
        explications.append("La fréquence cardiaque est anormale, ce qui peut indiquer une situation plus urgente.")

    temp = data.get("temp")
    if temp is not None and temp >= 38.5:
        explications.append("La température est élevée.")

    tension = data.get("tension_sys")
    if tension is not None and tension >= 170:
        explications.append("La tension systolique est élevée.")

    # Résumé des probabilités
    explications.append(
        f"Probabilités estimées par le modèle : "
        f"classe 0: {probas[0]*100:.1f}%, classe 1: {probas[1]*100:.1f}%, classe 2: {probas[2]*100:.1f}%. "
        f"Classe retenue : `{niveau_urgence}`."
    )

    if niveau_urgence == 2:
        explications.append(
            "Le modèle utilise un seuil plus prudent pour la classe `2` afin de limiter "
            "les urgences vitales manquées."
        )

    return explications

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
    interpretation = explain_prediction(data, prediction, probas)

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
        "interpretation": interpretation,
    }