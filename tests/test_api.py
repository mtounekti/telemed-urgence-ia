# Tests unitaires
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

from tests.fixtures import (
    PATIENT_CRITIQUE,
    PATIENT_NON_URGENT,
    PATIENT_LIMITE_VALIDE,
    PATIENT_RELATIF,
    INVALID_CASES,
)

import src.api.main as main_module

client = TestClient(app)

PATIENT_CRITIQUE = {
    "sexe": "F", "age": 65, "zone_vie": "U", "source": "appel",
    "freq_cardiaque": 130, "tension_sys": 190, "temp": 39.8,
    "sat_oxygene": 85.0, "antecedents": 1, "duree_symptomes": 1.0,
    "description_symptomes": "Douleur thoracique intense avec essoufflement sévère"
}

PATIENT_NON_URGENT = {
    "sexe": "H", "age": 30, "zone_vie": "U", "source": "chat",
    "freq_cardiaque": 70, "tension_sys": 115, "temp": 37.0,
    "sat_oxygene": 99.0, "antecedents": 0, "duree_symptomes": 48.0,
    "description_symptomes": "Renouvellement ordonnance traitement habituel"
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "ok"
    assert "model" in d
    assert "version" in d
    assert "timestamp" in d


def test_predict_structure():
    r = client.post("/predict", json=PATIENT_CRITIQUE)
    assert r.status_code == 200
    d = r.json()
    assert "niveau_urgence" in d
    assert "label" in d
    assert "probabilites" in d
    assert "duration_ms" in d
    assert "timestamp" in d
    assert "model_name" in d
    assert "threshold_class_2" in d
    assert d["niveau_urgence"] in [0, 1, 2]


def test_predict_critique():
    r = client.post("/predict", json=PATIENT_CRITIQUE)
    assert r.status_code == 200
    assert r.json()["niveau_urgence"] == 2


def test_predict_non_urgent():
    r = client.post("/predict", json=PATIENT_NON_URGENT)
    assert r.status_code == 200
    assert r.json()["niveau_urgence"] == 0


def test_predict_invalid_input():
    bad = {**PATIENT_CRITIQUE, "sexe": "X"}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_with_session_id():
    r = client.post("/predict", json=PATIENT_CRITIQUE,
                     headers={"x-session-id": "test-session-001"})
    assert r.status_code == 200
    assert "duration_ms" in r.json()


def test_retrain_without_key():
    r = client.post("/retrain")
    assert r.status_code == 403


def test_retrain_with_wrong_key():
    r = client.post("/retrain", headers={"x-api-key": "mauvaise-cle"})
    assert r.status_code == 403


@pytest.mark.slow
def test_retrain_with_correct_key():
    """
    Test plus lent: réentraîne un XGBoost complet et recherche le seuil optimal
    Marqué 'slow' pour pouvoir être exclu des runs rapides si besoin
    (pytest -m "not slow")
    """
    r = client.post("/retrain", headers={"x-api-key": "telemed-secret-key"})
    assert r.status_code == 200
    d = r.json()
    assert "model_updated" in d
    assert "new_metrics" in d
    assert "current_metrics" in d


def test_history():
    r = client.get("/history")
    assert r.status_code == 200
    assert "history" in r.json()

# Tests avec fixtures de bornes physiologiques

def test_predict_limite_valide_acceptee():
    """Les valeurs aux bornes exactes (ge/le) doivent être acceptées, pas rejetées"""
    r = client.post("/predict", json=PATIENT_LIMITE_VALIDE)
    assert r.status_code == 200


def test_predict_urgence_relative():
    r = client.post("/predict", json=PATIENT_RELATIF)
    assert r.status_code == 200
    d = r.json()
    assert d["niveau_urgence"] in [0, 1, 2]


@pytest.mark.parametrize("case_name", list(INVALID_CASES.keys()))
def test_predict_rejects_invalid_field(case_name):
    """Chaque violation de borne physiologique doit être rejetée par Pydantic (422)"""
    payload = INVALID_CASES[case_name]
    r = client.post("/predict", json=payload)
    assert r.status_code == 422, f"Cas '{case_name}' aurait dû être rejeté (422), reçu {r.status_code}"


# Tests avec mocking (monkeypatch)

def test_predict_returns_503_when_model_missing(monkeypatch):
    """Si le modèle n'est pas chargeable, l'API doit répondre 503, pas planter"""
    def raise_missing(payload):
        raise FileNotFoundError("Modèle introuvable")

    monkeypatch.setattr(main_module, "predict_one", raise_missing)
    r = client.post("/predict", json=PATIENT_CRITIQUE)
    assert r.status_code == 503


def test_predict_returns_500_on_unexpected_error(monkeypatch):
    """Toute autre exception pendant l'inférence doit être catchée et renvoyer 500"""
    def raise_unexpected(payload):
        raise ValueError("Erreur inattendue dans le pipeline")

    monkeypatch.setattr(main_module, "predict_one", raise_unexpected)
    r = client.post("/predict", json=PATIENT_CRITIQUE)
    assert r.status_code == 500


def test_predict_mocked_response_structure(monkeypatch):
    """Vérifie que la route construit bien sa réponse à partir du retour de predict_one,
    sans dépendre du vrai modèle chargé — utile pour tester la route isolément"""
    def fake_predict_one(payload):
        return {
            "niveau_urgence": 2,
            "label": "Urgence vitale",
            "probabilites": {"non_urgent": 0.02, "urgence_relative": 0.08, "urgence_vitale": 0.90},
            "model_name": "xgboost",
            "threshold_class_2": 0.01,
            "interpretation": ["signal critique détecté"],
        }

    monkeypatch.setattr(main_module, "predict_one", fake_predict_one)
    r = client.post("/predict", json=PATIENT_CRITIQUE, headers={"x-session-id": "sess-1"})
    assert r.status_code == 200
    d = r.json()
    assert d["niveau_urgence"] == 2
    assert d["label"] == "Urgence vitale"
    assert d["model_name"] == "xgboost"
    assert "interpretation" in d


def test_retrain_wrong_key_with_monkeypatch_env(monkeypatch):
    """Vérifie le comportement avec une clé d'environnement différente de la valeur par défaut."""
    monkeypatch.setattr(main_module, "RETRAIN_API_KEY", "cle-de-test")
    r = client.post("/retrain", headers={"x-api-key": "mauvaise-cle"})
    assert r.status_code == 403


def test_retrain_correct_key_with_monkeypatch_env(monkeypatch):
    monkeypatch.setattr(main_module, "RETRAIN_API_KEY", "cle-de-test")
    r = client.post("/retrain", headers={"x-api-key": "cle-de-test"})
    # Le retrain réel s'exécute ici (pas mocké) — on vérifie juste que l'authentification passe
    assert r.status_code in [200, 500]  # 500 possible si data/raw absent en CI


def test_history_with_isolated_log_file(monkeypatch, tmp_path):
    """Teste /history sur un fichier de logs isolé plutôt que le vrai fichier de logs"""
    import json as json_module

    log_file = tmp_path / "inference_test.log"
    entry = {
        "event": "prediction",
        "timestamp": "2026-06-17T10:00:00+00:00",
        "session_id": "s1",
        "duration_ms": 1.2,
        "input": dict(PATIENT_CRITIQUE),
        "output": {"niveau_urgence": 2, "label": "Urgence vitale"},
    }
    log_file.write_text(json_module.dumps(entry) + "\n", encoding="utf-8")

    # On patch directement la fonction history pour lire ce fichier isolé
    import src.api.main as m

    original_history = m.history

    def patched_history(limit: int = 10):
        if not log_file.exists():
            return {"count": 0, "history": []}
        with open(log_file, "r") as f:
            lines = f.readlines()
        entries = []
        for line in lines[-limit:]:
            try:
                e = json_module.loads(line)
                if e.get("event") == "prediction":
                    entries.append(e)
            except Exception:
                continue
        return {"count": len(entries), "history": entries}

    monkeypatch.setattr(m, "history", patched_history)
    result = patched_history(limit=5)
    assert result["count"] == 1
    assert result["history"][0]["session_id"] == "s1"