# Tests unitaires
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

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

def test_retrain_with_correct_key():
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