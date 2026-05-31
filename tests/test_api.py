# Tests unitaires

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

# données de test
PATIENT_CRITIQUE = {
    "sexe": "F",
    "age": 65,
    "zone_vie": "U",
    "source": "appel",
    "freq_cardiaque": 130,
    "tension_sys": 190,
    "temp": 39.8,
    "sat_oxygene": 85.0,
    "antecedents": 1,
    "duree_symptomes": 1.0,
    "description_symptomes": "Douleur thoracique intense avec essoufflement sévère"
}

PATIENT_NON_URGENT = {
    "sexe": "H",
    "age": 30,
    "zone_vie": "U",
    "source": "chat",
    "freq_cardiaque": 70,
    "tension_sys": 115,
    "temp": 37.0,
    "sat_oxygene": 99.0,
    "antecedents": 0,
    "duree_symptomes": 48.0,
    "description_symptomes": "Renouvellement ordonnance traitement habituel"
}

# Tests
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model" in data
    assert "version" in data

def test_predict_returns_valid_structure():
    response = client.post("/predict", json=PATIENT_CRITIQUE)
    assert response.status_code == 200
    data = response.json()
    assert "niveau_urgence" in data
    assert "label" in data
    assert "probabilites" in data
    assert data["niveau_urgence"] in [0, 1, 2]

def test_predict_critique():
    response = client.post("/predict", json=PATIENT_CRITIQUE)
    assert response.status_code == 200
    data = response.json()
    # un patient critique doit être classé urgence vitale
    assert data["niveau_urgence"] == 2

def test_predict_non_urgent():
    response = client.post("/predict", json=PATIENT_NON_URGENT)
    assert response.status_code == 200
    data = response.json()
    # un patient non urgent doit être classé 0
    assert data["niveau_urgence"] == 0

def test_predict_invalid_input():
    # sexe invalide
    bad_patient = {**PATIENT_CRITIQUE, "sexe": "X"}
    response = client.post("/predict", json=bad_patient)
    assert response.status_code == 422

def test_history():
    response = client.get("/history")
    assert response.status_code == 200
    data = response.json()
    assert "history" in data