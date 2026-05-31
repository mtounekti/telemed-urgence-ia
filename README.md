# 🏥 Telemed Urgence IA

> Système de diagnostic assisté et tri d'urgence multimodal en télémédecine  
> Projet de fin de formation — Promo Upskilling Atlas CISIA — Mars 2026

![CI/CD](https://github.com/mtounekti/telemed-urgence-ia/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)
![Docker](https://img.shields.io/badge/Docker-ready-blue)

---

## Objectif

Classification supervisée à 3 classes du degré d'urgence d'une demande entrante :

| Classe | Label | Description |
|--------|-------|-------------|
| `0` | Non urgent | Pas de caractère urgent |
| `1` | Urgence relative | Nécessite une consultation rapide |
| `2` | Urgence vitale ⚠️ | Action immédiate requise |

---

## 📊 Résultats

| Modèle | Scénario | Accuracy | F1-weighted | Recall Classe 2 |
|--------|----------|----------|-------------|-----------------|
| **LogisticRegression** | **S1 Multimodal** | **94.94%** | **94.92%** | **93.22%** ⭐ |
| LightGBM | S1 Multimodal | 95.34% | 95.32% | 90.85% |
| XGBoost | S1 Multimodal | 95.49% | 95.47% | 89.83% |
| LogisticRegression | S3 NLP seul | 91.07% | 91.16% | 90.85% |
| LogisticRegression | S4 Tabulaire seul | 85.86% | 85.85% | 78.64% |

> ⚠️ La métrique prioritaire est le **Recall de la classe 2** (urgence vitale) — une erreur sur ce cas est critique.

---

## Architecture

```
[ Streamlit UI :8501 ] ──► [ FastAPI :8000 ] ──► [ LogisticRegression ]
                                  │
                            [ Logs JSON ]
                                  │
                            [ MLflow UI :5000 ]
```

---

## Structure du projet

```
telemed-urgence-ia/
├── .github/workflows/    # CI/CD GitHub Actions
├── data/
│   ├── raw/              # Données brutes (non committées)
│   └── processed/        # Scénarios preprocessés
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   └── 03_Modelisation.ipynb
├── src/
│   ├── api/main.py       # FastAPI
│   ├── models/           # Modèles entraînés
│   └── ui/app.py         # Streamlit
├── tests/
│   └── test_api.py       # Tests unitaires
├── Dockerfile
├── Dockerfile.streamlit
├── docker-compose.yml
└── requirements.txt
```

---

## Installation & Lancement

### 1. Cloner le repo

```bash
git clone https://github.com/mtounekti/telemed-urgence-ia.git
cd telemed-urgence-ia
```

### 2. Environnement virtuel

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download fr_core_news_sm
```

### 3. Lancer l'API

```bash
uvicorn src.api.main:app --reload --port 8000
```

### 4. Lancer l'interface

```bash
streamlit run src/ui/app.py
```

### 5. Avec Docker

```bash
docker-compose up --build
```

---

## 🔌 API Routes

| Méthode | Route | Description |
|---------|-------|-------------|
| `GET` | `/health` | Santé de l'API |
| `POST` | `/predict` | Prédiction niveau d'urgence |
| `POST` | `/retrain` | Réentraînement monitoré |
| `GET` | `/history` | Historique des inférences |
| `GET` | `/docs` | Documentation Swagger |

### Exemple `/predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

Réponse :

```json
{
  "niveau_urgence": 2,
  "label": "Urgence vitale ⚠️",
  "probabilites": {
    "non_urgent": 0.0,
    "urgence_relative": 0.0424,
    "urgence_vitale": 0.9576
  },
  "timestamp": "2026-05-31T..."
}
```

---

## Tests

```bash
pytest tests/ -v
```

```
tests/test_api.py::test_health                  PASSED
tests/test_api.py::test_predict_returns_valid_structure  PASSED
tests/test_api.py::test_predict_critique        PASSED
tests/test_api.py::test_predict_non_urgent      PASSED
tests/test_api.py::test_predict_invalid_input   PASSED
tests/test_api.py::test_history                 PASSED
6 passed in 1.78s
```

---

## 📈 MLflow

```bash
mlflow ui --backend-store-uri mlflow/
# http://127.0.0.1:5000
```

---

## ⚖️ Éthique & RGPD

- `patient_id` supprimé dès le chargement (identifiant direct)
- Variables sensibles isolées dans le **Scénario 2** (sexe, zone_vie, antécédents)
- Données de santé = article 9 RGPD → base légale obligatoire
- Toutes les inférences sont journalisées (traçabilité)
- Modèle optimisé pour minimiser les faux négatifs sur la classe 2

---

## Auteur

**Marou Tounekti** — Promo Upskilling Atlas CISIA — Mars 2026 :)