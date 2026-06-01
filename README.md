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

Classification supervisée à 3 classes du degré d'urgence d'une demande entrante:

| Classe | Label | Description |
|--------|-------|-------------|
| `0` | Non urgent | Pas de caractère urgent |
| `1` | Urgence relative | Nécessite une consultation rapide |
| `2` | Urgence vitale ⚠️ | Action immédiate requise |

---

## Résultats

| Modèle | Scénario | Accuracy | F1-weighted | Recall Classe 2 |
|--------|----------|----------|-------------|-----------------|
| **LogisticRegression** | **S1 Multimodal** | **94.94%** | **94.92%** | **93.22%** ⭐ |
| LightGBM | S1 Multimodal | 95.34% | 95.32% | 90.85% |
| XGBoost | S1 Multimodal | 95.49% | 95.47% | 89.83% |
| LogisticRegression | S3 NLP seul | 91.07% | 91.16% | 90.85% |
| LogisticRegression | S4 Tabulaire seul | 85.86% | 85.85% | 78.64% |

> ⚠️ La métrique prioritaire est le **Recall de la classe 2** (urgence vitale)

---

## Architecture

```
[ Streamlit UI :8501 ] ──► [ FastAPI :8000 ] ──► [ LogisticRegression ]
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
              [ inference.log ]          [ feedback.db ]
              (JSON structuré)           (SQLite)
                    │
              [ Prometheus :9090 ] ──► [ Grafana :3000 ]
              [ Uptime Kuma :3001 ]
```

---

## Structure du projet

```
telemed-urgence-ia/
├── .github/workflows/
│   └── ci.yml                # CI/CD GitHub Actions
├── data/
│   ├── raw/                  # Données brutes (non committées)
│   └── processed/            # Scénarios preprocessés
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   └── 03_Modelisation.ipynb
├── src/
│   ├── api/
│   │   ├── main.py           # Routes FastAPI
│   │   ├── schemas.py        # Modèles Pydantic
│   │   ├── predict.py        # Logique de prédiction
│   │   └── logger.py         # Logging structuré JSON
│   ├── models/               # Artefacts ML
│   └── ui/app.py             # Interface Streamlit
├── logs/
│   └── inference.log         # Logs structurés JSON
├── tests/
│   └── test_api.py           # 10 tests unitaires
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

### 3. Variables d'environnement

```bash
# Clé API pour sécuriser /retrain (optionnel, défaut: telemed-secret-key)
export RETRAIN_API_KEY=votre-cle-secrete
```

### 4. Lancer l'API

```bash
uvicorn src.api.main:app --reload --port 8000
```

### 5. Lancer l'interface

```bash
streamlit run src/ui/app.py
```

### 6. Avec Docker

```bash
docker-compose up --build
```

---

## 🔌 API Routes

| Méthode | Route | Auth | Description |
|---------|-------|------|-------------|
| `GET` | `/health` | — | Santé de l'API |
| `POST` | `/predict` | — | Prédiction niveau d'urgence |
| `POST` | `/retrain` | `X-API-Key` 🔐 | Réentraînement monitoré |
| `GET` | `/history` | — | Historique des inférences |
| `POST` | `/feedback` | — | Enregistrer un feedback utilisateur |
| `GET` | `/feedbacks` | — | Historique des feedbacks |
| `GET` | `/metrics` | — | Métriques Prometheus |
| `GET` | `/docs` | — | Documentation Swagger |

### Exemple `/predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: session-001" \
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
  "timestamp": "2026-05-31T20:45:01+00:00",
  "duration_ms": 35.8
}
```

### Exemple `/retrain`

```bash
curl -X POST http://localhost:8000/retrain \
  -H "X-API-Key: votre-cle-secrete"
```

---

## Monitoring

| Service | URL | Rôle |
|---------|-----|------|
| Prometheus | :9090 | Collecte métriques (requêtes, latence, prédictions) |
| Grafana | :3000 | Dashboard temps réel |
| Uptime Kuma | :3001 | Uptime + alertes |

---

## Feedback utilisateur

Après chaque prédiction, l'utilisateur peut soumettre un retour via l'interface Streamlit :
- ✅ Prédiction correcte ou incorrecte
- Niveau réel observé (optionnel)
- Commentaire libre (optionnel)

Les feedbacks sont stockés en **SQLite** (`data/feedback.db`) et consultables via `/feedbacks`.

---

## 📋 Logging structuré

Chaque inférence est loggée dans `logs/inference.log` au format JSON :

```json
{
  "event": "prediction",
  "timestamp": "2026-05-31T20:45:01+00:00",
  "session_id": "session-001",
  "user_id": null,
  "duration_ms": 35.8,
  "input": { "sexe": "F", "age": 65, "..." },
  "output": { "niveau_urgence": 2, "label": "Urgence vitale ⚠️" }
}
```

---

## Tests

```bash
pytest tests/ -v
```

```
test_health                    PASSED
test_predict_structure         PASSED
test_predict_critique          PASSED
test_predict_non_urgent        PASSED
test_predict_invalid_input     PASSED
test_predict_with_session_id   PASSED
test_retrain_without_key       PASSED
test_retrain_with_wrong_key    PASSED
test_retrain_with_correct_key  PASSED
test_history                   PASSED
10 passed
```

---

## 📈 MLflow

```bash
mlflow ui --backend-store-uri mlflow/
# http://127.0.0.1:5000
```

---

## 📊 Résultats

### Baseline (hyperparamètres par défaut)

| Modèle | Scénario | Accuracy | F1-weighted | Recall Classe 2 |
|--------|----------|----------|-------------|-----------------|
| LogisticRegression | S1 Multimodal | 94.94% | 94.92% | 93.22% |
| LightGBM | S1 Multimodal | 95.34% | 95.32% | 90.85% |
| XGBoost | S1 Multimodal | 95.49% | 95.47% | 89.83% |
| MLP | S1 Multimodal | 94.44% | 94.43% | 88.47% |

### Après tuning (RandomizedSearchCV — 30 itérations)

| Modèle | Accuracy | F1-weighted | Recall Classe 2 | Gain |
|--------|----------|-------------|-----------------|------|
| **LogisticRegression** | **95.09%** | **95.08%** | **93.22%** ⭐ | ➡️ stable |
| LightGBM | 95.97% | 95.83% | 93.22% | 📈 +2.37% |
| XGBoost | 95.52% | 95.52% | 90.17% | 📈 +0.34% |
| MLP | 95.53% | 95.53% | 92.54% | 📈 +4.07% |

> ⚠️ La métrique prioritaire est le **Recall de la classe 2** (urgence vitale)
> LogisticRegression est retenu comme modèle final : même Recall C2 que LightGBM tuné,
> latence 10x plus faible (~35ms) et interprétabilité totale (Art. 22 RGPD)

---

## ⚖️ Éthique & RGPD

- `patient_id` supprimé dès le chargement (identifiant direct)
- Variables sensibles isolées dans le **Scénario 2** (sexe, zone_vie, antécédents)
- Données de santé = article 9 RGPD → base légale obligatoire
- Toutes les inférences sont journalisées avec horodatage UTC
- Modèle optimisé pour minimiser les faux négatifs sur la classe 2 (urgence vitale)
- `/retrain` protégé par clé API → contrôle d'accès strict

---

## Auteur

**Maroua Tounekti** — Promo Upskilling Atlas CISIA — Mars 2026