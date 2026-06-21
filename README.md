# Telemed Urgence IA

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
| `2` | Urgence vitale | Action immédiate requise |

La métrique prioritaire est le **recall de la classe 2** : une urgence vitale manquée est une erreur bien plus grave qu'un cas non urgent sur-évalué. Toute la chaîne de décision — choix du modèle, métriques, seuil de classification — est pilotée par cette asymétrie de coût.

---

## Résultats

### Comparaison initiale des modèles (scénario complet)

| Modèle | Accuracy | F1 pondéré | Recall classe 2 | Erreurs critiques |
|--------|----------|-------------|------------------|--------------------|
| LogisticRegression | 94.94% | 94.92% | 93.22% | 20 |
| RandomForest | 94.05% | 94.04% | 90.85% | 27 |
| XGBoost | 95.49% | 95.47% | 89.83% | 30 |

### Après optimisation du seuil de décision (classe 2)

| Modèle | Seuil retenu | Accuracy | F1 pondéré | Recall classe 2 | Erreurs critiques |
|--------|--------------|----------|-------------|------------------|--------------------|
| LogisticRegression | 0.15 | 92.61% | 92.66% | 96.61% | 10 |
| RandomForest | 0.10 | 91.47% | 91.69% | 98.98% | 3 |
| **XGBoost** | **0.01** | **93.80%** | **93.88%** | **98.98%** | **3** |

Le modèle final retenu est **XGBoost avec seuil de classe 2 ajusté à 0.01**. À recall et erreurs critiques strictement identiques à RandomForest, XGBoost conserve la meilleure accuracy et le meilleur F1 pondéré parmi les modèles atteignant ce niveau de sécurité.

Détail de la démarche : `notebooks/03_Modelisation.ipynb`, `notebooks/04_Hyperparameter_Tuning.ipynb` et `analyses/synthese_ajustement_seuil_decision.md`

---

## Architecture

```
[ Streamlit UI :8501 ] ──► [ FastAPI :8000 ] ──► [ XGBoost + seuil optimisé ]
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
│   └── ci.yml                          # CI/CD GitHub Actions
├── analyses/                           # Rapports générés par les scripts src/
│   ├── synthese_modeles_baseline.md
│   ├── synthese_scenarios_comparatifs.md
│   ├── synthese_ajustement_seuil_decision.md
│   └── synthese_vocabulaire_discriminant.md
├── data/
│   ├── raw/                            # Données brutes (non versionnées)
│   └── processed/                      # Scénarios prétraités
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Modelisation.ipynb
│   ├── 04_Hyperparameter_Tuning.ipynb
│   └── 05_Synthese_Finale.ipynb        # Journal de bord et démonstration
├── src/
│   ├── api/
│   │   ├── main.py                     # Routes FastAPI
│   │   ├── schemas.py                  # Modèles Pydantic
│   │   ├── predict.py                  # Logique de prédiction + interprétation
│   │   ├── database.py                 # Stockage SQLite du feedback
│   │   └── logger.py                   # Logging structuré JSON
│   ├── models/                         # Artefacts ML (preprocessors)
│   ├── ui/app.py                       # Interface Streamlit
│   ├── compare_models.py               # Comparaison initiale des modèles
│   ├── compare_scenarios.py            # Comparaison des 4 scénarios
│   ├── optimize_critical_errors.py     # Recherche du seuil optimal
│   └── text_interpretation.py          # Vocabulaire discriminant par classe
├── models/
│   └── triage_model_optimized.joblib   # Modèle final servi par l'API
├── logs/
│   └── inference.log                   # Logs structurés JSON
├── tests/
│   ├── fixtures.py                     # Données de test (patients types, cas limites)
│   └── test_api.py                     # Tests unitaires (fonctionnels, bornes, mocking)
├── docker/
│   ├── prometheus.yml
│   └── grafana/
│       ├── provisioning/
│       │   ├── datasources/prometheus.yml
│       │   └── dashboards/dashboard-provider.yml
│       └── dashboards/telemed-dashboard.json
├── Dockerfile
├── Dockerfile.streamlit
├── docker-compose.yml
└── requirements.txt
```

---

## Installation et lancement

### 1. Cloner le dépôt

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
# Clé API pour sécuriser /retrain (optionnel, défaut : telemed-secret-key)
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

### 6. Avec Docker (API + UI + monitoring)

```bash
docker-compose up --build
```

---

## API — Routes

| Méthode | Route | Authentification | Description |
|---------|-------|-------------------|--------------|
| `GET` | `/health` | — | Santé de l'API et modèle actif |
| `POST` | `/predict` | — | Prédiction du niveau d'urgence |
| `POST` | `/retrain` | Clé API (`X-API-Key`) | Réentraînement monitoré |
| `GET` | `/history` | — | Historique des inférences |
| `POST` | `/feedback` | — | Enregistrement d'un retour utilisateur |
| `GET` | `/feedbacks` | — | Historique des feedbacks |
| `GET` | `/metrics` | — | Métriques au format Prometheus |
| `GET` | `/docs` | — | Documentation Swagger interactive |

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
  "label": "Urgence vitale",
  "probabilites": {
    "non_urgent": 0.0,
    "urgence_relative": 0.0395,
    "urgence_vitale": 0.9605
  },
  "model_name": "xgboost",
  "threshold_class_2": 0.01,
  "interpretation": [
    "La description mentionne une douleur thoracique, signal souvent associé à un risque élevé.",
    "La description mentionne un essoufflement, ce qui peut orienter vers une situation plus urgente.",
    "Probabilités estimées par le modèle : classe 0: 0.0%, classe 1: 4.0%, classe 2: 96.1%. Classe retenue : 2.",
    "Le modèle utilise un seuil plus prudent pour la classe 2 afin de limiter les urgences vitales manquées."
  ],
  "timestamp": "2026-06-17T10:45:01+00:00",
  "duration_ms": 35.8
}
```

### Exemple `/retrain`

```bash
curl -X POST http://localhost:8000/retrain \
  -H "X-API-Key: votre-cle-secrete"
```

Réentraîne un XGBoost sur les données disponibles, recherche automatiquement le meilleur seuil de classe 2, puis compare au modèle en production. Le nouveau modèle ne remplace l'ancien que s'il égale ou dépasse le recall classe 2 actuel.

---

## Monitoring

| Service | URL | Rôle |
|---------|-----|------|
| Prometheus | `:9090` | Collecte des métriques (requêtes, latence, prédictions par classe) |
| Grafana | `:3000` | Tableau de bord temps réel |
| Uptime Kuma | `:3001` | Surveillance de disponibilité |

La source de données Prometheus et le dashboard Grafana sont provisionnés automatiquement au démarrage du conteneur (`docker/grafana/provisioning/`), sans configuration manuelle requise.

---

## Feedback utilisateur

Après chaque prédiction, l'utilisateur peut soumettre un retour via l'interface Streamlit :

- Prédiction jugée correcte ou incorrecte
- Niveau réel observé (optionnel)
- Commentaire libre (optionnel)

Les feedbacks sont stockés en SQLite (`data/feedback.db`) et consultables via `/feedbacks`.

---

## Logging structuré

Chaque inférence est journalisée dans `logs/inference.log` au format JSON :

```json
{
  "event": "prediction",
  "timestamp": "2026-06-17T10:45:01+00:00",
  "session_id": "session-001",
  "user_id": null,
  "duration_ms": 35.8,
  "input": { "sexe": "F", "age": 65 },
  "output": { "niveau_urgence": 2, "label": "Urgence vitale" }
}
```

---

## Tests

```bash
pytest tests/ -v -m "not slow"
```

Les tests sont organisés en trois familles :

- **Tests fonctionnels de base** : santé de l'API, structure de la réponse, distinction urgence vitale et non urgente, historique
- **Tests de validation des bornes physiologiques** : chaque variable contrainte (âge, fréquence cardiaque, tension, température, saturation, etc.) est testée individuellement contre une valeur hors limite via `tests/fixtures.py`, qui suit strictement la structure du dataset
- **Tests avec mocking** (`monkeypatch`) : simulent un modèle absent (503), une erreur d'inférence inattendue (500), ou isolent la route `/predict` du vrai modèle chargé pour vérifier uniquement la construction de la réponse

```
test_health                                   PASSED
test_predict_structure                        PASSED
test_predict_critique                         PASSED
test_predict_non_urgent                       PASSED
test_predict_invalid_input                    PASSED
test_predict_with_session_id                  PASSED
test_predict_limite_valide_acceptee           PASSED
test_predict_urgence_relative                 PASSED
test_predict_rejects_invalid_field            PASSED (x15, paramétré)
test_predict_returns_503_when_model_missing   PASSED
test_predict_returns_500_on_unexpected_error  PASSED
test_predict_mocked_response_structure        PASSED
test_retrain_without_key                      PASSED
test_retrain_with_wrong_key                   PASSED
test_retrain_with_correct_key                 PASSED
test_retrain_wrong_key_with_monkeypatch_env   PASSED
test_history                                  PASSED
test_history_with_isolated_log_file           PASSED
```

Fichier de fixtures : `tests/fixtures.py` — patients types (critique, non urgent, urgence relative, cas limite valide) et 15 cas de violation de bornes physiologiques, tous construits selon la structure exacte du `dataset_telemed.csv`.

---

## MLflow

```bash
mlflow ui --backend-store-uri mlflow/
# http://127.0.0.1:5000
```

Plus de 25 expériences tracées : comparaison des modèles, des 4 scénarios, du tuning d'hyperparamètres et des seuils de décision.

---

## Les quatre scénarios de données

| Scénario | Description | Recall classe 2 |
|----------|--------------|------------------|
| Multimodal complet | Données tabulaires et texte | 93.22% |
| Sans variables sensibles | Retrait de sexe et zone_vie | 93.22% |
| Texte seul | `description_symptomes` uniquement | 91.53% |
| Tabulaire seul | Constantes vitales et âge | 78.64% |

Le scénario sans variables sensibles obtient un recall identique au scénario complet, ce qui justifie le retrait de ces variables pour limiter le risque de biais, sans perte de performance mesurable.

Détail : `analyses/synthese_scenarios_comparatifs.md`

---

## Éthique et RGPD

- `patient_id` supprimé dès le chargement (identifiant direct)
- Variables sensibles isolées et testées séparément (scénario sans variables sensibles)
- Les données de santé relèvent de l'article 9 du RGPD : minimisation des données conservées
- Chaque inférence est journalisée avec horodatage UTC, session et durée
- Le modèle est optimisé pour minimiser les faux négatifs sur la classe 2 (urgence vitale)
- La route `/retrain` est protégée par une clé API
- Le système reste un outil d'aide à la décision : la décision médicale finale appartient au professionnel de santé

---

## Documentation complémentaire

- `notebooks/05_Synthese_Finale.ipynb` : journal de bord chronologique et démonstration d'inférence
- `analyses/` : rapports détaillés générés par les scripts `src/`

---

## Auteur

**Maroua Tounekti** — Promo Upskilling Atlas CISIA — Mars 2026