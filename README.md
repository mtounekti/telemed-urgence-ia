# 🏥 Telemed Urgence IA

> Système de diagnostic assisté et tri d'urgence multimodal en télémédecine  
> Projet de fin de formation — Promo Upskilling Atlas CISIA — Mars 2026

## Objectif

Classification supervisée à 3 classes du degré d'urgence d'une demande entrante :
- `0` — Non urgent
- `1` — Urgence relative  
- `2` — Urgence vitale ⚠️

## Stack

- **ML** : scikit-learn, XGBoost, LightGBM
- **NLP** : TF-IDF, spaCy (fr)
- **API** : FastAPI
- **UI** : Streamlit
- **Tracking** : MLflow
- **Infra** : Docker, GitHub Actions

## Structure

telemed-urgence-ia/
├── data/
│   ├── raw/          # Données brutes (non committées)
│   └── processed/    # Données prétraitées
├── notebooks/        # EDA, expérimentations
├── src/
│   ├── preprocessing/
│   ├── models/
│   ├── api/
│   └── ui/
├── tests/
├── mlflow/
└── docker/

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
