# Dockerfile Telemed Urgence IA

# Image de base Python 3.11 slim (légère)
FROM python:3.11-slim

# Répertoire de travail dans le container
WORKDIR /app

# Copie des dépendances en premier (optimise le cache Docker)
COPY requirements.txt .

# install dépendances
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download fr_core_news_sm

# Copie du code source
COPY src/ ./src/
COPY data/processed/ ./data/processed/

# dossier logs
RUN mkdir -p logs

EXPOSE 8000

# Lancement de l'API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]