# FraudGuard serving image — single FastAPI service, no external infra.
FROM python:3.12-slim

WORKDIR /app

# Serving deps only (lean image; the full requirements.txt is for training/analysis).
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

# Application code + the small committed artifacts the service needs at runtime.
COPY src/ src/
COPY models/phase4_winner.joblib models/phase5_threshold.json \
     models/phase2_encoders.json models/serving_raw_schema.json models/
COPY data/processed/feature_manifest.json data/processed/feature_manifest.json
COPY tests/fixtures/feature_store_snapshot.json tests/fixtures/feature_store_snapshot.json

ENV PYTHONPATH=/app
# Standalone by default: serve from the committed feature-store snapshot. In
# production, mount the full engineered data and unset this to build the live store.
ENV FRAUDGUARD_STORE=snapshot

EXPOSE 8000
CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
