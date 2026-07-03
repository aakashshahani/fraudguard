"""
FraudGuard — Phase 7 Part B: FastAPI scoring service.

POST /predict accepts the RAW pre-engineering transaction fields a real upstream
system would send. The service looks up current aggregate state from the in-memory
feature store, applies Phase 2's persisted encoders + missingness/sentinel logic,
assembles the feature vector in the manifest's exact order, and scores it with
Phase 5's frozen model + threshold. Nothing is retrained or re-tuned.

Unknown card1/uid (no history) fall back to the Phase 2.5 insufficient-history
sentinel via the feature store — never a crash, never a silent zero.

Run
---
    uvicorn src.serving.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.data_prep import MERGED_PARQUET
from src.evaluation import PHASE5_THRESHOLD_JSON
from src.feature_engineering import FEATURES_PARQUET
from src.modeling import WINNER_ARTIFACT
from src.serving.assemble import FeatureAssembler
from src.serving.feature_store import FeatureStore

FIXTURE_SNAPSHOT = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "feature_store_snapshot.json"

log = logging.getLogger("fraudguard.api")
STATE: dict[str, Any] = {}


def _build_store() -> FeatureStore:
    """Full-data store in production; the tiny committed snapshot otherwise (CI/demo)."""
    import os
    if os.environ.get("FRAUDGUARD_STORE") != "snapshot" and (
            FEATURES_PARQUET.exists() and MERGED_PARQUET.exists()):
        log.info("Building feature store from full engineered data")
        return FeatureStore.from_parquet(FEATURES_PARQUET, MERGED_PARQUET)
    log.info("Loading committed feature-store snapshot")
    return FeatureStore.from_snapshot(FIXTURE_SNAPSHOT)


def build_state() -> dict:
    store = _build_store()
    return {
        "model": joblib.load(WINNER_ARTIFACT),
        "threshold": float(json.loads(PHASE5_THRESHOLD_JSON.read_text())["threshold"]),
        "assembler": FeatureAssembler.load(store),
        "store": store,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE.update(build_state())
    log.info("Loaded model + threshold %.4f + store (%d uids)",
             STATE["threshold"], len(STATE["store"]))
    yield
    STATE.clear()


app = FastAPI(title="FraudGuard", version="1.0", lifespan=lifespan)


class PredictResponse(BaseModel):
    fraud_probability: float = Field(..., ge=0, le=1)
    decision: int = Field(..., description="1 = block (fraud), 0 = allow")
    threshold: float
    uid_known: bool = Field(..., description="False = no prior history for this card/uid (sentinel used)")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "store_uids": len(STATE.get("store", []) or [])}


@app.post("/predict", response_model=PredictResponse)
def predict(transaction: dict) -> PredictResponse:
    asm: FeatureAssembler = STATE["assembler"]
    vector = asm.assemble(transaction)
    prob = float(STATE["model"].predict_proba(vector)[:, 1][0])
    thr = STATE["threshold"]
    uid = asm.uid_for(transaction)
    return PredictResponse(
        fraud_probability=prob,
        decision=int(prob >= thr),
        threshold=thr,
        uid_known=STATE["store"].has_uid(uid),
    )
