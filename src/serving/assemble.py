"""
FraudGuard — Phase 7 Part B core: raw transaction -> model feature vector.

This is where training/serving skew would live, so it reuses Phase 2/2.5's EXACT
functions (persisted encoders, the same missingness/sentinel logic, the same uid
construction) and assembles the vector in ``feature_manifest.json``'s exact column
order. The Part C consistency test asserts this reproduces the offline
``features.parquet`` vector column-for-column.

Order of operations mirrors Phase 2 precisely:
  1. missingness signals (has_identity, v_missing_count) from RAW values,
  2. aggregate lookups by RAW card1/uid (before card1 is encoded),
  3. log1p, then
  4. categorical encoding (overwrites the raw categorical columns), then
  5. V-column -999 sentinel fill (after v_missing_count is counted).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_engineering import (
    V_FILL_SENTINEL,
    apply_frequency,
    apply_label,
)
from src.modeling import MANIFEST_JSON, MODELS_DIR, manifest_features
from src.serving.feature_store import FeatureStore
from src.uid_features import build_uid

ENCODERS_JSON = MODELS_DIR / "phase2_encoders.json"
RAW_SCHEMA_JSON = MODELS_DIR / "serving_raw_schema.json"

_V_RE = re.compile(r"V\d+")


class FeatureAssembler:
    def __init__(self, encoders: dict, raw_schema: dict, manifest: list[str],
                 store: FeatureStore):
        self.freq = encoders["frequency"]
        self.label = encoders["label"]
        self.raw_schema = raw_schema
        self.manifest = manifest
        self.store = store
        self._id_cols = [c for c in raw_schema if c.startswith("id_")] + [
            c for c in ("DeviceType", "DeviceInfo") if c in raw_schema]
        self._v_cols = [c for c in raw_schema if _V_RE.fullmatch(c)]

    @classmethod
    def load(cls, store: FeatureStore) -> "FeatureAssembler":
        return cls(
            encoders=json.loads(ENCODERS_JSON.read_text()),
            raw_schema=json.loads(RAW_SCHEMA_JSON.read_text()),
            manifest=manifest_features(),
            store=store,
        )

    # ---- raw dict -> typed 1-row frame ----------------------------------- #
    def _raw_frame(self, raw: dict) -> pd.DataFrame:
        df = pd.DataFrame([{c: raw.get(c, None) for c in self.raw_schema}])
        for c, dt in self.raw_schema.items():
            if dt in ("str", "string", "object"):
                df[c] = df[c].astype("string")           # None -> <NA>
            else:
                num = pd.to_numeric(df[c], errors="coerce")
                try:
                    df[c] = num.astype(dt)               # exact merged dtype (matters for
                except (ValueError, TypeError):          # string-key encoding fidelity)
                    df[c] = num
        return df

    # ---- the assembly ---------------------------------------------------- #
    def assemble(self, raw: dict) -> pd.DataFrame:
        df = self._raw_frame(raw)

        # 1-3. engineered features computed from RAW (before card1 is encoded),
        #      collected into one frame so the wide df isn't fragmented.
        day = (df["TransactionDT"] // 86_400).astype("int32")
        d1_norm = (df["D1"] - day).astype("float32")
        uid = build_uid(df["card1"], df["addr1"], d1_norm).iloc[0]
        ustate = self.store.lookup_uid(uid)
        eng = pd.DataFrame({
            "has_identity": df[self._id_cols].notna().any(axis=1).to_numpy(),
            "v_missing_count": df[self._v_cols].isna().sum(axis=1).astype("int16").to_numpy(),
            "TransactionAmt_log1p": np.log1p(df["TransactionAmt"]).astype("float32").to_numpy(),
            "uid_prior_count": np.int64(ustate["uid_prior_count"]),
            "uid_amt_expanding_mean": np.float32(ustate["uid_amt_expanding_mean"]),
            "uid_amt_expanding_std": np.float32(ustate["uid_amt_expanding_std"]),
            "card1_prior_count": np.int64(self.store.lookup_card1(df["card1"].iloc[0])),
        })

        # 4. categorical encoding (persisted maps — NEVER refit here)
        for col, fmap in self.freq.items():
            if col in df.columns:
                df[col] = apply_frequency(df[col], fmap)
        for col, lmap in self.label.items():
            if col in df.columns:
                df[col] = apply_label(df[col], lmap)

        # 5. V-column sentinel (after v_missing_count already counted)
        df[self._v_cols] = df[self._v_cols].fillna(V_FILL_SENTINEL)

        df = pd.concat([df.reset_index(drop=True), eng], axis=1)
        missing = [c for c in self.manifest if c not in df.columns]
        if missing:
            raise RuntimeError(f"Assembled frame missing manifest columns: {missing[:5]}")
        return df[self.manifest]

    def uid_for(self, raw: dict) -> str:
        """Expose the reconstructed uid (handy for diagnostics / unknown-key checks)."""
        df = self._raw_frame(raw)
        day = (df["TransactionDT"] // 86_400).astype("int32")
        d1_norm = (df["D1"] - day).astype("float32")
        return build_uid(df["card1"], df["addr1"], d1_norm).iloc[0]
