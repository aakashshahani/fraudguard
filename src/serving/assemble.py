"""
FraudGuard — Phase 7 Part B core: raw transaction -> model feature vector.

This is where training/serving skew would live, so it reuses Phase 2/2.5's EXACT
artifacts and rules (persisted encoders keyed the same way as apply_frequency /
apply_label, the same missingness/sentinel logic, the same uid construction) and assembles the vector in ``feature_manifest.json``'s exact column
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

import numpy as np
import pandas as pd

from src.feature_engineering import (
    MISSING_TOKEN,
    UNKNOWN_LABEL,
    V_FILL_SENTINEL,
)
from src.modeling import MODELS_DIR, manifest_features
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
        # Coerce every numeric field in ONE vectorized call (and every string
        # field in another), then build the frame once. Converting ~440 one-row
        # columns individually dominated per-request latency.
        str_cols = [c for c, dt in self.raw_schema.items() if dt in ("str", "string", "object")]
        num_cols = [c for c in self.raw_schema if c not in set(str_cols)]
        strs = pd.Series([raw.get(c) for c in str_cols], dtype=object).astype("string")  # None -> <NA>
        nums = pd.to_numeric(pd.Series([raw.get(c) for c in num_cols], dtype=object),
                             errors="coerce").astype("float64").to_numpy()

        columns: dict[str, object] = {}
        for c, v in zip(str_cols, strs):
            columns[c] = pd.array([v], dtype="string")
        for c, x in zip(num_cols, nums):
            dt = np.dtype(self.raw_schema[c])
            if dt.kind in "iu":
                info = np.iinfo(dt)
                # Missing values can't be ints, and an out-of-range value must
                # never be narrowed: pandas wraps silently (card1=999999 as int16
                # becomes 16959 — a real card), which would score an unknown card
                # with another card's encodings and history. Keep both as float.
                if np.isnan(x) or not (info.min <= x <= info.max):
                    columns[c] = np.array([x], dtype="float64")
                    continue
            columns[c] = np.array([x], dtype=dt)   # exact merged dtype (matters for
                                                   # string-key encoding fidelity)
        return pd.DataFrame(columns)[list(self.raw_schema)]

    # ---- the assembly ---------------------------------------------------- #
    def assemble(self, raw: dict) -> pd.DataFrame:
        return self.assemble_with_uid(raw)[0]

    def assemble_with_uid(self, raw: dict) -> tuple[pd.DataFrame, str]:
        """The model vector plus the reconstructed uid, from ONE pass over the
        raw frame (serving needs both; typing ~440 columns twice per request
        was pure overhead)."""
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

        # 4. categorical encoding (persisted maps — NEVER refit here). Same keys as
        #    apply_frequency/apply_label (_as_cat: astype("string"), NaN -> token),
        #    computed for every encoded column in one call and looked up in the
        #    dicts directly; a per-column Series.map was most of the request time.
        enc_cols = [c for c in (*self.freq, *self.label) if c in df.columns]
        keys = df[enc_cols].astype("string").fillna(MISSING_TOKEN).iloc[0]
        encoded = {c: np.array([self.freq[c].get(keys[c], 0)], dtype="int64")
                   for c in self.freq if c in df.columns}
        encoded.update({c: np.array([self.label[c].get(keys[c], UNKNOWN_LABEL)], dtype="int64")
                        for c in self.label if c in df.columns})

        # 5. V-column sentinel (after v_missing_count already counted)
        filled = df[self._v_cols].fillna(V_FILL_SENTINEL)

        rest = [c for c in df.columns if c not in encoded and c not in set(self._v_cols)]
        df = pd.concat([df[rest].reset_index(drop=True), pd.DataFrame(encoded),
                        filled.reset_index(drop=True), eng], axis=1)
        missing = [c for c in self.manifest if c not in df.columns]
        if missing:
            raise RuntimeError(f"Assembled frame missing manifest columns: {missing[:5]}")
        return df[self.manifest], uid

    def uid_for(self, raw: dict) -> str:
        """Expose the reconstructed uid (handy for diagnostics / unknown-key checks)."""
        df = self._raw_frame(raw)
        day = (df["TransactionDT"] // 86_400).astype("int32")
        d1_norm = (df["D1"] - day).astype("float32")
        return build_uid(df["card1"], df["addr1"], d1_norm).iloc[0]
