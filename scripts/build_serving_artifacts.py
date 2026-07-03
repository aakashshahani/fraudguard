"""
Generate the small, committed serving artifacts + the CI skew-test fixture.

Runs locally (needs the full engineered data once); its outputs are tiny and are
committed so serving + the Part C consistency test run in CI with NO Kaggle data:

  models/phase2_encoders.json         persisted frequency/label maps (Phase 2, verbatim)
  models/serving_raw_schema.json      merged-frame dtypes (for exact string-key encoding)
  tests/fixtures/serving_fixture.json 10 test rows: raw fields + precomputed feature vector
  tests/fixtures/feature_store_snapshot.json  uid/card1 state for those 10 rows

    python scripts/build_serving_artifacts.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.data_prep import MERGED_PARQUET, SPLIT_INDICES_PARQUET
from src.feature_engineering import FEATURES_PARQUET, fit_encoders
from src.modeling import manifest_features
from src.serving.assemble import ENCODERS_JSON, RAW_SCHEMA_JSON
from src.serving.feature_store import UID_AGG_COLS
from src.uid_features import build_uid

FIX_DIR = None  # set below
N_FIXTURE = 10


def _jsonify(v):
    if isinstance(v, float) and np.isnan(v):
        return None
    if v is None or (hasattr(pd, "isna") and np.ndim(v) == 0 and pd.isna(v)):
        return None
    if isinstance(v, (np.generic,)):
        return v.item()
    return v


def main() -> None:
    from pathlib import Path
    fix_dir = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
    fix_dir.mkdir(parents=True, exist_ok=True)

    merged = pd.read_parquet(MERGED_PARQUET)
    splits = pd.read_parquet(SPLIT_INDICES_PARQUET)[["TransactionID", "split"]]
    merged = merged.merge(splits, on="TransactionID", how="left")

    # 1. encoders (fit on train — deterministic, identical to Phase 2)
    encoders = fit_encoders(merged)
    ENCODERS_JSON.write_text(json.dumps(encoders))
    print(f"wrote {ENCODERS_JSON.name}: "
          f"{len(encoders['frequency'])} freq + {len(encoders['label'])} label maps")

    # 2. raw schema (merged dtypes, excluding the target)
    raw_schema = {c: str(merged[c].dtype) for c in merged.columns if c not in ("isFraud", "split")}
    RAW_SCHEMA_JSON.write_text(json.dumps(raw_schema))
    print(f"wrote {RAW_SCHEMA_JSON.name}: {len(raw_schema)} raw columns")

    # 3. pick 10 TEST rows with DISTINCT card1 AND uid, so each key's snapshot
    #    holds that row's own point-in-time state (no cross-fixture overwrite).
    feats = pd.read_parquet(FEATURES_PARQUET)
    manifest = manifest_features()
    raw_cols = [c for c in merged.columns if c not in ("isFraud", "split")]

    test_ids = feats.loc[feats["split"] == "test", "TransactionID"]
    cand = (merged[merged["TransactionID"].isin(test_ids)][["TransactionID", "card1", "addr1"]]
            .merge(feats[["TransactionID", "D1_normalized", "isFraud"]], on="TransactionID"))
    cand["uid"] = build_uid(cand["card1"], cand["addr1"], cand["D1_normalized"])

    picked, seen_card1, seen_uid = [], set(), set()

    def pick_from(subset, max_take):
        taken = 0
        for r in subset.itertuples():
            if taken >= max_take or len(picked) >= N_FIXTURE:
                break
            if int(r.card1) in seen_card1 or r.uid in seen_uid:
                continue
            seen_card1.add(int(r.card1)); seen_uid.add(r.uid); picked.append(int(r.TransactionID))
            taken += 1

    pick_from(cand[cand.isFraud == 1], 4)          # up to 4 frauds
    pick_from(cand[cand.isFraud == 0], N_FIXTURE)  # fill with legit

    fixtures, uid_state, card1_state = [], {}, {}
    for tid in picked:
        mrow = merged.loc[merged["TransactionID"] == tid].iloc[0]
        frow = feats.loc[feats["TransactionID"] == tid].iloc[0]

        raw = {c: _jsonify(mrow[c]) for c in raw_cols}
        expected = {c: _jsonify(frow[c]) for c in manifest}
        fixtures.append({"transaction_id": int(tid), "raw": raw, "expected_features": expected})

        uid = build_uid(pd.Series([mrow["card1"]]), pd.Series([mrow["addr1"]]),
                        pd.Series([frow["D1_normalized"]])).iloc[0]
        uid_state[uid] = {k: (int(frow[k]) if k == "uid_prior_count" else float(frow[k]))
                          for k in UID_AGG_COLS}
        card1_state[str(int(mrow["card1"]))] = int(frow["card1_prior_count"])

    (fix_dir / "serving_fixture.json").write_text(json.dumps(fixtures, indent=2))
    (fix_dir / "feature_store_snapshot.json").write_text(
        json.dumps({"uid_state": uid_state, "card1_state": card1_state}, indent=2))
    print(f"wrote fixture ({len(fixtures)} rows) + store snapshot ({len(uid_state)} uids)")


if __name__ == "__main__":
    main()
