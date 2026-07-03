"""
FraudGuard — Phase 7 Part A: in-memory feature store.

Simulates the "current production state" a real feature store (Feast/Tecton +
Redis) would maintain live: for each ``card1`` and each ``uid``, the LATEST known
expanding-aggregate state as of the dataset's final timestamp. At serving time a
new transaction looks up its key's current state rather than recomputing history.

Two constructors:
- ``from_parquet`` — production build over the full engineered dataset (needs the
  raw card1/addr1 from train_merged to reconstruct the uid key, exactly as
  Phase 2.5 did).
- ``from_snapshot`` — load a tiny committed snapshot (used by CI / the skew test,
  and for lightweight deployments), so nothing here needs the full Kaggle data.

Unknown keys (no prior history) fall back to the Phase 2.5 insufficient-history
sentinel convention — never a crash, never a silent zero for the amount stats.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.uid_features import UID_SENTINEL, build_uid

UID_AGG_COLS = ["uid_prior_count", "uid_amt_expanding_mean", "uid_amt_expanding_std"]

# Sentinel state for a key with no history (matches Phase 2.5: 0 priors -> mean/std
# undefined -> -999; count 0).
UNKNOWN_UID_STATE = {
    "uid_prior_count": 0,
    "uid_amt_expanding_mean": float(UID_SENTINEL),
    "uid_amt_expanding_std": float(UID_SENTINEL),
}
UNKNOWN_CARD1_COUNT = 0


class FeatureStore:
    def __init__(self, uid_state: dict[str, dict], card1_state: dict[str, int]):
        self._uid = uid_state
        self._card1 = card1_state

    # ---- constructors ---------------------------------------------------- #
    @classmethod
    def from_parquet(cls, features_parquet: Path, merged_parquet: Path) -> "FeatureStore":
        """Production build: latest state per key over the full engineered dataset."""
        feats = pd.read_parquet(
            features_parquet,
            columns=["TransactionID", "TransactionDT", "D1_normalized",
                     "card1_prior_count", *UID_AGG_COLS],
        )
        raw = pd.read_parquet(merged_parquet, columns=["TransactionID", "card1", "addr1"])
        df = feats.merge(raw, on="TransactionID", how="left")
        df["uid"] = build_uid(df["card1"], df["addr1"], df["D1_normalized"])
        return cls._build(df)

    @classmethod
    def _build(cls, df: pd.DataFrame) -> "FeatureStore":
        df = df.sort_values("TransactionDT")
        uid_latest = df.groupby("uid", sort=False).tail(1)
        uid_state = {
            row.uid: {k: (int(getattr(row, k)) if k == "uid_prior_count"
                          else float(getattr(row, k))) for k in UID_AGG_COLS}
            for row in uid_latest.itertuples()
        }
        card1_latest = df.groupby("card1", sort=False).tail(1)
        card1_state = {str(row.card1): int(row.card1_prior_count)
                       for row in card1_latest.itertuples()}
        return cls(uid_state, card1_state)

    @classmethod
    def from_snapshot(cls, path: Path) -> "FeatureStore":
        snap = json.loads(Path(path).read_text())
        return cls(snap["uid_state"], snap["card1_state"])

    # ---- persistence ----------------------------------------------------- #
    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(
            {"uid_state": self._uid, "card1_state": self._card1}, indent=2))

    # ---- lookups --------------------------------------------------------- #
    def lookup_uid(self, uid: str) -> dict:
        """Latest uid aggregate state; sentinel state if the uid has no history."""
        return dict(self._uid.get(uid, UNKNOWN_UID_STATE))

    def lookup_card1(self, card1) -> int:
        """Latest card1 prior-count; 0 if this card has never been seen."""
        return int(self._card1.get(str(card1), UNKNOWN_CARD1_COUNT))

    def has_uid(self, uid: str) -> bool:
        return uid in self._uid

    def __len__(self) -> int:
        return len(self._uid)
