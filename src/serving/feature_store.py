"""
FraudGuard — Phase 7 Part A: in-memory feature store.

Simulates the "current production state" a real feature store (Feast/Tecton +
Redis) would maintain live: for each ``card1`` and each ``uid``, the aggregate
state a NEW transaction arriving after the dataset's final timestamp should see —
i.e. over ALL of that key's history, since every stored row is strictly earlier.
At serving time a new transaction looks up its key's state rather than
recomputing history.

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
        """Production build over the full engineered dataset."""
        feats = pd.read_parquet(features_parquet, columns=["TransactionID", "D1_normalized"])
        raw = pd.read_parquet(merged_parquet,
                              columns=["TransactionID", "card1", "addr1", "TransactionAmt"])
        df = feats.merge(raw, on="TransactionID", how="left")
        df["uid"] = build_uid(df["card1"], df["addr1"], df["D1_normalized"])
        return cls._build(df)

    @classmethod
    def _build(cls, df: pd.DataFrame) -> "FeatureStore":
        """State for a transaction after the end of ``df``: every row is strictly
        earlier, so priors = the key's full history. (Copying the last row's
        stored features would be off by one — those exclude that row itself — so
        a client with one past transaction would look like it had none.)"""
        amt = df["TransactionAmt"].astype("float64")
        g = amt.groupby(df["uid"], sort=False)
        stats = pd.DataFrame({"n": g.size(), "avg": g.mean(), "sd": g.std(ddof=1)})
        uid_state = {
            r.Index: {
                "uid_prior_count": int(r.n),
                "uid_amt_expanding_mean": float(r.avg),
                # sample std is undefined below two observations -> Phase 2.5 sentinel
                "uid_amt_expanding_std": float(r.sd) if r.n >= 2 else float(UID_SENTINEL),
            }
            for r in stats.itertuples()
        }
        card1_state = {str(int(k)): int(v)
                       for k, v in df.groupby("card1", sort=False).size().items()}
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
