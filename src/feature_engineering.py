"""
FraudGuard — Phase 2: Feature engineering.

Builds the model-ready feature set on top of Phase 1's artifacts. It does NOT
re-derive the temporal split — it loads ``split_indices.parquet`` and carries
the existing ``train`` / ``val`` / ``test`` assignment straight through.

Scope: feature engineering ONLY. No model training, no evaluation.

What it produces
----------------
1. Missingness-as-signal:
   - ``has_identity``   : did this transaction have a matching identity record?
   - ``v_missing_count``: number of null V-columns for the row.
2. Categorical encoding **fit strictly on the train split**, applied to
   val/test (categories unseen in train map to a designated "unknown" value):
   - frequency encoding for high-cardinality cols (card1-6, addr1/2, email
     domains, DeviceInfo);
   - label encoding for lower-cardinality cols (ProductCD, M1-M9, DeviceType,
     the string-typed id_12..id_38);
   - any remaining non-numeric column is frequency-encoded as a safety net.
3. Temporal-safe aggregates (causal — a row only ever "sees" the past):
   - ``card1_prior_count`` and ``P_emaildomain_prior_count``: number of prior
     occurrences of that key with a **strictly earlier** TransactionDT.
4. Other transforms:
   - ``TransactionAmt_log1p`` (the amount is heavily right-skewed);
   - ``D1_normalized`` = D1 - day-of-transaction (a more stable tenure signal);
     the raw ``D1`` is kept alongside for comparison;
   - V-column nulls filled with the ``-999`` sentinel so tree models can split
     on "was missing" as its own value.

Run
---
    python -m src.feature_engineering
"""

from __future__ import annotations

import logging
import re

import pandas as pd

from src.data_prep import (
    MERGED_PARQUET,
    PROCESSED_DIR,
    PROJECT_ROOT,
    SPLIT_INDICES_PARQUET,
)

FEATURES_PARQUET = PROCESSED_DIR / "features.parquet"

# Encoding sentinels.
MISSING_TOKEN = "__missing__"  # stands in for NaN so missingness is learnable
UNKNOWN_LABEL = -1             # label code for categories unseen in train
V_FILL_SENTINEL = -999         # tree-friendly "was missing" marker for V-cols

# Column groups (per the Phase 2 spec).
FREQ_COLS = [
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "P_emaildomain", "R_emaildomain", "DeviceInfo",
]
LABEL_COLS_BASE = ["ProductCD", "DeviceType"] + [f"M{i}" for i in range(1, 10)]
ID_LABEL_RANGE = [f"id_{i:02d}" for i in range(12, 39)]  # id_12 .. id_38

PRIOR_COUNT_KEYS = ["card1", "P_emaildomain"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fraudguard.feature_engineering")


# --------------------------------------------------------------------------- #
# Small, testable encoding primitives (fit on train, apply anywhere)
# --------------------------------------------------------------------------- #
def _as_cat(series: pd.Series) -> pd.Series:
    """Normalise any column to a string category, NaN -> MISSING_TOKEN."""
    return series.astype("string").fillna(MISSING_TOKEN)


def fit_frequency_map(train_series: pd.Series) -> dict:
    """Category -> occurrence count, built from TRAIN rows only."""
    return _as_cat(train_series).value_counts().to_dict()


def apply_frequency(series: pd.Series, freq_map: dict) -> pd.Series:
    """Map categories to their train frequency; unseen -> 0."""
    return _as_cat(series).map(freq_map).fillna(0).astype("int64")


def fit_label_map(train_series: pd.Series) -> dict:
    """Category -> integer code (0..k-1), built from TRAIN rows only."""
    cats = sorted(_as_cat(train_series).unique())
    return {cat: i for i, cat in enumerate(cats)}


def apply_label(series: pd.Series, label_map: dict) -> pd.Series:
    """Map categories to their train code; unseen -> UNKNOWN_LABEL (-1)."""
    return _as_cat(series).map(label_map).fillna(UNKNOWN_LABEL).astype("int64")


# --------------------------------------------------------------------------- #
# Temporal-safe aggregate (the critical, leakage-sensitive part)
# --------------------------------------------------------------------------- #
def expanding_prior_count(
    df: pd.DataFrame, key_col: str, time_col: str = "TransactionDT"
) -> pd.Series:
    """
    For each row, the number of earlier rows sharing ``key_col`` — counting
    ONLY rows whose ``time_col`` is strictly smaller (never equal, never later).

    Implemented with ``rank(method="min")`` on the time column within each key
    group: the min-rank of a value equals ``1 + (# strictly-smaller)``, so
    ``rank - 1`` is exactly the strictly-earlier count. This is order-
    independent (it keys off the timestamp, not the row position), so it is
    correct regardless of how the frame is sorted — future rows can never
    contribute. Ties on ``time_col`` do not count one another.
    """
    key = df[key_col].astype("string").fillna(MISSING_TOKEN)
    rank_min = df[time_col].groupby(key).rank(method="min")
    return (rank_min - 1).astype("int64")


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def load_phase1() -> pd.DataFrame:
    """Load the merged frame and attach the existing Phase 1 split labels."""
    if not MERGED_PARQUET.exists() or not SPLIT_INDICES_PARQUET.exists():
        raise FileNotFoundError(
            "Phase 1 artifacts missing. Run `python -m src.data_prep` first."
        )
    df = pd.read_parquet(MERGED_PARQUET)
    splits = pd.read_parquet(SPLIT_INDICES_PARQUET)[["TransactionID", "split"]]
    df = df.merge(splits, on="TransactionID", how="left")
    if df["split"].isna().any():
        raise ValueError("Some transactions have no split label — Phase 1 mismatch.")
    log.info("Loaded %s rows x %s cols with split attached", f"{len(df):,}", df.shape[1])
    return df


# --------------------------------------------------------------------------- #
# Feature builders
# --------------------------------------------------------------------------- #
def add_missingness_features(df: pd.DataFrame) -> pd.DataFrame:
    """has_identity + v_missing_count. Must run BEFORE any null-filling."""
    id_source_cols = [c for c in df.columns if c.startswith("id_")]
    id_source_cols += [c for c in ("DeviceType", "DeviceInfo") if c in df.columns]
    df["has_identity"] = df[id_source_cols].notna().any(axis=1)

    v_cols = [c for c in df.columns if re.fullmatch(r"V\d+", c)]
    df["v_missing_count"] = df[v_cols].isna().sum(axis=1).astype("int16")

    log.info(
        "Missingness: has_identity coverage = %.1f%% | v_missing_count over %d V-cols",
        df["has_identity"].mean() * 100,
        len(v_cols),
    )
    return df


def add_temporal_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Causal prior-occurrence counts. Uses RAW keys, so run before encoding."""
    for key in PRIOR_COUNT_KEYS:
        df[f"{key}_prior_count"] = expanding_prior_count(df, key)
        log.info(
            "Temporal aggregate: %s_prior_count (max prior seen = %d)",
            key,
            int(df[f"{key}_prior_count"].max()),
        )
    return df


def add_other_features(df: pd.DataFrame) -> pd.DataFrame:
    """log1p amount, day-of-transaction, and normalized D1 (raw D1 kept)."""
    import numpy as np

    df["TransactionAmt_log1p"] = np.log1p(df["TransactionAmt"]).astype("float32")

    # Day index from the seconds offset (see Phase 1: reference is undisclosed,
    # so this is an exact relative-day count, not a calendar day).
    df["TransactionDay"] = (df["TransactionDT"] // 86_400).astype("int32")

    # D1 drifts with calendar time; subtracting the day yields a stable tenure.
    df["D1_normalized"] = (df["D1"] - df["TransactionDay"]).astype("float32")

    log.info("Other features: TransactionAmt_log1p, TransactionDay, D1_normalized")
    return df


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Fit every encoder on the TRAIN split only, then apply to all rows.

    Returns the frame plus the fitted maps (so they are inspectable / reusable
    and the leakage test can assert they contain no val/test-only categories).
    """
    freq_cols, label_cols = select_encoder_columns(df)
    train_mask = df["split"] == "train"
    fitted_maps: dict = {"frequency": {}, "label": {}}

    for col in freq_cols:
        fmap = fit_frequency_map(df.loc[train_mask, col])
        df[col] = apply_frequency(df[col], fmap)
        fitted_maps["frequency"][col] = fmap

    for col in label_cols:
        lmap = fit_label_map(df.loc[train_mask, col])
        df[col] = apply_label(df[col], lmap)
        fitted_maps["label"][col] = lmap

    log.info(
        "Encoded %d frequency + %d label columns (fit on %s train rows)",
        len(freq_cols), len(label_cols), f"{int(train_mask.sum()):,}",
    )
    return df, fitted_maps


def select_encoder_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Decide which columns get frequency vs label encoding (shared by the Phase 2
    transform and by serving-time encoder persistence, so both agree exactly).
    Label-encode only the *string* id columns in id_12..id_38 — numeric id_* are
    already model-ready and must not be turned into arbitrary codes.
    """
    non_numeric = [
        c for c in df.columns
        if c not in ("split",)
        and not pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
    ]
    id_label_cols = [c for c in ID_LABEL_RANGE if c in non_numeric]
    label_cols = [c for c in LABEL_COLS_BASE + id_label_cols if c in df.columns]
    freq_cols = [c for c in FREQ_COLS if c in df.columns]
    handled = set(freq_cols) | set(label_cols)
    auto_freq = [c for c in non_numeric if c not in handled]  # safety net
    return freq_cols + auto_freq, label_cols


def fit_encoders(df: pd.DataFrame) -> dict:
    """Fit frequency/label maps on the TRAIN split only (no transform). Serving reuses this."""
    freq_cols, label_cols = select_encoder_columns(df)
    train = df[df["split"] == "train"]
    maps = {"frequency": {}, "label": {}}
    for col in freq_cols:
        maps["frequency"][col] = {str(k): int(v) for k, v in fit_frequency_map(train[col]).items()}
    for col in label_cols:
        maps["label"][col] = {str(k): int(v) for k, v in fit_label_map(train[col]).items()}
    return maps


def fill_v_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """Fill V-column nulls with -999 (done AFTER v_missing_count is computed)."""
    v_cols = [c for c in df.columns if re.fullmatch(r"V\d+", c)]
    df[v_cols] = df[v_cols].fillna(V_FILL_SENTINEL)
    log.info("Filled nulls in %d V-columns with sentinel %d", len(v_cols), V_FILL_SENTINEL)
    return df


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Run every step in the order the leakage/missingness constraints require."""
    df = add_missingness_features(df)   # before any fill
    df = add_temporal_aggregates(df)    # raw keys, before encoding
    df = add_other_features(df)
    df, maps = encode_categoricals(df)  # fit on train only
    df = fill_v_sentinels(df)           # after v_missing_count
    return df, maps


def _assert_model_ready(df: pd.DataFrame) -> None:
    """Every feature column (all but the string ``split``) must be numeric/bool."""
    bad = [
        c for c in df.columns
        if c != "split"
        and not pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
    ]
    if bad:
        raise TypeError(f"Non-numeric feature columns survived encoding: {bad}")


def run() -> pd.DataFrame:
    df = load_phase1()
    df, _maps = build_features(df)
    _assert_model_ready(df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATURES_PARQUET, index=False)
    log.info(
        "Wrote %s rows x %s cols -> %s",
        f"{len(df):,}", df.shape[1], FEATURES_PARQUET.relative_to(PROJECT_ROOT),
    )
    log.info(
        "Split preserved: %s",
        df["split"].value_counts().reindex(["train", "val", "test"]).to_dict(),
    )
    log.info("Phase 2 complete.")
    return df


if __name__ == "__main__":
    run()
