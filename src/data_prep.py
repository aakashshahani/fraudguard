"""
FraudGuard — Phase 1: Data acquisition, merge, and temporal split.

Pipeline
--------
1. Load ``train_transaction.csv`` and ``train_identity.csv`` (IEEE-CIS).
2. Left-join identity onto transactions on ``TransactionID``.
   Identity does NOT cover every transaction; the resulting nulls are
   expected and are intentionally preserved (not dropped).
3. Downcast numeric dtypes (float64 -> float32, int64 -> int32 where
   the value range fits) to shrink the in-memory footprint on a laptop.
4. Persist the merged frame as Parquet to ``data/processed``.
5. Build a strictly chronological train / validation / test split on
   ``TransactionDT`` (70 / 15 / 15), sorted ascending, never shuffled.
   Split indices and the audit boundaries are saved to ``data/processed``.

Scope note: Phase 1 is data prep ONLY. No encoding, no feature
engineering, no modeling happens here.

Run
---
    python -m src.data_prep            # from the project root
    python src/data_prep.py
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRANSACTION_CSV = RAW_DIR / "train_transaction.csv"
IDENTITY_CSV = RAW_DIR / "train_identity.csv"

MERGED_PARQUET = PROCESSED_DIR / "train_merged.parquet"
SPLIT_INDICES_PARQUET = PROCESSED_DIR / "split_indices.parquet"
SPLIT_BOUNDARIES_JSON = PROCESSED_DIR / "split_boundaries.json"

# Chronological split fractions (must sum to 1.0).
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# TransactionDT is a time delta in SECONDS from a reference datetime that the
# IEEE-CIS competition deliberately does NOT disclose (so competitors can't join
# external calendar data — holidays, weekends — against it). We therefore report
# the split in RELATIVE DAYS (day 0 = the reference), which is exact and needs no
# assumption. The calendar dates below are a purely illustrative, ASSUMED
# convention (a common community guess of ~Dec 2017 for day 0) — they are NOT
# official and nothing downstream depends on them. Always labelled "assumed".
ASSUMED_REFERENCE_DATE = datetime(2017, 12, 1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fraudguard.data_prep")


# --------------------------------------------------------------------------- #
# Load + merge
# --------------------------------------------------------------------------- #
def load_raw(
    transaction_csv: Path = TRANSACTION_CSV,
    identity_csv: Path = IDENTITY_CSV,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the raw transaction and identity CSVs."""
    for path in (transaction_csv, identity_csv):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing raw file: {path}\n"
                "Download the dataset first — see the README 'Data acquisition' "
                "section (Kaggle account + accepted competition rules + "
                "kaggle.json are required)."
            )

    log.info("Loading transactions: %s", transaction_csv.name)
    transactions = pd.read_csv(transaction_csv)
    log.info("  -> %s rows x %s cols", f"{len(transactions):,}", transactions.shape[1])

    log.info("Loading identity: %s", identity_csv.name)
    identity = pd.read_csv(identity_csv)
    log.info("  -> %s rows x %s cols", f"{len(identity):,}", identity.shape[1])

    return transactions, identity


def merge_transaction_identity(
    transactions: pd.DataFrame, identity: pd.DataFrame
) -> pd.DataFrame:
    """
    Left-join identity onto transactions on ``TransactionID``.

    A left join is deliberate: identity records exist for only a subset of
    transactions, and we must keep every transaction. The rows without a
    matching identity record become nulls in the identity columns — that
    gap is a real, informative signal, not a data-quality bug, so we do
    NOT drop those rows.
    """
    merged = transactions.merge(identity, on="TransactionID", how="left")

    matched = identity["TransactionID"].isin(transactions["TransactionID"]).sum()
    coverage = matched / len(transactions)
    log.info(
        "Merged: %s rows x %s cols | identity coverage = %.1f%% "
        "(%s of %s transactions have identity data)",
        f"{len(merged):,}",
        merged.shape[1],
        coverage * 100,
        f"{matched:,}",
        f"{len(transactions):,}",
    )
    return merged


# --------------------------------------------------------------------------- #
# Memory reduction
# --------------------------------------------------------------------------- #
def downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast numeric columns to the smallest dtype that safely holds their
    range: float64 -> float32, int64 -> int32 (or smaller) where it fits.

    Uses ``pd.to_numeric(downcast=...)`` which only narrows when the values
    round-trip exactly, so this is lossless for integers and is a controlled,
    negligible precision trade for floats (float32 keeps ~7 significant
    digits — ample for these features).
    """
    before_mb = df.memory_usage(deep=True).sum() / 1024**2

    float_cols = df.select_dtypes(include=["float64"]).columns
    for col in float_cols:
        df[col] = df[col].astype("float32")

    int_cols = df.select_dtypes(include=["int64"]).columns
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    after_mb = df.memory_usage(deep=True).sum() / 1024**2
    log.info(
        "Downcast numerics: %.1f MB -> %.1f MB (%.0f%% reduction)",
        before_mb,
        after_mb,
        (1 - after_mb / before_mb) * 100 if before_mb else 0.0,
    )
    return df


# --------------------------------------------------------------------------- #
# Temporal split
# --------------------------------------------------------------------------- #
def _dt_to_relative_day(dt_seconds: float) -> float:
    """TransactionDT (seconds) as a relative day index. Exact, no assumption."""
    return round(float(dt_seconds) / 86_400, 2)


def _dt_to_assumed_date(dt_seconds: float) -> str:
    """
    Illustrative calendar date under an ASSUMED reference (see
    ``ASSUMED_REFERENCE_DATE``). NOT an official IEEE-CIS date — for readability
    only, and always presented with an 'assumed' label.
    """
    return (ASSUMED_REFERENCE_DATE + timedelta(seconds=float(dt_seconds))).isoformat()


def temporal_split(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
) -> tuple[pd.DataFrame, dict]:
    """
    Assign each row to ``train`` / ``val`` / ``test`` strictly by time.

    The frame is ordered by ``TransactionDT`` ascending (never shuffled,
    never random). Rows are cut at the ``TransactionDT`` quantiles rather
    than by raw row position so that a single timestamp can never land in
    two splits. That guarantees a hard, leak-free boundary:

        max(TransactionDT | train) < min(TransactionDT | val)
        max(TransactionDT | val)   < min(TransactionDT | test)

    Returns a ``(labels_df, boundaries)`` pair where ``labels_df`` carries
    ``TransactionID``, ``TransactionDT`` and a ``split`` column, and
    ``boundaries`` is an audit dict of the raw seconds + relative-day cut points
    (plus an illustrative, clearly-labelled assumed calendar date).
    """
    if "TransactionDT" not in df.columns:
        raise KeyError("TransactionDT column is required for the temporal split.")

    dt = df["TransactionDT"]
    cut_train = dt.quantile(train_frac)
    cut_val = dt.quantile(train_frac + val_frac)

    # Strict inequalities on the value (not the row index) prevent tie leakage.
    split = pd.Series("test", index=df.index, dtype="object")
    split[dt < cut_train] = "train"
    split[(dt >= cut_train) & (dt < cut_val)] = "val"

    labels = pd.DataFrame(
        {
            "TransactionID": df["TransactionID"].to_numpy(),
            "TransactionDT": dt.to_numpy(),
            "split": split.to_numpy(),
        }
    )

    def _bounds(name: str) -> dict:
        s = labels.loc[labels["split"] == name, "TransactionDT"]
        return {
            "n": int(s.size),
            "frac": float(s.size / len(labels)),
            "dt_min": int(s.min()),
            "dt_max": int(s.max()),
            "day_min": _dt_to_relative_day(s.min()),
            "day_max": _dt_to_relative_day(s.max()),
            # Illustrative only — NOT an official date. See ASSUMED_REFERENCE_DATE.
            "assumed_date_min": _dt_to_assumed_date(s.min()),
            "assumed_date_max": _dt_to_assumed_date(s.max()),
        }

    boundaries = {
        "unit": "TransactionDT is seconds from an undisclosed reference; "
        "relative days are exact, assumed_* dates are illustrative only.",
        "assumed_reference_date": ASSUMED_REFERENCE_DATE.isoformat(),
        "cut_train_dt": int(cut_train),
        "cut_val_dt": int(cut_val),
        "train": _bounds("train"),
        "val": _bounds("val"),
        "test": _bounds("test"),
    }
    return labels, boundaries


def log_boundaries(boundaries: dict) -> None:
    """Print the split boundaries so the split is auditable at a glance."""
    log.info(
        "Temporal split boundaries — primary unit is relative days "
        "(day 0 = undisclosed reference); calendar dates are ASSUMED, not official:"
    )
    for name in ("train", "val", "test"):
        b = boundaries[name]
        log.info(
            "  %-5s | n=%9s (%.1f%%) | DT [%d .. %d] | day [%.1f .. %.1f] "
            "| assumed date %s .. %s",
            name,
            f"{b['n']:,}",
            b["frac"] * 100,
            b["dt_min"],
            b["dt_max"],
            b["day_min"],
            b["day_max"],
            b["assumed_date_min"][:10],
            b["assumed_date_max"][:10],
        )


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_outputs(
    merged: pd.DataFrame, labels: pd.DataFrame, boundaries: dict
) -> None:
    """Write the merged frame, split labels, and audit boundaries to disk."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    merged.to_parquet(MERGED_PARQUET, index=False)
    log.info("Wrote merged frame -> %s", MERGED_PARQUET.relative_to(PROJECT_ROOT))

    labels.to_parquet(SPLIT_INDICES_PARQUET, index=False)
    log.info("Wrote split labels -> %s", SPLIT_INDICES_PARQUET.relative_to(PROJECT_ROOT))

    SPLIT_BOUNDARIES_JSON.write_text(json.dumps(boundaries, indent=2))
    log.info("Wrote split boundaries -> %s", SPLIT_BOUNDARIES_JSON.relative_to(PROJECT_ROOT))


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run() -> dict:
    """Execute the full Phase 1 pipeline and return the audit boundaries."""
    transactions, identity = load_raw()
    merged = merge_transaction_identity(transactions, identity)
    merged = downcast_numeric(merged)

    labels, boundaries = temporal_split(merged)
    log_boundaries(boundaries)

    save_outputs(merged, labels, boundaries)
    log.info("Phase 1 complete.")
    return boundaries


if __name__ == "__main__":
    run()
