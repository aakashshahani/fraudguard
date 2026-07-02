"""
FraudGuard — Phase 2.5: UID pseudo-identity aggregation.

Extends Phase 2's ``features.parquet`` (does NOT rebuild it) with three
backward-looking aggregate features over a reconstructed client identity.

Why a UID at all
----------------
The competition anonymises the client, but ``card1 + addr1 + D1_normalized``
(card, billing region, and a time-stabilised tenure signal) reconstructs a
*persistent pseudo-identity* — a documented top-solution technique for this
dataset. Aggregating a client's own transaction history is where the strongest
signal in IEEE-CIS lives.

Causal, full-dataset (NOT fit-on-train) — and why that is correct
-----------------------------------------------------------------
Unlike the Phase 2 categorical encoders (which learn a *parameter* — a
category's train-set frequency/code — and therefore must never see val/test),
these aggregates are per-row **lookups into the past**. Each row reads only
rows with a strictly-earlier ``TransactionDT``. A validation row summarising
earlier training rows is past -> future, which is exactly what happens at
inference time; it leaks nothing. So they are computed over the FULL,
time-ordered dataset. The two rules are not in tension: "fit encoders on train
only" is about not learning parameters from held-out data; "aggregate over the
full timeline" is about a causal window that only ever looks backward.

Edge cases (never a silent 0 or unflagged NaN)
----------------------------------------------
``uid_amt_expanding_std`` is undefined with < 2 prior observations; for
``uid_prior_count`` in {0, 1} it is set to the ``-999`` sentinel (the V-column
convention from Phase 2). ``uid_amt_expanding_mean`` is undefined with 0 priors
and is likewise sentinelled. ``uid_prior_count`` itself encodes history depth,
so the model can tell "genuinely low variance" (count >= 2, small std) from
"not enough history to know" (std == -999).

Run
---
    python -m src.uid_features
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data_prep import MERGED_PARQUET, PROJECT_ROOT
from src.feature_engineering import (
    FEATURES_PARQUET,
    MISSING_TOKEN,
    V_FILL_SENTINEL,
    _assert_model_ready,
    expanding_prior_count,
)

UID_SENTINEL = V_FILL_SENTINEL  # -999, reused for insufficient-history cases

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fraudguard.uid_features")


# --------------------------------------------------------------------------- #
# UID construction
# --------------------------------------------------------------------------- #
def _as_int_str(series: pd.Series) -> pd.Series:
    """Integer-valued column -> stable string, NaN -> MISSING_TOKEN."""
    return series.round().astype("Int64").astype("string").fillna(MISSING_TOKEN)


def build_uid(card1: pd.Series, addr1: pd.Series, d1_normalized: pd.Series) -> pd.Series:
    """
    Concatenate card1 + addr1 + D1_normalized into one pseudo-identity key.

    All three components are integer-valued in this dataset; they are rendered
    as stable integer strings (missing -> ``__missing__``) so the key is
    deterministic and NaN never silently collapses distinct clients together.
    """
    return _as_int_str(card1) + "_" + _as_int_str(addr1) + "_" + _as_int_str(d1_normalized)


# --------------------------------------------------------------------------- #
# Causal expanding stats (strictly-earlier window; excludes the current row)
# --------------------------------------------------------------------------- #
def _expanding_prior_stats(
    df: pd.DataFrame,
    uid_col: str = "uid",
    amt_col: str = "TransactionAmt",
    time_col: str = "TransactionDT",
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    For each row return (n_prior, mean_prior, std_prior) over the same-uid rows
    whose ``time_col`` is STRICTLY smaller — the current row and any rows tied
    on ``time_col`` are excluded, matching ``expanding_prior_count`` exactly.

    Method: aggregate per (uid, time), then within each uid take the cumulative
    sum ordered by time and subtract the current time-group's own totals. That
    yields strictly-earlier prefix sums, tie-safe and order-independent. mean
    and std are derived from prefix sum / sum-of-squares; std is the sample
    (ddof=1) std, defined only for n_prior >= 2 (else NaN — sentinelled later).
    """
    work = pd.DataFrame(
        {
            "uid": df[uid_col].astype("string").fillna(MISSING_TOKEN).to_numpy(),
            "dt": df[time_col].to_numpy(),
            "amt": df[amt_col].astype("float64").to_numpy(),
        }
    )
    work["amt2"] = work["amt"] ** 2
    work["row_id"] = np.arange(len(work))

    agg = (
        work.groupby(["uid", "dt"], sort=True)
        .agg(cnt=("amt", "size"), s=("amt", "sum"), s2=("amt2", "sum"))
        .reset_index()
        .sort_values(["uid", "dt"])
    )
    grp = agg.groupby("uid", sort=False)
    # cumulative-through-current minus current  ->  strictly-earlier prefix.
    agg["pc"] = grp["cnt"].cumsum() - agg["cnt"]
    agg["ps"] = grp["s"].cumsum() - agg["s"]
    agg["ps2"] = grp["s2"].cumsum() - agg["s2"]

    merged = work.merge(
        agg[["uid", "dt", "pc", "ps", "ps2"]], on=["uid", "dt"], how="left"
    ).sort_values("row_id")

    n = merged["pc"].to_numpy()
    s = merged["ps"].to_numpy()
    s2 = merged["ps2"].to_numpy()

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n >= 1, s / n, np.nan)
        var = np.where(
            n >= 2,
            (s2 - (s * s) / np.where(n >= 1, n, 1)) / np.where(n >= 2, n - 1, 1),
            np.nan,
        )
    var = np.where(np.isfinite(var) & (var < 0), 0.0, var)  # clamp fp noise
    std = np.sqrt(var)

    idx = df.index
    return (
        pd.Series(n.astype("int64"), index=idx),
        pd.Series(mean, index=idx),
        pd.Series(std, index=idx),
    )


def compute_uid_features(
    df: pd.DataFrame,
    uid_col: str = "uid",
    amt_col: str = "TransactionAmt",
    time_col: str = "TransactionDT",
) -> pd.DataFrame:
    """
    Build the three uid_* feature columns for a frame that already carries a
    ``uid`` column. Testable in isolation on a small handcrafted frame.
    """
    prior_count = expanding_prior_count(df, uid_col, time_col)  # rank-based reuse
    n, mean, std = _expanding_prior_stats(df, uid_col, amt_col, time_col)

    # The rank-based count and the stats' own prior count must agree exactly.
    if not np.array_equal(n.to_numpy(), prior_count.to_numpy()):
        raise AssertionError("uid_prior_count disagrees with expanding-stats count")

    # Sentinel the insufficient-history cases (mean needs >=1 prior, std >=2).
    mean = mean.where(prior_count >= 1, other=UID_SENTINEL)
    std = std.where(prior_count >= 2, other=UID_SENTINEL)

    return pd.DataFrame(
        {
            "uid_prior_count": prior_count.astype("int64"),
            "uid_amt_expanding_mean": mean.astype("float32"),
            "uid_amt_expanding_std": std.astype("float32"),
        },
        index=df.index,
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run() -> pd.DataFrame:
    if not FEATURES_PARQUET.exists():
        raise FileNotFoundError(
            f"{FEATURES_PARQUET} not found. Run `python -m src.feature_engineering` first."
        )
    feats = pd.read_parquet(FEATURES_PARQUET)

    # Reconstruct the UID from RAW card1/addr1 (Phase 2 frequency-encoded these
    # in place, so the encoded counts would collide distinct clients).
    raw = pd.read_parquet(MERGED_PARQUET, columns=["TransactionID", "card1", "addr1"])
    raw = raw.rename(columns={"card1": "card1_raw", "addr1": "addr1_raw"})
    feats = feats.merge(raw, on="TransactionID", how="left")

    feats["uid"] = build_uid(feats["card1_raw"], feats["addr1_raw"], feats["D1_normalized"])
    log.info(
        "Built uid: %s unique pseudo-identities over %s rows",
        f"{feats['uid'].nunique():,}",
        f"{len(feats):,}",
    )

    uid_feats = compute_uid_features(feats)
    feats[uid_feats.columns] = uid_feats

    insufficient = int((feats["uid_prior_count"] < 2).sum())
    log.info(
        "uid features: prior_count max=%d | std sentinelled (<2 priors) for %s rows (%.1f%%)",
        int(feats["uid_prior_count"].max()),
        f"{insufficient:,}",
        insufficient / len(feats) * 100,
    )

    feats = feats.drop(columns=["uid", "card1_raw", "addr1_raw"])
    _assert_model_ready(feats)

    feats.to_parquet(FEATURES_PARQUET, index=False)
    log.info(
        "Overwrote %s -> %s rows x %s cols (split preserved: %s)",
        FEATURES_PARQUET.relative_to(PROJECT_ROOT),
        f"{len(feats):,}",
        feats.shape[1],
        feats["split"].value_counts().reindex(["train", "val", "test"]).to_dict(),
    )
    log.info("Phase 2.5 complete.")
    return feats


if __name__ == "__main__":
    run()
