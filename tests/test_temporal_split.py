"""
Guardrail tests for the temporal split.

The single most important invariant of this project is that the split leaks
no future information:

    max(TransactionDT | train) < min(TransactionDT | val)
    max(TransactionDT | val)   < min(TransactionDT | test)

If anyone ever swaps the chronological cut for a shuffled / random split
(e.g. sklearn ``train_test_split`` with its default ``shuffle=True``), these
tests must fail loudly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_prep import (
    SPLIT_INDICES_PARQUET,
    TRAIN_FRAC,
    VAL_FRAC,
    temporal_split,
)


def _synthetic(n: int = 10_000, ties: bool = False) -> pd.DataFrame:
    """A frame that mimics the columns temporal_split depends on."""
    rng = np.random.default_rng(42)
    if ties:
        # Many repeated timestamps to prove ties never straddle a boundary.
        dt = np.repeat(np.arange(n // 10), 10)
    else:
        dt = np.arange(n)
    # Shuffle the row order on the way in — the split must not rely on it.
    order = rng.permutation(len(dt))
    return pd.DataFrame(
        {
            "TransactionID": np.arange(len(dt))[order],
            "TransactionDT": dt[order],
            "isFraud": rng.integers(0, 2, len(dt)),
        }
    )


def _assert_strictly_chronological(labels: pd.DataFrame) -> None:
    train = labels.loc[labels["split"] == "train", "TransactionDT"]
    val = labels.loc[labels["split"] == "val", "TransactionDT"]
    test = labels.loc[labels["split"] == "test", "TransactionDT"]

    assert len(train) and len(val) and len(test), "every split must be non-empty"
    assert train.max() < val.min(), "train leaks into validation"
    assert val.max() < test.min(), "validation leaks into test"


def test_split_is_strictly_chronological():
    labels, _ = temporal_split(_synthetic())
    _assert_strictly_chronological(labels)


def test_split_strict_even_with_tied_timestamps():
    # Repeated timestamps are the classic way a naive split leaks. The cut is
    # on the DT *value*, so a tie group can never land on both sides.
    labels, _ = temporal_split(_synthetic(ties=True))
    _assert_strictly_chronological(labels)


def test_split_proportions_are_approximately_correct():
    labels, boundaries = temporal_split(_synthetic())
    assert boundaries["train"]["frac"] == pytest.approx(TRAIN_FRAC, abs=0.02)
    assert boundaries["val"]["frac"] == pytest.approx(VAL_FRAC, abs=0.02)


def test_split_covers_every_row_exactly_once():
    df = _synthetic()
    labels, _ = temporal_split(df)
    assert len(labels) == len(df)
    assert set(labels["split"]) == {"train", "val", "test"}


def test_a_shuffled_split_would_fail_this_guardrail():
    """
    Sanity check on the guardrail itself: a random split MUST trip the
    chronological assertion, proving the test can actually catch a leak.
    """
    df = _synthetic()
    rng = np.random.default_rng(0)
    labels = df[["TransactionID", "TransactionDT"]].copy()
    labels["split"] = rng.choice(["train", "val", "test"], size=len(df))
    with pytest.raises(AssertionError):
        _assert_strictly_chronological(labels)


@pytest.mark.skipif(
    not SPLIT_INDICES_PARQUET.exists(),
    reason="split_indices.parquet not built yet — run `python -m src.data_prep`",
)
def test_persisted_split_is_strictly_chronological():
    """The real, on-disk split (once generated) must hold the same invariant."""
    labels = pd.read_parquet(SPLIT_INDICES_PARQUET)
    _assert_strictly_chronological(labels)


# =========================================================================== #
# Phase 2 — feature-engineering leakage guardrails
#
# Two ways feature engineering can leak the future into the past:
#   1. A temporal aggregate that counts rows it shouldn't be able to "see" yet.
#   2. An encoder fit on val/test data instead of train only.
# Both must fail loudly here.
# =========================================================================== #

from src.feature_engineering import (  # noqa: E402
    UNKNOWN_LABEL,
    apply_frequency,
    apply_label,
    expanding_prior_count,
    fit_frequency_map,
    fit_label_map,
)


def test_expanding_prior_count_matches_hand_calc_and_ignores_future():
    """
    Distinct timestamps, one repeated card1. The prior count at each row must
    equal the hand-calculated number of *earlier* same-card1 rows — and must
    not change when the input rows are shuffled (proving it keys off the
    timestamp, not row position, so a future row can never contribute).
    """
    df = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5],
            "TransactionDT": [10, 20, 30, 40, 50],
            "card1": [100, 100, 200, 100, 200],
        }
    )
    # By hand, per row in time order:
    #   id1 card100 @10 -> 0 priors
    #   id2 card100 @20 -> 1 prior  (id1)
    #   id3 card200 @30 -> 0 priors
    #   id4 card100 @40 -> 2 priors (id1, id2)
    #   id5 card200 @50 -> 1 prior  (id3)
    expected = {1: 0, 2: 1, 3: 0, 4: 2, 5: 1}

    got = df.assign(pc=expanding_prior_count(df, "card1"))
    assert dict(zip(got["TransactionID"], got["pc"])) == expected

    # Shuffle the rows; per-TransactionID answers must be identical.
    shuffled = df.sample(frac=1.0, random_state=7).reset_index(drop=True)
    got_s = shuffled.assign(pc=expanding_prior_count(shuffled, "card1"))
    assert dict(zip(got_s["TransactionID"], got_s["pc"])) == expected


def test_expanding_prior_count_excludes_tied_timestamps():
    """
    Rows sharing the exact same TransactionDT must NOT count one another —
    only strictly-earlier rows count.
    """
    df = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "TransactionDT": [10, 10, 20],  # first two are tied
            "card1": [100, 100, 100],
        }
    )
    # id1 @10 -> 0 ; id2 @10 -> 0 (tie, does not see id1) ; id3 @20 -> 2
    expected = {1: 0, 2: 0, 3: 2}
    got = df.assign(pc=expanding_prior_count(df, "card1"))
    assert dict(zip(got["TransactionID"], got["pc"])) == expected


def _split_frame():
    """Tiny frame with a category that appears ONLY in the validation split."""
    return pd.DataFrame(
        {
            "split": ["train", "train", "train", "val", "test"],
            "cat": ["a", "a", "b", "val_only", "b"],
        }
    )


def test_frequency_encoding_is_fit_on_train_only():
    df = _split_frame()
    fmap = fit_frequency_map(df.loc[df["split"] == "train", "cat"])

    # The val-only category must NOT be in the fitted train map.
    assert "val_only" not in fmap
    assert fmap == {"a": 2, "b": 1}

    encoded = apply_frequency(df["cat"], fmap)
    # Unseen category encodes to 0; train categories to their train counts.
    assert encoded.tolist() == [2, 2, 1, 0, 1]


def test_label_encoding_is_fit_on_train_only():
    df = _split_frame()
    lmap = fit_label_map(df.loc[df["split"] == "train", "cat"])

    # The val-only category must NOT be in the fitted train map.
    assert "val_only" not in lmap
    assert set(lmap) == {"a", "b"}

    encoded = apply_label(df["cat"], lmap)
    # Unseen category encodes to the designated unknown value (-1).
    assert encoded.iloc[3] == UNKNOWN_LABEL
    assert (encoded.iloc[[0, 1, 2, 4]] >= 0).all()


# =========================================================================== #
# Phase 2.5 — UID aggregation leakage guardrails
#
# The uid_* expanding stats must be causal: each row summarises only STRICTLY
# earlier same-uid rows, never its own amount, never a future row, and never
# reset at a split boundary. Insufficient history must be flagged, not faked.
# =========================================================================== #

from src.uid_features import (  # noqa: E402
    UID_SENTINEL,
    compute_uid_features,
)


def test_uid_expanding_stats_exclude_the_current_rows_own_amount():
    """
    Three transactions for one uid. At the last row, including its own amount
    (30) would change the mean from 15 -> 20; assert it stays 15.
    """
    df = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "TransactionDT": [1, 2, 3],
            "TransactionAmt": [10.0, 20.0, 30.0],
            "uid": ["u", "u", "u"],
        }
    )
    out = compute_uid_features(df)

    assert out["uid_prior_count"].tolist() == [0, 1, 2]
    # priors of row 3 are [10, 20] -> mean 15, NOT 20 (which includes current).
    assert out["uid_amt_expanding_mean"].iloc[2] == pytest.approx(15.0)
    assert out["uid_amt_expanding_mean"].iloc[2] != pytest.approx(20.0)
    # sample std of [10, 20] excludes the current 30.
    assert out["uid_amt_expanding_std"].iloc[2] == pytest.approx(
        pd.Series([10.0, 20.0]).std(), rel=1e-4
    )


def test_uid_features_span_split_boundary_without_reset():
    """
    Same uid across train/val/test. The stats must accumulate across the
    boundary — a test-split row must see all earlier train+val rows, proving
    the aggregate is computed over the full timeline, not refit per split.
    """
    df = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "TransactionDT": [10, 20, 30, 40],
            "TransactionAmt": [5.0, 7.0, 9.0, 11.0],
            "uid": ["u", "u", "u", "u"],
            "split": ["train", "train", "val", "test"],
        }
    )
    out = compute_uid_features(df)

    # val row sees 2 train priors; test row sees all 3 (train + val).
    assert out["uid_prior_count"].iloc[2] == 2
    assert out["uid_prior_count"].iloc[3] == 3
    assert out["uid_amt_expanding_mean"].iloc[3] == pytest.approx((5 + 7 + 9) / 3)


def test_uid_insufficient_history_produces_sentinel_not_zero_or_nan():
    """
    0 priors  -> mean & std sentinel.
    1 prior   -> mean is real, std sentinel (std undefined for n < 2).
    2+ priors -> std is a real value, never the sentinel.
    """
    df = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "TransactionDT": [1, 2, 3],
            "TransactionAmt": [10.0, 20.0, 30.0],
            "uid": ["u", "u", "u"],
        }
    )
    out = compute_uid_features(df)

    # 0 priors
    assert out["uid_amt_expanding_mean"].iloc[0] == UID_SENTINEL
    assert out["uid_amt_expanding_std"].iloc[0] == UID_SENTINEL
    # 1 prior -> mean real (10), std still sentinel
    assert out["uid_amt_expanding_mean"].iloc[1] == pytest.approx(10.0)
    assert out["uid_amt_expanding_std"].iloc[1] == UID_SENTINEL
    # 2 priors -> std is real, not the sentinel, and no unflagged NaN anywhere
    assert out["uid_amt_expanding_std"].iloc[2] != UID_SENTINEL
    assert not out["uid_amt_expanding_std"].isna().any()
    assert not out["uid_amt_expanding_mean"].isna().any()


# =========================================================================== #
# Phase 3 — adversarial validation must NEVER touch the test split
#
# The whole phase is train-vs-val; if test rows ever leak into the adversarial
# input, the feature manifest (and every Phase 4 decision built on it) is
# contaminated. This must fail loudly.
# =========================================================================== #

import json  # noqa: E402

from src.adversarial_validation import (  # noqa: E402
    candidate_features,
    load_train_val,
)
from src.data_prep import SPLIT_BOUNDARIES_JSON  # noqa: E402
from src.feature_engineering import FEATURES_PARQUET  # noqa: E402


def test_candidate_features_exclude_temporal_index_and_identifiers():
    """TransactionDT/Day/ID, the split label, and the target are never candidates."""
    df = pd.DataFrame(
        columns=[
            "TransactionID", "TransactionDT", "TransactionDay", "split", "isFraud",
            "card1_prior_count", "uid_prior_count", "V1",
        ]
    )
    cands = candidate_features(df)
    for banned in ["TransactionID", "TransactionDT", "TransactionDay", "split", "isFraud"]:
        assert banned not in cands
    for kept in ["card1_prior_count", "uid_prior_count", "V1"]:
        assert kept in cands


@pytest.mark.skipif(
    not (FEATURES_PARQUET.exists() and SPLIT_BOUNDARIES_JSON.exists()),
    reason="features.parquet / split_boundaries.json not built yet",
)
def test_adversarial_input_is_exactly_train_plus_val_never_test():
    """
    The adversarial input row count must equal train + val, and must NOT equal
    train + val + test. Uses the split *counts* from split_boundaries.json (an
    integer, not the test rows themselves) to prove the exclusion without ever
    loading a test-split row.
    """
    b = json.loads(SPLIT_BOUNDARIES_JSON.read_text())
    train_n, val_n, test_n = b["train"]["n"], b["val"]["n"], b["test"]["n"]

    adv = load_train_val(columns=["split"])

    assert len(adv) == train_n + val_n
    assert len(adv) != train_n + val_n + test_n
    assert set(adv["split"].unique()) == {"train", "val"}  # no 'test' anywhere


# =========================================================================== #
# Phase 4 — modeling must use ONLY manifest columns and NEVER load test rows
# =========================================================================== #

from src.modeling import (  # noqa: E402
    MANIFEST_JSON,
    _read_modeling_data,
    load_Xy,
    manifest_features,
)


def test_modeling_refuses_to_load_test_split():
    """The loader hard-fails on any request that includes the test split."""
    with pytest.raises(ValueError):
        _read_modeling_data(["test"])
    with pytest.raises(ValueError):
        _read_modeling_data(["train", "val", "test"])


@pytest.mark.skipif(
    not (FEATURES_PARQUET.exists() and MANIFEST_JSON.exists()),
    reason="features.parquet / feature_manifest.json not built yet",
)
def test_modeling_uses_only_manifest_columns():
    """
    The feature matrix handed to the models is EXACTLY the manifest's
    allowed_features — no extra columns, and never the target/split/identifiers.
    """
    feats = manifest_features()
    X, _ = load_Xy("val")

    assert list(X.columns) == feats
    for banned in ["isFraud", "split", "TransactionID", "TransactionDT", "TransactionDay"]:
        assert banned not in X.columns


def test_smote_only_ever_sees_training_data_never_validation():
    """
    SMOTE must be fit on train rows only and never touch validation. imblearn
    samplers run only inside .fit() and are bypassed at .predict_proba(); this
    spies on SMOTE to prove it is called exactly once, with the TRAIN row count,
    and that scoring val triggers no resampling and leaves val unchanged.
    """
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    from sklearn.linear_model import LogisticRegression

    seen: list[int] = []

    class SpySMOTE(SMOTE):
        def fit_resample(self, X, y):
            seen.append(len(X))  # record how many rows SMOTE was handed
            return super().fit_resample(X, y)

    rng = np.random.default_rng(0)
    n_tr, n_val = 300, 120
    X_tr = pd.DataFrame(rng.normal(size=(n_tr, 4)), columns=list("abcd"))
    y_tr = np.r_[np.ones(60, int), np.zeros(240, int)]  # 20% minority
    X_val = pd.DataFrame(rng.normal(size=(n_val, 4)), columns=list("abcd"))

    pipe = ImbPipeline([("smote", SpySMOTE(random_state=42)),
                        ("clf", LogisticRegression(max_iter=200))])
    pipe.fit(X_tr, y_tr)
    prob = pipe.predict_proba(X_val)[:, 1]

    assert seen == [n_tr], f"SMOTE saw {seen}, expected exactly one call with {n_tr} train rows"
    assert len(prob) == n_val  # validation was not resampled — one score per val row
