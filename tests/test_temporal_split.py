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
