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
