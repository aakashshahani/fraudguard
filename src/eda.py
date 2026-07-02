"""
FraudGuard — Phase 1 EDA.

Generates and saves the three exploratory views the project charter calls for:

1. Class imbalance         — the ``isFraud`` distribution.
2. Missingness patterns    — overall, and specifically the transaction/identity
                             join gap (identity columns are null wherever a
                             transaction had no matching identity record).
3. TransactionDT range     — the temporal span the split is built on, with the
                             train / val / test cut points overlaid.

Figures are written to ``reports/figures/``. Run AFTER ``src/data_prep.py`` so
that the merged parquet and split boundaries exist.

    python -m src.eda
"""

from __future__ import annotations

import json
import logging

import matplotlib

matplotlib.use("Agg")  # headless: save files, never open a window
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data_prep import (
    MERGED_PARQUET,
    PROJECT_ROOT,
    SPLIT_BOUNDARIES_JSON,
    _dt_to_assumed_date,
    _dt_to_relative_day,
)

FIG_DIR = PROJECT_ROOT / "reports" / "figures"

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
log = logging.getLogger("fraudguard.eda")

sns.set_theme(style="whitegrid")


def _load() -> pd.DataFrame:
    if not MERGED_PARQUET.exists():
        raise FileNotFoundError(
            f"{MERGED_PARQUET} not found. Run `python -m src.data_prep` first."
        )
    return pd.read_parquet(MERGED_PARQUET)


def plot_class_imbalance(df: pd.DataFrame) -> None:
    counts = df["isFraud"].value_counts().sort_index()
    fraud_rate = df["isFraud"].mean()

    labels_x = counts.index.astype(str)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=labels_x, y=counts.values, hue=labels_x, palette="deep", legend=False, ax=ax)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(["Legit (0)", "Fraud (1)"])
    ax.set_ylabel("Transactions")
    ax.set_xlabel("")
    ax.set_title(f"Class imbalance — fraud rate = {fraud_rate:.3%}")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    fig.tight_layout()
    out = FIG_DIR / "01_class_imbalance.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    log.info("fraud rate = %.3f%% | saved %s", fraud_rate * 100, out.name)


def plot_missingness(df: pd.DataFrame) -> None:
    # Top-30 most-missing columns.
    frac_missing = df.isna().mean().sort_values(ascending=False)
    top = frac_missing.head(30)

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.barplot(x=top.values, y=top.index, hue=top.index, palette="rocket", legend=False, ax=ax)
    ax.set_xlabel("Fraction missing")
    ax.set_ylabel("")
    ax.set_title("Top-30 columns by missingness")
    fig.tight_layout()
    out = FIG_DIR / "02_missingness_top30.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    log.info("saved %s", out.name)

    # Identity join gap: identity columns (id_* and DeviceType/DeviceInfo)
    # are null exactly where a transaction had no identity record.
    id_cols = [c for c in df.columns if c.startswith("id_")] + [
        c for c in ("DeviceType", "DeviceInfo") if c in df.columns
    ]
    if id_cols:
        has_identity = df[id_cols].notna().any(axis=1)
        gap = pd.Series(
            {
                "Has identity record": int(has_identity.sum()),
                "No identity record": int((~has_identity).sum()),
            }
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=gap.index, y=gap.values, hue=gap.index, palette="mako", legend=False, ax=ax)
        ax.set_ylabel("Transactions")
        ax.set_xlabel("")
        ax.set_title(
            f"Transaction/identity join gap "
            f"({has_identity.mean():.1%} have identity data)"
        )
        for i, v in enumerate(gap.values):
            ax.text(i, v, f"{v:,}", ha="center", va="bottom")
        fig.tight_layout()
        out = FIG_DIR / "03_identity_join_gap.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        log.info("identity coverage = %.1f%% | saved %s", has_identity.mean() * 100, out.name)


def plot_transaction_dt(df: pd.DataFrame) -> None:
    dt = df["TransactionDT"]

    boundaries = None
    if SPLIT_BOUNDARIES_JSON.exists():
        boundaries = json.loads(SPLIT_BOUNDARIES_JSON.read_text())

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(dt, bins=100, ax=ax, color="steelblue")
    ax.set_xlabel("TransactionDT (seconds from an undisclosed reference)")
    ax.set_ylabel("Transactions")
    ax.set_title(
        f"TransactionDT range: day {_dt_to_relative_day(dt.min()):.0f} → "
        f"day {_dt_to_relative_day(dt.max()):.0f} "
        f"({int(dt.min()):,} .. {int(dt.max()):,} sec)"
    )
    # Illustrative only: the true reference datetime is not disclosed by IEEE-CIS.
    ax.text(
        0.5,
        -0.28,
        f"Calendar dates are an assumed convention, not official: "
        f"~{_dt_to_assumed_date(dt.min())[:10]} → {_dt_to_assumed_date(dt.max())[:10]}",
        transform=ax.transAxes,
        ha="center",
        fontsize=8,
        color="#888888",
    )

    if boundaries:
        for label, key, color in [
            ("train|val", "cut_train_dt", "orange"),
            ("val|test", "cut_val_dt", "red"),
        ]:
            ax.axvline(boundaries[key], color=color, linestyle="--", label=label)
        ax.legend()

    fig.tight_layout()
    out = FIG_DIR / "04_transaction_dt_range.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    log.info("DT range = [%d .. %d] | saved %s", int(dt.min()), int(dt.max()), out.name)


def run() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = _load()
    plot_class_imbalance(df)
    plot_missingness(df)
    plot_transaction_dt(df)
    log.info("EDA complete — figures in %s", FIG_DIR.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    run()
