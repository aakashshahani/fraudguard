"""
FraudGuard — Phase 3: Adversarial validation + locked feature manifest.

Adversarial validation asks a blunt question: *can a classifier tell train rows
apart from validation rows?* On a temporal split some separability is expected
(time moves on), but a feature that is **individually** very separable is drifting
hard enough that a fraud model may latch onto a train-only artifact that no longer
holds by validation time. We flag those and drop them, then freeze the surviving
columns into a manifest that Phase 4 must use verbatim.

HARD REQUIREMENT: test-split rows are never loaded or referenced anywhere in this
phase. Every read goes through ``_read_train_val`` which filters to train+val at
the parquet level, so test rows never enter memory.

Scope: adversarial validation + feature manifest only. No fraud model training.

Run
---
    python -m src.adversarial_validation
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.data_prep import PROCESSED_DIR, PROJECT_ROOT
from src.feature_engineering import FEATURES_PARQUET

ADV_REPORT_JSON = PROCESSED_DIR / "adversarial_validation_report.json"
FEATURE_MANIFEST_JSON = PROCESSED_DIR / "feature_manifest.json"

TARGET = "isFraud"
# Excluded from candidates unconditionally: their train/val separability is
# definitional to the temporal split itself, not real drift signal.
EXCLUDED_UNCONDITIONAL = {
    "TransactionID": "row identifier — not a feature",
    "TransactionDT": "temporal index — separability is definitional to the split",
    # TransactionDay = TransactionDT // 86400 (a Phase 2 feature). It is a coarse
    # copy of the temporal index, so — by the exact same rationale the spec gives
    # for TransactionDT — it is definitionally separable (univariate AUC 1.0) and
    # is excluded as a candidate rather than "discovered" as drift. This also
    # keeps the overall adversarial AUC meaningful instead of a trivial 1.0.
    "TransactionDay": "coarse temporal index (TransactionDT // 86400) — definitional",
}
# Structurally not features either (the label we predict here, and the fraud target).
EXCLUDED_STRUCTURAL = {
    "split": "split label used as the adversarial target — not a feature",
    "isFraud": "fraud target — not an input feature",
}

DRIFT_THRESHOLD = 0.70  # top-solution convention for this competition

# Features kept despite a high-drift flag, each with an explicit reason.
# Populated deliberately after inspecting results (see run() logging / README).
KEEP_DESPITE_FLAG: dict[str, str] = {}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fraudguard.adversarial_validation")


# --------------------------------------------------------------------------- #
# Loading — the single choke point that guarantees test rows are never read
# --------------------------------------------------------------------------- #
def _read_train_val(columns: list[str] | None = None) -> pd.DataFrame:
    """Read features.parquet filtered to train+val ONLY (test never touched)."""
    if not FEATURES_PARQUET.exists():
        raise FileNotFoundError(
            f"{FEATURES_PARQUET} not found. Run the Phase 2/2.5 pipeline first."
        )
    return pd.read_parquet(
        FEATURES_PARQUET,
        columns=columns,
        filters=[("split", "in", ["train", "val"])],
    )


def load_train_val(columns: list[str] | None = None) -> pd.DataFrame:
    """Public loader for train+val rows (test-split rows are excluded by design)."""
    return _read_train_val(columns)


# --------------------------------------------------------------------------- #
# Adversarial scoring
# --------------------------------------------------------------------------- #
def candidate_features(df: pd.DataFrame) -> list[str]:
    """All columns except the unconditional and structural exclusions."""
    excluded = set(EXCLUDED_UNCONDITIONAL) | set(EXCLUDED_STRUCTURAL)
    return [c for c in df.columns if c not in excluded]


def univariate_adversarial_auc(feature: pd.Series, y: np.ndarray) -> float:
    """
    Direction-agnostic single-feature ROC-AUC for separating train (0) vs val (1).

    ``max(auc, 1 - auc)`` so a feature that separates in either direction scores
    high. Missing values are filled with a below-range sentinel so that
    *differential missingness* (a feature missing more in val than train) is
    itself captured as drift, not silently dropped.
    """
    x = feature
    if x.isna().any():
        lo = x.min()
        x = x.fillna((lo - 1) if pd.notna(lo) else 0)
    if x.nunique() <= 1:
        return 0.5  # constant feature separates nothing
    auc = roc_auc_score(y, x.astype("float64"))
    return float(max(auc, 1.0 - auc))


def overall_adversarial_auc(X: pd.DataFrame, y: np.ndarray, n_splits: int = 5) -> float:
    """Out-of-fold AUC of a full XGBoost model predicting the train/val label."""
    oof = np.zeros(len(y), dtype="float64")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            eval_metric="auc",
            n_jobs=-1,
            random_state=42,
        )
        model.fit(X.iloc[tr], y[tr])
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
        log.info("  fold %d/%d AUC = %.4f", fold, n_splits, roc_auc_score(y[va], oof[va]))
    return float(roc_auc_score(y, oof))


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run() -> dict:
    df = load_train_val()
    n_train = int((df["split"] == "train").sum())
    n_val = int((df["split"] == "val").sum())
    log.info("Loaded train+val only: %s train + %s val = %s rows (test never loaded)",
             f"{n_train:,}", f"{n_val:,}", f"{len(df):,}")

    y = (df["split"] == "val").to_numpy().astype("int8")
    feats = candidate_features(df)

    X = df[feats].copy()
    for c in X.select_dtypes(include=["bool"]).columns:  # XGBoost wants numeric
        X[c] = X[c].astype("int8")

    log.info("Scoring %d candidate features (test/DT/ID excluded)...", len(feats))
    per_feature = []
    for c in feats:
        auc = univariate_adversarial_auc(df[c], y)
        per_feature.append({"feature": c, "adversarial_auc": round(auc, 4),
                            "flagged": auc > DRIFT_THRESHOLD})
    per_feature.sort(key=lambda d: d["adversarial_auc"], reverse=True)

    log.info("Computing overall adversarial AUC (all candidates) via 5-fold XGBoost OOF...")
    overall = overall_adversarial_auc(X, y)
    log.info("Overall adversarial AUC (all candidates) = %.4f", overall)

    flagged = [d["feature"] for d in per_feature if d["flagged"]]
    log.info("High-drift features (univariate AUC > %.2f): %d", DRIFT_THRESHOLD, len(flagged))
    for d in per_feature:
        if d["flagged"]:
            log.info("  FLAG %-28s auc=%.4f", d["feature"], d["adversarial_auc"])

    # ---- manifest set: drop flagged (unless explicitly kept with a reason) ----
    dropped = [f for f in flagged if f not in KEEP_DESPITE_FLAG]
    kept_despite = {f: KEEP_DESPITE_FLAG[f] for f in flagged if f in KEEP_DESPITE_FLAG}
    allowed = [f for f in feats if f not in dropped]

    # Residual separability once the flagged drifters are removed — the number
    # that actually describes the frozen modeling set (vs the all-candidate AUC).
    log.info("Computing residual adversarial AUC on the %d allowed features...", len(allowed))
    overall_allowed = overall_adversarial_auc(X[allowed], y)
    log.info("Residual adversarial AUC (allowed set) = %.4f", overall_allowed)

    # ---- report ----
    report = {
        "overall_adversarial_auc_all_candidates": round(overall, 4),
        "overall_adversarial_auc_allowed_set": round(overall_allowed, 4),
        "n_train": n_train,
        "n_val": n_val,
        "n_rows": len(df),
        "drift_threshold": DRIFT_THRESHOLD,
        "per_feature_method": (
            "single-feature univariate ROC-AUC, direction-agnostic max(auc, 1-auc); "
            "missing filled with a below-range sentinel to capture differential "
            "missingness. Chosen over full-model importance (confounded by "
            "correlated features, no per-feature AUC) and per-feature XGBoost "
            "(far slower, negligible gain on monotonic tabular features)."
        ),
        "n_flagged": len(flagged),
        "interpretation": (
            "Univariate flagging caught the individually-strongest drifting "
            "features. That dropping them barely moved the combined AUC "
            f"({round(overall, 4)} -> {round(overall_allowed, 4)}) shows the "
            "separability comes from MANY features acting jointly, not one or two "
            "bad actors — the known blind spot of per-feature flagging vs. "
            "full-model importance. This residual is an EXPECTED property of using "
            "inherently time-cumulative features (uid/card prior counts) by design "
            "— a live model sees the same growth daily — and is NOT evidence the "
            "feature set generalizes cleanly. Whether the model over-relies on "
            "'how much history exists' as a shortcut is deferred to a later SHAP "
            "analysis."
        ),
        "features": per_feature,
    }
    ADV_REPORT_JSON.write_text(json.dumps(report, indent=2))
    log.info("Wrote %s", ADV_REPORT_JSON.relative_to(PROJECT_ROOT))

    # ---- manifest ----
    excluded_columns = {**EXCLUDED_UNCONDITIONAL, **EXCLUDED_STRUCTURAL}
    auc_by_feat = {d["feature"]: d["adversarial_auc"] for d in per_feature}
    for f in dropped:
        excluded_columns[f] = f"high-drift: adversarial AUC={auc_by_feat[f]} > {DRIFT_THRESHOLD}"

    manifest = {
        "target": TARGET,
        "allowed_features": allowed,
        "n_allowed": len(allowed),
        "excluded_columns": excluded_columns,
        "kept_despite_flag": kept_despite,
        "adversarial": {
            "overall_auc_all_candidates": round(overall, 4),
            "overall_auc_allowed_set": round(overall_allowed, 4),
            "threshold": DRIFT_THRESHOLD,
            "method": "univariate direction-agnostic ROC-AUC",
        },
        "note": "Built from train+val only; test-split rows were never loaded. "
                "Phase 4 must use allowed_features verbatim — no ad hoc selection.",
    }
    FEATURE_MANIFEST_JSON.write_text(json.dumps(manifest, indent=2))
    log.info("Manifest: %d allowed features, %d dropped, %d kept-despite-flag -> %s",
             len(allowed), len(dropped), len(kept_despite),
             FEATURE_MANIFEST_JSON.relative_to(PROJECT_ROOT))
    log.info("Phase 3 (adversarial validation) complete.")
    return report


if __name__ == "__main__":
    run()
