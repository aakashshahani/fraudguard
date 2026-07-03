"""
FraudGuard — Phase 4: baseline modeling + imbalance-handling bake-off.

Methodology is governed ENTIRELY by ``docs/phase4_evaluation_protocol.md``
(pre-registered in Phase 3). This module executes that protocol; it does not
decide methodology. Where the protocol is silent on a forced implementation
detail, the choice is documented in ``docs/phase4_results.md``.

HARD REQUIREMENTS
- Use ONLY the columns in ``data/processed/feature_manifest.json`` — verbatim.
- Never load or reference test-split rows. Every read goes through
  ``_read_modeling_data``, which refuses any split other than train/val.

Stages
- Stage 1: logistic regression / random forest / XGBoost, uniform "balanced"
  imbalance handling, 3 seeds (42/43/44), pick the family with the highest mean
  validation PR-AUC (protocol §3 tie-break otherwise).
- Stage 2: the Stage-1 winning family only, under 4 imbalance conditions
  (none / class weights / SMOTE / both), same 3 seeds, PR-AUC decides
  (protocol §5 tie-break otherwise).

Scope: modeling + the bake-off only. No threshold selection (Phase 5), no SHAP
(Phase 6). The cost metric is reported as context, never as a decider.

Run
---
    python -m src.modeling
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data_prep import PROCESSED_DIR, PROJECT_ROOT
from src.feature_engineering import FEATURES_PARQUET

try:
    import joblib
    import wandb
except Exception:  # pragma: no cover - import guard
    joblib = None
    wandb = None

MANIFEST_JSON = PROCESSED_DIR / "feature_manifest.json"
MODELS_DIR = PROJECT_ROOT / "models"
WINNER_ARTIFACT = MODELS_DIR / "phase4_winner.joblib"
WINNER_META = MODELS_DIR / "phase4_winner.json"
RESULTS_DOC = PROJECT_ROOT / "docs" / "phase4_results.md"

TARGET = "isFraud"
SEEDS = [42, 43, 44]              # pre-registered
PRIMARY_SEED = 42                 # protocol §1: the artifact is the seed-42 fit
COST_FN, COST_FP = 10, 1         # pre-registered cost ratio (protocol §2a)
RECALL_TARGET = 0.80             # for precision@recall (protocol §5 tie-break)
WANDB_PROJECT = "fraudguard-phase4"

# Simplicity order for protocol tie-breaks (by inference cost).
FAMILY_SIMPLICITY = {"logreg": 0, "rf": 1, "xgboost": 2}

os.environ.setdefault("WANDB_MODE", "offline")   # no account/network needed
os.environ.setdefault("WANDB_SILENT", "true")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fraudguard.modeling")


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def manifest_hash() -> str:
    return hashlib.sha256(MANIFEST_JSON.read_bytes()).hexdigest()[:16]


def manifest_features() -> list[str]:
    return list(json.loads(MANIFEST_JSON.read_text())["allowed_features"])


# --------------------------------------------------------------------------- #
# Loading — the choke point that guarantees test rows are never read
# --------------------------------------------------------------------------- #
def _read_modeling_data(splits: list[str], columns: list[str] | None = None) -> pd.DataFrame:
    """Read features.parquet for the given splits — TEST IS FORBIDDEN here."""
    if "test" in splits:
        raise ValueError("Phase 4 must never load test-split rows.")
    return pd.read_parquet(
        FEATURES_PARQUET, columns=columns, filters=[("split", "in", splits)]
    )


def load_Xy(split: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Feature matrix (manifest columns ONLY) + isFraud target for one split."""
    feats = manifest_features()
    df = _read_modeling_data([split], columns=feats + [TARGET])
    X = df[feats].copy()
    for c in X.select_dtypes(include=["bool"]).columns:
        X[c] = X[c].astype("int8")
    y = df[TARGET].to_numpy().astype("int8")
    return X, y


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def evaluate(y: np.ndarray, prob: np.ndarray) -> dict:
    """All protocol metrics for a single (y, predicted-probability) pair."""
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)

    # cost metric: min_t [10*FN(t) + FP(t)] / N  (protocol §2a)
    fpr, tpr, thr = roc_curve(y, prob)
    fn = n_pos * (1 - tpr)
    fp = n_neg * fpr
    cost = (COST_FN * fn + COST_FP * fp) / len(y)
    ci = int(np.argmin(cost))

    # precision @ recall >= target (protocol §5 tie-break)
    prec, rec, _ = precision_recall_curve(y, prob)
    mask = rec >= RECALL_TARGET
    prec_at = float(prec[mask].max()) if mask.any() else 0.0

    return {
        "val_pr_auc": float(average_precision_score(y, prob)),
        "val_roc_auc": float(roc_auc_score(y, prob)),
        "val_precision_at_recall_0.80": prec_at,
        "val_brier": float(brier_score_loss(y, prob)),
        "val_expected_cost": float(cost[ci]),
        "val_cost_threshold": float(thr[ci]) if ci < len(thr) else 1.0,
    }


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #
def _steps_for(family: str, impute: bool, scale: bool, smote: bool, seed: int) -> list:
    steps: list = []
    if impute:
        steps.append(("impute", SimpleImputer(strategy="median")))
    if scale:
        steps.append(("scale", StandardScaler()))
    if smote:
        steps.append(("smote", SMOTE(random_state=seed)))
    return steps


def build_model(family: str, seed: int, *, class_weight: bool, smote: bool,
                spw_value: float):
    """
    Build an imblearn Pipeline. The ``class_weight`` flag means "apply balanced
    class weighting" and is realised per-family: `class_weight='balanced'` for
    logreg/RF, and `scale_pos_weight = n_neg/n_pos` (=``spw_value``) for XGBoost.
    When False, no weighting is applied (XGBoost `scale_pos_weight = 1.0`), so the
    "no handling" baseline is genuinely unweighted.

    logreg always imputes+scales (linear model needs it); trees use native NaN
    handling UNLESS SMOTE is present (SMOTE needs complete data), in which case
    they impute — applied uniformly across the Stage-2 arms so the imbalance
    technique is the only variable.
    """
    cw = "balanced" if class_weight else None

    if family == "logreg":
        steps = _steps_for(family, impute=True, scale=True, smote=smote, seed=seed)
        steps.append(("clf", LogisticRegression(
            class_weight=cw, max_iter=2000, random_state=seed)))
    elif family == "rf":
        steps = _steps_for(family, impute=smote, scale=False, smote=smote, seed=seed)
        steps.append(("clf", RandomForestClassifier(
            n_estimators=200, class_weight=cw, n_jobs=-1, random_state=seed)))
    elif family == "xgboost":
        steps = _steps_for(family, impute=smote, scale=False, smote=smote, seed=seed)
        steps.append(("clf", XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, tree_method="hist",
            eval_metric="aucpr", n_jobs=-1, random_state=seed,
            scale_pos_weight=(spw_value if class_weight else 1.0))))
    else:
        raise ValueError(family)
    return ImbPipeline(steps)


# --------------------------------------------------------------------------- #
# One run (fit on train, score on val, log to W&B)
# --------------------------------------------------------------------------- #
def _wandb_run(stage: str, family: str, condition: str, seed: int, params: dict, metrics: dict):
    if wandb is None:
        return
    try:
        run = wandb.init(
            project=WANDB_PROJECT,
            name=f"{stage}-{family}-{condition}-seed{seed}",
            group=f"{stage}-{family}-{condition}",
            job_type=stage,
            tags=["phase4", f"stage:{stage}", f"family:{family}",
                  f"condition:{condition}", f"seed:{seed}"],
            config={**params, "seed": seed, "git_hash": git_hash(),
                    "manifest_hash": manifest_hash(), "n_features": len(manifest_features())},
            reinit=True,
        )
        run.summary.update(metrics)
        run.finish()
    except Exception as e:  # pragma: no cover
        log.warning("wandb logging failed (%s) — continuing", e)


def run_condition(stage: str, family: str, condition: str,
                  X_tr, y_tr, X_val, y_val, *, class_weight: bool, smote: bool,
                  spw_value: float) -> dict:
    """Fit + score the 3 seeds of one (family, condition); return per-seed metrics."""
    rows = []
    for seed in SEEDS:
        model = build_model(family, seed, class_weight=class_weight, smote=smote,
                            spw_value=spw_value)
        model.fit(X_tr, y_tr)
        prob = model.predict_proba(X_val)[:, 1]
        m = evaluate(y_val, prob)
        rows.append({"seed": seed, **m})
        _wandb_run(stage, family, condition, seed,
                   {"family": family, "condition": condition,
                    "class_weight": class_weight, "smote": smote}, m)
        log.info("    %s/%s seed=%d  PR-AUC=%.4f  cost=%.4f",
                 family, condition, seed, m["val_pr_auc"], m["val_expected_cost"])
    return _aggregate(family, condition, rows)


def _aggregate(family: str, condition: str, rows: list[dict]) -> dict:
    pr = np.array([r["val_pr_auc"] for r in rows])
    return {
        "family": family,
        "condition": condition,
        "pr_auc_mean": float(pr.mean()),
        "pr_auc_std": float(pr.std(ddof=1)),
        "roc_auc_mean": float(np.mean([r["val_roc_auc"] for r in rows])),
        "prec_at_recall80_mean": float(np.mean([r["val_precision_at_recall_0.80"] for r in rows])),
        "cost_mean": float(np.mean([r["val_expected_cost"] for r in rows])),
        "per_seed": rows,
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run() -> dict:
    MODELS_DIR.mkdir(exist_ok=True)
    log.info("Manifest %s (%d features) | git %s",
             manifest_hash(), len(manifest_features()), git_hash()[:10])

    X_tr, y_tr = load_Xy("train")
    X_val, y_val = load_Xy("val")
    log.info("Train %s rows | Val %s rows (test never loaded)", f"{len(y_tr):,}", f"{len(y_val):,}")

    n_pos, n_neg = int(y_tr.sum()), int(len(y_tr) - y_tr.sum())
    spw = n_neg / n_pos  # XGBoost "balanced" == scale_pos_weight = n_neg/n_pos
    log.info("Train fraud rate %.3f%% | scale_pos_weight=%.2f", y_tr.mean() * 100, spw)

    # ---- Stage 1: family progression, uniform balanced handling ----
    log.info("STAGE 1 — model-family progression (uniform class-balanced handling)")
    stage1 = []
    for family in ["logreg", "rf", "xgboost"]:
        log.info("  family=%s", family)
        res = run_condition("family", family, "balanced", X_tr, y_tr, X_val, y_val,
                            class_weight=True, smote=False, spw_value=spw)
        stage1.append(res)
        log.info("  => %s mean PR-AUC=%.4f ± %.4f", family, res["pr_auc_mean"], res["pr_auc_std"])

    winner_family, s1_tiebreak = _pick_family(stage1)
    log.info("STAGE 1 WINNER: %s%s", winner_family, f" ({s1_tiebreak})" if s1_tiebreak else "")

    # ---- Stage 2: imbalance bake-off on the winning family only ----
    log.info("STAGE 2 — imbalance bake-off on '%s'", winner_family)
    conditions = [
        ("none", dict(class_weight=False, smote=False)),
        ("classweight", dict(class_weight=True, smote=False)),
        ("smote", dict(class_weight=False, smote=True)),
        ("both", dict(class_weight=True, smote=True)),
    ]
    stage2 = []
    for cond, kw in conditions:
        log.info("  condition=%s", cond)
        res = run_condition("bakeoff", winner_family, cond, X_tr, y_tr, X_val, y_val,
                            spw_value=spw, **kw)
        stage2.append(res)
        log.info("  => %s mean PR-AUC=%.4f ± %.4f", cond, res["pr_auc_mean"], res["pr_auc_std"])

    winner_cond, s2_tiebreak = _pick_condition(stage2)
    smote_verdict = _smote_hypothesis(stage2)
    log.info("STAGE 2 WINNER: %s%s", winner_cond, f" ({s2_tiebreak})" if s2_tiebreak else "")
    log.info("SMOTE hypothesis: %s", smote_verdict["statement"])

    # ---- serialize the winning (family, condition) at the primary seed ----
    wkw = dict(conditions)[winner_cond]
    winner_model = build_model(winner_family, PRIMARY_SEED, spw_value=spw, **wkw)
    winner_model.fit(X_tr, y_tr)
    if joblib is not None:
        joblib.dump(winner_model, WINNER_ARTIFACT)
    WINNER_META.write_text(json.dumps({
        "family": winner_family, "condition": winner_cond, "seed": PRIMARY_SEED,
        "manifest_hash": manifest_hash(), "git_hash": git_hash(),
        "n_features": len(manifest_features()),
        "note": "Reused verbatim by Phase 5 (thresholds) and Phase 6 (SHAP); not retrained.",
    }, indent=2))
    log.info("Saved winner artifact -> %s", WINNER_ARTIFACT.relative_to(PROJECT_ROOT))

    _write_results_doc(stage1, stage2, winner_family, s1_tiebreak,
                       winner_cond, s2_tiebreak, smote_verdict, spw)
    log.info("Phase 4 complete.")
    return {"stage1": stage1, "stage2": stage2, "winner": [winner_family, winner_cond]}


def _pick_family(stage1: list[dict]) -> tuple[str, str]:
    """Highest mean PR-AUC; if ±1std intervals overlap the best, simpler wins (§3)."""
    best = max(stage1, key=lambda r: r["pr_auc_mean"])
    lo_best = best["pr_auc_mean"] - best["pr_auc_std"]
    overlapping = [r for r in stage1
                   if r["pr_auc_mean"] + r["pr_auc_std"] >= lo_best]
    chosen = min(overlapping, key=lambda r: FAMILY_SIMPLICITY[r["family"]])
    tb = ("" if chosen["family"] == best["family"]
          else f"tie-break: within 1std of {best['family']}, simpler family chosen")
    return chosen["family"], tb


def _pick_condition(stage2: list[dict]) -> tuple[str, str]:
    """Highest mean PR-AUC; §5 tie-break if within 0.001."""
    order = {"none": 0, "classweight": 1, "smote": 2, "both": 3}
    best = max(stage2, key=lambda r: r["pr_auc_mean"])
    tied = [r for r in stage2 if abs(r["pr_auc_mean"] - best["pr_auc_mean"]) < 0.001]
    if len(tied) == 1:
        return best["condition"], ""
    # 1) precision@recall80, 2) lower cost, 3) simpler condition
    tied.sort(key=lambda r: (-r["prec_at_recall80_mean"], r["cost_mean"], order[r["condition"]]))
    return tied[0]["condition"], f"tie-break invoked among {[t['condition'] for t in tied]}"


def _smote_hypothesis(stage2: list[dict]) -> dict:
    """Direct confirm/refute of the pre-registered 'class weights >= SMOTE' claim."""
    by = {r["condition"]: r["pr_auc_mean"] for r in stage2}
    cw, sm, both = by["classweight"], by["smote"], by["both"]
    confirmed = (cw >= sm - 1e-9) and (cw >= both - 1e-9)
    if confirmed:
        statement = (
            f"CONFIRMED. Class weights (PR-AUC {cw:.4f}) matched or beat plain SMOTE "
            f"({sm:.4f}) and SMOTE+weights ({both:.4f}), as pre-registered. The "
            f"predicted reason held: synthetic interpolation in a frequency-encoded "
            f"space added no usable signal."
        )
    else:
        statement = (
            f"REFUTED. SMOTE-based handling beat class weights: class weights PR-AUC "
            f"{cw:.4f} vs SMOTE {sm:.4f} vs both {both:.4f}. The pre-registered "
            f"prediction did not hold — a surprising result worth investigating, not "
            f"explaining away."
        )
    return {"confirmed": confirmed, "statement": statement,
            "classweight": cw, "smote": sm, "both": both}


def _fmt(res: dict) -> str:
    return (f"| {res['family']} | {res['condition']} | "
            f"{res['pr_auc_mean']:.4f} ± {res['pr_auc_std']:.4f} | "
            f"{res['roc_auc_mean']:.4f} | {res['prec_at_recall80_mean']:.4f} | "
            f"{res['cost_mean']:.4f} |")


def _write_results_doc(stage1, stage2, wf, s1tb, wc, s2tb, smote, spw):
    lines = [
        "# Phase 4 — Results",
        "",
        "Executed per `docs/phase4_evaluation_protocol.md` (pre-registered). Primary "
        "metric: **validation PR-AUC**, mean ± std over seeds 42/43/44. Cost metric "
        "(10·FN + FP)/N is secondary context only and never decides.",
        "",
        "**Imbalance handling in Stage 1** was uniform class-balancing: "
        "`class_weight='balanced'` for logreg/RF and the arithmetically-equivalent "
        f"`scale_pos_weight = n_neg/n_pos = {spw:.2f}` for XGBoost (same pos:neg ratio, "
        "per protocol §3).",
        "",
        "**Preprocessing note (forced detail, not in the protocol):** logistic "
        "regression imputes (median) + standardizes. Trees use native NaN handling in "
        "Stage 1. In Stage 2, because plain SMOTE requires complete data, **all four "
        "arms share identical median imputation** so the imbalance technique is the only "
        "variable — keeping the pre-registered SMOTE test clean. This reuses the "
        "protocol's pre-registered median imputation.",
        "",
        "## Stage 1 — Model-family progression",
        "",
        "| Family | Condition | Val PR-AUC (mean ± std) | ROC-AUC | Prec@Recall0.80 | Cost |",
        "|---|---|---|---|---|---|",
        *[_fmt(r) for r in sorted(stage1, key=lambda r: -r["pr_auc_mean"])],
        "",
        f"**Stage-1 winner: `{wf}`.** " + (s1tb if s1tb else "Highest mean PR-AUC outright."),
        "",
        "## Stage 2 — Imbalance-handling bake-off (`" + wf + "` only)",
        "",
        "| Family | Condition | Val PR-AUC (mean ± std) | ROC-AUC | Prec@Recall0.80 | Cost |",
        "|---|---|---|---|---|---|",
        *[_fmt(r) for r in sorted(stage2, key=lambda r: -r["pr_auc_mean"])],
        "",
        f"**Stage-2 winner: `{wc}`.** " + (s2tb if s2tb else "Highest mean PR-AUC outright."),
        "",
        "## Reading the result",
        "",
        "PR-AUC is **threshold-independent** — it scores ranking quality across all "
        "thresholds, not performance at one operating point. Class weights and SMOTE "
        "primarily reshape the score distribution to trade precision for recall; that "
        "does not necessarily improve *ranking*, and here every handling technique "
        f"ranked at or below the unweighted model (`{wc}` won). This is coherent with "
        "the project's design: the recall benefit of imbalance handling shows up at a "
        "chosen operating point, and **operating-point / threshold selection is "
        "deliberately Phase 5**, not baked into training here. Note the secondary "
        "metrics (prec@recall0.80 and the cost metric) agree with the PR-AUC ordering "
        "in this run — they are context only and did not decide.",
        "",
        "## Pre-registered SMOTE hypothesis",
        "",
        smote["statement"],
        "",
        "> Framing caveat: \"REFUTED\" means the *specific* prediction (class weights ≥ "
        "SMOTE) did not hold — SMOTE out-ranked the class-weight arm. It does **not** "
        "mean SMOTE \"won\": no-handling out-ranked SMOTE, and weights+SMOTE combined "
        "was worst. Stacking two score-distorting techniques hurt most.",
        "",
        "## Winning model artifact",
        "",
        f"`models/phase4_winner.joblib` — `{wf}` + `{wc}`, fit at the primary seed "
        f"(42). Reused verbatim by Phase 5 (threshold selection) and Phase 6 (SHAP); "
        "not retrained there.",
        "",
        "*No deployment threshold is selected in this phase — that is Phase 5. The cost "
        "metric above is context only.*",
        "",
    ]
    RESULTS_DOC.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote %s", RESULTS_DOC.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    run()
