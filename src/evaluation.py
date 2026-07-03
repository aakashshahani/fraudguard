"""
FraudGuard — Phase 5: threshold selection + final test evaluation.

Sequencing is load-bearing and enforced by the order of ``run()``:

    Part A  cost-minimising threshold on VALIDATION (test untouched)
    Part B  cost-ratio sensitivity on VALIDATION (context only; A's decision stands)
    Part C  RandomForest secondary check on VALIDATION (each model at its OWN best point)
    Part D  FINAL test evaluation — the test split is unsealed here, ONCE, read-only

The Phase 4 winner (`xgboost` + `none`, seed 42) is LOADED from
``models/phase4_winner.joblib`` and never retrained. The RF in Part C is retrained
only for a validation-only secondary comparison and is never saved or reused.

Run
---
    python -m src.evaluation
"""

from __future__ import annotations

import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.data_prep import PROCESSED_DIR, PROJECT_ROOT
from src.feature_engineering import FEATURES_PARQUET
from src.modeling import (
    TARGET,
    WINNER_ARTIFACT,
    build_model,
    load_Xy,
    manifest_features,
)

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

FIG_DIR = PROJECT_ROOT / "reports" / "figures"
SWEEP_CSV = FIG_DIR / "phase5_threshold_sweep.csv"
RESULTS_DOC = PROJECT_ROOT / "docs" / "phase5_results.md"
ADV_REPORT = PROCESSED_DIR / "adversarial_validation_report.json"

PRIMARY_FN_RATIO = 10          # pre-registered 10:1 (protocol §2a)
SENSITIVITY_RATIOS = [5, 20, 50]
PHASE4_VAL_PR_AUC_SEED42 = 0.5937   # winner artifact is seed 42
PHASE4_VAL_PR_AUC_MEAN = 0.5897     # 3-seed mean reported in phase4_results.md

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("fraudguard.evaluation")


# --------------------------------------------------------------------------- #
# Cost sweep (shared by A, B, C)
# --------------------------------------------------------------------------- #
def cost_sweep(y: np.ndarray, prob: np.ndarray, fn_ratio: int = PRIMARY_FN_RATIO,
               fp_cost: int = 1) -> pd.DataFrame:
    """
    Exact operating-point sweep. Candidate thresholds are the predicted
    probabilities themselves (so the true cost minimum can't fall between grid
    points). Cost is normalised per row: (fn_ratio*FN + fp_cost*FP) / N — the
    same normalisation as Phase 4's cost metric.
    """
    y = y.astype(int)
    order = np.argsort(-prob, kind="mergesort")     # high prob first
    ps, ys = prob[order], y[order]
    n, pos = len(y), int(y.sum())

    cum_tp = np.cumsum(ys)                            # TP among the top-k
    cum_fp = np.cumsum(1 - ys)                        # FP among the top-k
    tp, fp = cum_tp, cum_fp
    fn = pos - tp
    cost = (fn_ratio * fn + fp_cost * fp) / n
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(pos, 1)
    denom = precision + recall
    f1 = np.where(denom > 0, 2 * precision * recall / np.where(denom > 0, denom, 1), 0.0)

    df = pd.DataFrame({"threshold": ps, "cost": cost, "precision": precision,
                       "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn})
    # collapse tied thresholds to their fully-included operating point
    return df.drop_duplicates(subset="threshold", keep="last").reset_index(drop=True)


def best_operating_point(sweep: pd.DataFrame) -> dict:
    row = sweep.loc[sweep["cost"].idxmin()]
    return {"threshold": float(row["threshold"]), "cost": float(row["cost"]),
            "precision": float(row["precision"]), "recall": float(row["recall"]),
            "f1": float(row["f1"]), "tp": int(row["tp"]), "fp": int(row["fp"]),
            "fn": int(row["fn"])}


def metrics_at_threshold(y: np.ndarray, prob: np.ndarray, thr: float,
                         fn_ratio: int = PRIMARY_FN_RATIO) -> dict:
    y = y.astype(int)
    pred = (prob >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": thr, "precision": precision, "recall": recall, "f1": f1,
            "cost": (fn_ratio * fn + fp) / len(y), "tp": tp, "fp": fp, "fn": fn,
            "pr_auc": float(average_precision_score(y, prob))}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_winner():
    if joblib is None or not WINNER_ARTIFACT.exists():
        raise FileNotFoundError("Phase 4 winner artifact missing. Run `python -m src.modeling`.")
    return joblib.load(WINNER_ARTIFACT)


def _load_test_Xy() -> tuple[pd.DataFrame, np.ndarray]:
    """TEST loader — called ONLY in Part D. This is the one, deliberate unseal."""
    feats = manifest_features()
    df = pd.read_parquet(FEATURES_PARQUET, columns=feats + [TARGET],
                         filters=[("split", "in", ["test"])])
    X = df[feats].copy()
    for c in X.select_dtypes(include=["bool"]).columns:
        X[c] = X[c].astype("int8")
    return X, df[TARGET].to_numpy().astype("int8")


# --------------------------------------------------------------------------- #
# Parts
# --------------------------------------------------------------------------- #
def part_a(prob_val, y_val) -> tuple[dict, pd.DataFrame]:
    log.info("PART A — cost-minimising threshold on validation (10:1)")
    sweep = cost_sweep(y_val, prob_val, PRIMARY_FN_RATIO)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(SWEEP_CSV, index=False)
    best = best_operating_point(sweep)
    log.info("  chosen threshold=%.4f  cost=%.4f  precision=%.4f  recall=%.4f  f1=%.4f",
             best["threshold"], best["cost"], best["precision"], best["recall"], best["f1"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sweep["threshold"], sweep["cost"], color="steelblue", lw=1.5)
    ax.axvline(best["threshold"], color="red", ls="--",
               label=f"min-cost t={best['threshold']:.3f}")
    ax.scatter([best["threshold"]], [best["cost"]], color="red", zorder=5)
    ax.set_xlabel("Decision threshold"); ax.set_ylabel("Expected cost / txn  (10·FN + FP)/N")
    ax.set_title("Part A — cost vs threshold (validation, 10:1)")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "phase5_threshold_cost.png", dpi=120)
    plt.close(fig)
    return best, sweep


def part_b(prob_val, y_val, primary: dict) -> list[dict]:
    log.info("PART B — cost-ratio sensitivity on validation (context only)")
    rows = [{"ratio": f"{PRIMARY_FN_RATIO}:1 (primary)", **best_operating_point(
        cost_sweep(y_val, prob_val, PRIMARY_FN_RATIO))}]
    for r in SENSITIVITY_RATIOS:
        bp = best_operating_point(cost_sweep(y_val, prob_val, r))
        rows.append({"ratio": f"{r}:1", **bp})
    for row in sorted(rows, key=lambda d: float(str(d["ratio"]).split(":")[0])):
        log.info("  %-14s threshold=%.4f  precision=%.4f  recall=%.4f",
                 row["ratio"], row["threshold"], row["precision"], row["recall"])

    fig, ax = plt.subplots(figsize=(7, 4.2))
    thr = [r["threshold"] for r in rows]; rr = [float(str(r["ratio"]).split(":")[0]) for r in rows]
    ax.plot(rr, thr, "o-", color="darkorange")
    for r in rows:
        ax.annotate(f"P={r['precision']:.2f}\nR={r['recall']:.2f}",
                    (float(str(r["ratio"]).split(":")[0]), r["threshold"]),
                    textcoords="offset points", xytext=(6, -4), fontsize=8)
    ax.set_xlabel("FN:FP cost ratio"); ax.set_ylabel("Selected threshold")
    ax.set_title("Part B — how the chosen threshold moves with the cost assumption")
    fig.tight_layout(); fig.savefig(FIG_DIR / "phase5_cost_sensitivity.png", dpi=120)
    plt.close(fig)
    return rows


def part_c(X_tr, y_tr, X_val, y_val, xgb_best: dict) -> dict:
    log.info("PART C — RandomForest secondary check (validation only, retrain seed 42)")
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    rf = build_model("rf", 42, class_weight=True, smote=False, spw_value=spw)  # exact Stage-1 RF
    rf.fit(X_tr, y_tr)
    rf_prob = rf.predict_proba(X_val)[:, 1]
    rf_best = best_operating_point(cost_sweep(y_val, rf_prob, PRIMARY_FN_RATIO))
    log.info("  RF   own best: threshold=%.4f cost=%.4f precision=%.4f recall=%.4f",
             rf_best["threshold"], rf_best["cost"], rf_best["precision"], rf_best["recall"])
    log.info("  XGB  own best: cost=%.4f precision=%.4f recall=%.4f",
             xgb_best["cost"], xgb_best["precision"], xgb_best["recall"])
    resolved = xgb_best["cost"] <= rf_best["cost"]
    return {"rf": rf_best, "xgb": xgb_best, "xgb_cheaper": resolved}


def part_d(model, X_val, prob_val, y_val, threshold: float) -> dict:
    log.info("PART D — FINAL test evaluation (unsealing test, once, read-only)")
    X_test, y_test = _load_test_Xy()
    log.info("  test rows: %s (fraud rate %.3f%%)", f"{len(y_test):,}", y_test.mean() * 100)
    prob_test = model.predict_proba(X_test)[:, 1]

    test_m = metrics_at_threshold(y_test, prob_test, threshold, PRIMARY_FN_RATIO)
    val_pr_auc = float(average_precision_score(y_val, prob_val))
    log.info("  TEST  PR-AUC=%.4f  precision=%.4f  recall=%.4f  f1=%.4f  cost=%.4f",
             test_m["pr_auc"], test_m["precision"], test_m["recall"], test_m["f1"], test_m["cost"])
    log.info("  VAL   PR-AUC=%.4f (recomputed from artifact)", val_pr_auc)
    return {"test": test_m, "val_pr_auc": val_pr_auc}


# --------------------------------------------------------------------------- #
# Orchestration + reporting
# --------------------------------------------------------------------------- #
def _residual_adv_auc() -> float | None:
    if ADV_REPORT.exists():
        return json.loads(ADV_REPORT.read_text()).get("overall_adversarial_auc_allowed_set")
    return None


def run() -> dict:
    model = load_winner()
    X_val, y_val = load_Xy("val")
    X_tr, y_tr = load_Xy("train")
    prob_val = model.predict_proba(X_val)[:, 1]

    a_best, _ = part_a(prob_val, y_val)          # test untouched
    b_rows = part_b(prob_val, y_val, a_best)      # test untouched
    c = part_c(X_tr, y_tr, X_val, y_val, a_best)  # test untouched
    d = part_d(model, X_val, prob_val, y_val, a_best["threshold"])  # test unsealed here

    _write_doc(a_best, b_rows, c, d)
    log.info("Phase 5 complete.")
    return {"a": a_best, "b": b_rows, "c": c, "d": d}


def _write_doc(a, b, c, d):
    adv = _residual_adv_auc()
    gap = d["test"]["pr_auc"] - d["val_pr_auc"]
    drift_line = (
        f"Test PR-AUC ({d['test']['pr_auc']:.4f}) vs validation PR-AUC "
        f"({d['val_pr_auc']:.4f}) — a change of {gap:+.4f}. "
    )
    if adv is not None:
        if gap < -0.02:
            drift_line += (
                f"A drop of this size is **consistent with** Phase 3's finding that the "
                f"feature set is ~{adv} adversarially separable across time: the test "
                f"period sits further in the future than validation, so history-"
                f"accumulation features drift further and ranking degrades somewhat. "
                f"It points to expected temporal decay, not a new problem.")
        else:
            drift_line += (
                f"Despite Phase 3's ~{adv} adversarial separability (strong temporal "
                f"drift in the features), ranking held up on the further-future test "
                f"period — the drift did not translate into a large ranking loss, which "
                f"is a genuinely reassuring, and mildly surprising, result.")

    def brow(r):
        return (f"| {r['ratio']} | {r['threshold']:.4f} | {r['precision']:.4f} | "
                f"{r['recall']:.4f} | {r['cost']:.4f} |")

    lines = [
        "# Phase 5 — Threshold Selection & Final Evaluation",
        "",
        "Sequencing enforced: Parts A–C are validation-only; the test split is unsealed "
        "exactly once, in Part D. The Phase 4 winner (`xgboost` + `none`, seed 42) is "
        "loaded from `models/phase4_winner.joblib` and **not retrained**.",
        "",
        "## Part A — Cost-minimising threshold (validation, 10:1)",
        "",
        f"- **Chosen threshold: `{a['threshold']:.4f}`** (minimises (10·FN + FP)/N on validation).",
        f"- Cost `{a['cost']:.4f}` | precision `{a['precision']:.4f}` | recall "
        f"`{a['recall']:.4f}` | F1 `{a['f1']:.4f}` (TP {a['tp']}, FP {a['fp']}, FN {a['fn']}).",
        "- Full sweep: `reports/figures/phase5_threshold_sweep.csv`; plot: "
        "`reports/figures/phase5_threshold_cost.png`.",
        "",
        "## Part B — Cost-ratio sensitivity (validation, context only)",
        "",
        "The 10:1 decision from Part A stands; this only shows how sensitive the operating "
        "point is to the cost assumption.",
        "",
        "| FN:FP ratio | Threshold | Precision | Recall | Cost/txn |",
        "|---|---|---|---|---|",
        *[brow(r) for r in b],
        "",
        "As FN gets more expensive (5:1 → 50:1) the threshold drops and recall rises at the "
        "cost of precision — the expected direction. Plot: "
        "`reports/figures/phase5_cost_sensitivity.png`.",
        "",
        "## Part C — RandomForest secondary check (validation only)",
        "",
        "Each model is compared **at its own cost-minimising operating point** (not at a "
        "shared threshold). RF is the exact Stage-1 config (`n_estimators=200`, "
        "`class_weight='balanced'`, seed 42), retrained on train only for this comparison "
        "and never saved.",
        "",
        "| Model (own best point) | Threshold | Precision | Recall | Cost/txn |",
        "|---|---|---|---|---|",
        f"| XGBoost + none (deployed) | {a['threshold']:.4f} | {a['precision']:.4f} | "
        f"{a['recall']:.4f} | {a['cost']:.4f} |",
        f"| RandomForest (secondary) | {c['rf']['threshold']:.4f} | {c['rf']['precision']:.4f} | "
        f"{c['rf']['recall']:.4f} | {c['rf']['cost']:.4f} |",
        "",
        (f"**Resolves the Stage 1 divergence.** In Phase 4, cost favoured RF *only* when "
         f"XGBoost carried `scale_pos_weight` (the balanced arm). The deployed model is "
         f"`none`, and at each model's own best operating point XGBoost is "
         f"{'cheaper' if c['xgb_cheaper'] else 'NOT cheaper'} "
         f"({a['cost']:.4f} vs RF {c['rf']['cost']:.4f}). "
         + ("The divergence was an artifact of the balanced XGBoost arm; the production "
            "model dominates RF on cost too, so there is nothing left to resolve in RF's "
            "favour." if c["xgb_cheaper"] else
            "RF remains cheaper even here — the divergence is real and worth revisiting.")),
        "",
        "## Part D — Final test evaluation (test unsealed once, read-only)",
        "",
        f"Scoring the deployed model at the Part-A threshold `{a['threshold']:.4f}` on the "
        "sealed test split:",
        "",
        "| Metric | Test |",
        "|---|---|",
        f"| PR-AUC | {d['test']['pr_auc']:.4f} |",
        f"| Precision | {d['test']['precision']:.4f} |",
        f"| Recall | {d['test']['recall']:.4f} |",
        f"| F1 | {d['test']['f1']:.4f} |",
        f"| Cost/txn (10:1) | {d['test']['cost']:.4f} |",
        "",
        "### Test vs validation PR-AUC — read in light of Phase 3",
        "",
        drift_line,
        "",
        "*This is the final, one-time test evaluation. Whatever it shows, it stands.*",
        "",
    ]
    RESULTS_DOC.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote %s", RESULTS_DOC.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    run()
