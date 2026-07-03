# Phase 4 — Results

Executed per `docs/phase4_evaluation_protocol.md` (pre-registered). Primary metric: **validation PR-AUC**, mean ± std over seeds 42/43/44. Cost metric (10·FN + FP)/N is secondary context only and never decides.

**Imbalance handling in Stage 1** was uniform class-balancing: `class_weight='balanced'` for logreg/RF and the arithmetically-equivalent `scale_pos_weight = n_neg/n_pos = 27.43` for XGBoost (same pos:neg ratio, per protocol §3).

**Preprocessing note (forced detail, not in the protocol):** logistic regression imputes (median) + standardizes. Trees use native NaN handling in Stage 1. In Stage 2, because plain SMOTE requires complete data, **all four arms share identical median imputation** so the imbalance technique is the only variable — keeping the pre-registered SMOTE test clean. This reuses the protocol's pre-registered median imputation.

## Stage 1 — Model-family progression

| Family | Condition | Val PR-AUC (mean ± std) | ROC-AUC | Prec@Recall0.80 | Cost |
|---|---|---|---|---|---|
| xgboost | balanced | 0.5446 ± 0.0011 | 0.9072 | 0.1638 | 0.1682 |
| rf | balanced | 0.5406 ± 0.0020 | 0.9135 | 0.1777 | 0.1634 |
| logreg | balanced | 0.3623 ± 0.0000 | 0.8445 | 0.0914 | 0.2168 |

**Stage-1 winner: `xgboost`.** Highest mean PR-AUC outright.

**Secondary-metric divergence:** the PR-AUC winner is `xgboost`, but lowest cost is `rf` (0.1634 vs 0.1682) and best precision@recall0.80 is `rf` (0.1777 vs 0.1638). PR-AUC (ranking) and the fixed-operating-point metrics disagree here — exactly the tension Phase 5's threshold selection exists to resolve; the pre-registered decider is PR-AUC, so it stands.

*(Logistic regression uses the deterministic lbfgs solver, so its three seeds are identical — std 0.0000 is expected, not a stuck seed.)*

## Stage 2 — Imbalance-handling bake-off (`xgboost` only)

| Family | Condition | Val PR-AUC (mean ± std) | ROC-AUC | Prec@Recall0.80 | Cost |
|---|---|---|---|---|---|
| xgboost | none | 0.5897 ± 0.0038 | 0.9187 | 0.1957 | 0.1544 |
| xgboost | smote | 0.5752 ± 0.0051 | 0.9106 | 0.1793 | 0.1583 |
| xgboost | classweight | 0.5446 ± 0.0011 | 0.9072 | 0.1638 | 0.1682 |
| xgboost | both | 0.5335 ± 0.0018 | 0.8930 | 0.1412 | 0.1751 |

**Stage-2 winner: `none`.** Highest mean PR-AUC outright.

Secondary metrics concur: `none` is also lowest-cost and highest precision@recall0.80 — no operating-point/ranking tension here.

## Reading the result

PR-AUC is **threshold-independent** — it scores ranking quality across all thresholds, not performance at one operating point. Class weights and SMOTE primarily reshape the score distribution to trade precision for recall; that does not necessarily improve *ranking*, and here every handling technique ranked at or below the unweighted model (`none` won). This is coherent with the project's design: the recall benefit of imbalance handling shows up at a chosen operating point, and **operating-point / threshold selection is deliberately Phase 5**, not baked into training here. Where the fixed-operating-point secondary metrics disagree with the PR-AUC ordering (see the concordance notes under each stage), that is the ranking-vs-operating-point tension itself — surfaced, not smoothed over, and left for Phase 5 to resolve. The secondary metrics are context only; they did not decide.

## Pre-registered SMOTE hypothesis

REFUTED. SMOTE-based handling beat class weights: class weights PR-AUC 0.5446 vs SMOTE 0.5752 vs both 0.5335. The pre-registered prediction did not hold — a surprising result worth investigating, not explaining away.

> Framing caveat: "REFUTED" means the *specific* prediction (class weights ≥ SMOTE) did not hold — SMOTE out-ranked the class-weight arm. It does **not** mean SMOTE "won": no-handling out-ranked SMOTE, and weights+SMOTE combined was worst. Stacking two score-distorting techniques hurt most.

## Winning model artifact

`models/phase4_winner.joblib` — `xgboost` + `none`, fit at the primary seed (42). Reused verbatim by Phase 5 (threshold selection) and Phase 6 (SHAP); not retrained there.

*No deployment threshold is selected in this phase — that is Phase 5. The cost metric above is context only.*
