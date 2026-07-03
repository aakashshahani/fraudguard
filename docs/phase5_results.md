# Phase 5 — Threshold Selection & Final Evaluation

Sequencing enforced: Parts A–C are validation-only; the test split is unsealed exactly once, in Part D. The Phase 4 winner (`xgboost` + `none`, seed 42) is loaded from `models/phase4_winner.joblib` and **not retrained**.

## Part A — Cost-minimising threshold (validation, 10:1)

- **Chosen threshold: `0.0783`** (minimises (10·FN + FP)/N on validation).
- Cost `0.1517` | precision `0.3926` | recall `0.6604` | F1 `0.4925` (TP 2009, FP 3108, FN 1033).
- Full sweep: `reports/figures/phase5_threshold_sweep.csv`; plot: `reports/figures/phase5_threshold_cost.png`.

## Part B — Cost-ratio sensitivity (validation, context only)

The 10:1 decision from Part A stands; this only shows how sensitive the operating point is to the cost assumption.

| FN:FP ratio | Threshold | Precision | Recall | Cost/txn |
|---|---|---|---|---|
| 10:1 (primary) | 0.0783 | 0.3926 | 0.6604 | 0.1517 |
| 5:1 | 0.1224 | 0.5084 | 0.5901 | 0.0900 |
| 20:1 | 0.0342 | 0.2198 | 0.7784 | 0.2471 |
| 50:1 | 0.0153 | 0.1191 | 0.8912 | 0.4132 |

As FN gets more expensive (5:1 → 50:1) the threshold drops and recall rises at the cost of precision — the expected direction. Plot: `reports/figures/phase5_cost_sensitivity.png`.

## Part C — RandomForest secondary check (validation only)

Each model is compared **at its own cost-minimising operating point** (not at a shared threshold). RF is the exact Stage-1 config (`n_estimators=200`, `class_weight='balanced'`, seed 42), retrained on train only for this comparison and never saved.

| Model (own best point) | Threshold | Precision | Recall | Cost/txn |
|---|---|---|---|---|
| XGBoost + none (deployed) | 0.0783 | 0.3926 | 0.6604 | 0.1517 |
| RandomForest (secondary) | 0.1500 | 0.3086 | 0.6706 | 0.1647 |

**Resolves the Stage 1 divergence.** In Phase 4, cost favoured RF *only* when XGBoost carried `scale_pos_weight` (the balanced arm). The deployed model is `none`, and at each model's own best operating point XGBoost is cheaper (0.1517 vs RF 0.1647). The divergence was an artifact of the balanced XGBoost arm; the production model dominates RF on cost too, so there is nothing left to resolve in RF's favour.

## Part D — Final test evaluation (test unsealed once, read-only)

Scoring the deployed model at the Part-A threshold `0.0783` on the sealed test split:

| Metric | Test |
|---|---|
| PR-AUC | 0.5266 |
| Precision | 0.3179 |
| Recall | 0.6293 |
| F1 | 0.4224 |
| Cost/txn (10:1) | 0.1760 |

### Test vs validation PR-AUC — read in light of Phase 3

Test PR-AUC (0.5266) vs validation PR-AUC (0.5937) — a change of -0.0671. A drop of this size is **consistent with** Phase 3's finding that the feature set is ~0.9995 adversarially separable across time: the test period sits further in the future than validation, so history-accumulation features drift further and ranking degrades somewhat. It points to expected temporal decay, not a new problem.

*This is the final, one-time test evaluation. Whatever it shows, it stands.*
