# Phase 4 — Pre-registered Evaluation Protocol

**Status: pre-registered.** This document is written and committed in Phase 3,
**before any Phase 4 model is trained**, so that the metric, the comparison
design, and the winner rule are provably decided before any result exists. Its
purpose is to make Phase 4 a pre-committed experiment, not a post-hoc search for
the best-looking number. Any deviation from this protocol during Phase 4 must be
recorded explicitly in the Phase 4 writeup with its justification.

---

## 1. Data contract (non-negotiable)

- **Features:** exactly the `allowed_features` list in
  `data/processed/feature_manifest.json`, loaded **verbatim**. No ad hoc feature
  selection, addition, or transformation happens in Phase 4.
- **Target:** `isFraud`.
- **Train:** the `train` split only, for fitting.
- **Validation:** the `val` split only, for **all** model/technique selection and
  threshold tuning.
- **Test:** the `test` split stays **sealed**. It is not loaded, inspected, or
  scored until a single final model has been chosen by this protocol. Exactly one
  test evaluation is permitted, at the very end, and it selects nothing.
- **Reproducibility:** fixed seed `42` everywhere; the manifest is content-hashed
  and the hash logged with every run.

## 2. Primary metric

- **PR-AUC (average precision), computed on the validation split only.**
- Rationale: at a 3.5% fraud base rate, ROC-AUC is inflated by the large negative
  class; PR-AUC reflects performance on the positive (fraud) class, which is what
  matters. ROC-AUC may be logged as a secondary read but **never** decides
  anything.
- Secondary (reported, non-deciding): precision@recall=0.80, ROC-AUC, Brier score.

## 3. Stage 1 — Model-family progression (pick the strongest family)

Run three families, **each with simple default imbalance handling**, purely to
find the strongest family. This is *not* the imbalance experiment.

| Family | Default imbalance handling |
|---|---|
| Logistic regression | `class_weight="balanced"` (with standardization + median imputation, since linear models need it) |
| Random forest | `class_weight="balanced"` |
| XGBoost | `scale_pos_weight = n_negative / n_positive` (train split) |

- Each is trained on `train`, scored on `val` by PR-AUC.
- **Stage-1 winner = highest validation PR-AUC.** This single family advances to
  Stage 2. The other families are not carried forward.

## 4. Stage 2 — Imbalance-handling bake-off (winning family ONLY)

Run the Stage-1 winning family under four imbalance conditions — **only** on that
one family, deliberately **not** a full family × technique cross-product (that
would multiply runs without answering the question, and conflate family choice
with technique choice):

1. **No handling** — baseline / reference (no weights, no resampling).
2. **Class weights only** — the family's native cost reweighting.
3. **SMOTE only** — synthetic minority oversampling on the **train split only**
   (never fit on or applied to val/test).
4. **Both** — class weights + SMOTE combined.

All four are scored on `val` by PR-AUC.

## 5. Winner rule (imbalance conditions)

- **Winner = highest validation PR-AUC.**
- **Tie-break (explicit):** two conditions are "tied" if their validation PR-AUC
  differ by **< 0.001 (absolute)**. Ties are broken in this fixed order:
  1. Higher **precision@recall=0.80** on validation.
  2. If still tied, prefer the **simpler / more robust** condition, in the order
     `no handling` > `class weights` > `SMOTE` > `both` — favouring no synthetic
     data (more reproducible, no train/serve distribution mismatch) and fewer
     moving parts.
- The winning `(family, condition)` pair is the single final model. Only then is
  the sealed test split scored, once, for the final reported number.

## 6. Weights & Biases conventions

- **Project:** `fraudguard-phase4`.
- **Run name:** `{stage}-{family}-{condition}-seed{seed}`, e.g.
  `family-logreg-balanced-seed42`, `bakeoff-xgboost-classweight-seed42`,
  `bakeoff-xgboost-smote-seed42`, `final-xgboost-classweight-seed42`.
- **Tags:** `phase4`, one of `stage:family` / `stage:bakeoff` / `stage:final`,
  `family:<name>`, `condition:<name>`.
- **Grouping:** group by `stage` so the family progression and the bake-off read
  as two clean cohorts.
- **Logged per run:** the manifest content-hash, `n_features`, all model
  hyperparameters, the seed, and — as the summary metric — `val_pr_auc`
  (primary), with `val_roc_auc`, `val_precision_at_recall_0.80`, and `val_brier`
  as secondary.
- `wandb` will be added to `requirements.txt` in Phase 4 (it is intentionally not
  a dependency yet).

---

*Pre-registered in Phase 3. Feature set frozen in `feature_manifest.json`;
adversarial-validation report in `adversarial_validation_report.json`.*
