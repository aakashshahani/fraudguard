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
- Secondary (reported, non-deciding): precision@recall=0.80, ROC-AUC, Brier score,
  and the **cost metric** below.

### 2a. Cost-based secondary metric (reported, non-deciding)

Fraud is a cost-asymmetric problem, so alongside PR-AUC we report an explicit
expected-cost metric that a real fraud team reasons in:

- **Pre-registered cost ratio: a missed fraud (false negative) costs 10× a false
  alarm (false positive).** This ratio is fixed here, now, before any results —
  it is illustrative but committed, not tuned post-hoc.
- **Reported quantity:** the minimum expected total cost on the validation split,
  minimised over the decision threshold, i.e.
  `min_t [ 10 * FN(t) + 1 * FP(t) ] / N_val`, with the cost-minimising threshold
  reported alongside.
- This is **secondary and non-deciding** — PR-AUC still selects the winner. The
  cost metric is context for the eventual dedicated threshold-selection step and
  an amount-weighted variant is deferred to a later phase.

## 3. Stage 1 — Model-family progression (pick the strongest family)

Run three families, **each with simple default imbalance handling**, purely to
find the strongest family. This is *not* the imbalance experiment.

| Family | Default imbalance handling |
|---|---|
| Logistic regression | `class_weight="balanced"` (with standardization + median imputation, since linear models need it) |
| Random forest | `class_weight="balanced"` |
| XGBoost | `scale_pos_weight = n_negative / n_positive` (train split) |

- Each is trained on `train`, scored on `val` by PR-AUC.
- **Seed-averaging:** every family is run with **3 seeds** (`42, 43, 44`) and its
  score is the **mean validation PR-AUC across the 3 seeds**, with the standard
  deviation reported. This is required, not optional: because train and val are
  ~0.9995 adversarially separable, a single-seed gap between families could be
  noise. Deciding on a 3-seed mean makes "family A beat family B" a real claim.
- **Stage-1 winner = highest mean validation PR-AUC (across 3 seeds).** This
  single family advances to Stage 2. The other families are not carried forward.
  If two families' mean PR-AUC intervals overlap within 1 std, the simpler family
  (logistic regression > random forest > XGBoost, by inference cost) advances.

## 4. Stage 2 — Imbalance-handling bake-off (winning family ONLY)

Run the Stage-1 winning family under four imbalance conditions — **only** on that
one family, deliberately **not** a full family × technique cross-product (that
would multiply runs without answering the question, and conflate family choice
with technique choice):

1. **No handling** — baseline / reference (no weights, no resampling).
2. **Class weights only** — the family's native cost reweighting.
3. **SMOTE only** — **plain `SMOTE`** (imbalanced-learn), fit and applied on the
   **train split only**, never on val/test. `SMOTE` (not `SMOTENC`) is chosen
   deliberately: Phase 2 already encoded every categorical column to a numeric
   value, so there are no genuinely-categorical columns left for `SMOTENC` to
   protect — the matrix is fully numeric at this point in the pipeline.
4. **Both** — class weights + SMOTE combined.

All four use the Stage-1 winning family, are run with the **same 3 seeds**
(`42, 43, 44`), and are scored on `val` by **mean PR-AUC across seeds** (std
reported).

### Pre-registered prediction (stated before results exist)

We **predict class weights will match or beat SMOTE (and beat the SMOTE+weights
combo)**, for a stated reason: SMOTE synthesises minority points by interpolating
between existing fraud rows, but in this **frequency-/label-encoded, high-
cardinality** space, a point "between" two encoded categories is not semantically
meaningful, and interpolating across fraud cases blends them across the very
**temporal structure** this project has been careful to respect. SMOTE stays in
the bake-off deliberately — the goal is to **test** this hypothesis, not assume
it. If SMOTE loses as predicted, that is a confirmed, empirically-grounded result
("we hypothesised X for reason Y, tested it, confirmed it"); if it wins, that is a
surprising finding worth investigating, not something to explain away.

## 5. Winner rule (imbalance conditions)

- **Winner = highest mean validation PR-AUC (across the 3 seeds).**
- **Tie-break (explicit):** two conditions are "tied" if their mean validation
  PR-AUC differ by **< 0.001 (absolute)**. Ties are broken in this fixed order:
  1. Higher **precision@recall=0.80** on validation (mean across seeds).
  2. If still tied, lower pre-registered **expected cost** (§2a).
  3. If still tied, prefer the **simpler / more robust** condition, in the order
     `no handling` > `class weights` > `SMOTE` > `both` — favouring no synthetic
     data (more reproducible, no train/serve distribution mismatch) and fewer
     moving parts.
- The winning `(family, condition)` pair is the single final model. Only then is
  the sealed test split scored, once, for the final reported number.

## 6. Weights & Biases conventions

- **Project:** `fraudguard-phase4`.
- **Run name:** `{stage}-{family}-{condition}-seed{seed}`, one W&B run per seed,
  e.g. `family-logreg-balanced-seed42`, `bakeoff-xgboost-smote-seed43`,
  `final-xgboost-classweight-seed42`. Each `(stage, family, condition)` triple is
  a W&B **group** so its 3 seeds aggregate into one mean±std.
- **Tags:** `phase4`, one of `stage:family` / `stage:bakeoff` / `stage:final`,
  `family:<name>`, `condition:<name>`, `seed:<n>`.
- **Grouping:** `group = {stage}-{family}-{condition}` (aggregates seeds); the
  W&B `job_type` = `stage` so the family progression and bake-off read as two
  clean cohorts.
- **Logged per run:** the manifest content-hash, `n_features`, all model
  hyperparameters, the seed, and — as the summary metric — `val_pr_auc`
  (primary), with `val_roc_auc`, `val_precision_at_recall_0.80`, `val_brier`, and
  `val_expected_cost` as secondary. The per-group **mean/std of `val_pr_auc`** is
  what the winner rules in §3 and §5 read.
- **New Phase 4 dependencies (not added yet):** `wandb` (tracking) and
  `imbalanced-learn` (SMOTE) will be added to `requirements.txt` at the start of
  Phase 4. They are intentionally absent now.

---

*Pre-registered in Phase 3. Feature set frozen in `feature_manifest.json`;
adversarial-validation report in `adversarial_validation_report.json`.*
