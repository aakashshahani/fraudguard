# Phase 6 — Explainability & Drift Monitoring

Read-only phase: the finalized model (`xgboost` + `none`, seed 42), the Phase 5 threshold (`0.0783`), and the feature manifest are loaded and never modified, retrained, or re-tuned.

## Part A — SHAP explainability (validation only)

- Global beeswarm: `reports/figures/phase6_shap_summary.png`.
- Waterfalls: `phase6_waterfall_true_positive.png` (confident correct catch, p=1.000), `phase6_waterfall_false_negative.png` (near-miss fraud, p=0.078), `phase6_waterfall_false_positive.png` (confident wrong block, p=0.998).

### History-vs-content importance (quantified)

Summed mean|SHAP| over the 6 history/count features vs all other content features (of 438 total):

| Group | Share of total SHAP importance |
|---|---|
| History / count features | 4.9% |
| Transaction-content features | 95.1% |

Per-history-feature importance: `card1_prior_count` 0.1723, `uid_amt_expanding_mean` 0.0366, `v_missing_count` 0.0343, `uid_amt_expanding_std` 0.0328, `uid_prior_count` 0.0152, `has_identity` 0.0000

6 history features (1.4% of the columns) carry 4.9% of total SHAP importance — 3.6× their share under a uniform split. **History features are over-represented but not dominant:** they punch well above their weight per-feature, yet transaction-content features still hold the majority of total importance. Concentrated, not over-reliant.

This directly answers the question left open in Phase 3/5 — *is the model leaning on 'how much history exists' as a shortcut rather than genuine fraud signal?* **No.** History/count features carry under 5% of importance; the model is driven overwhelmingly by transaction content. (`card1_prior_count` is the one prominent history feature; the rest are minor.)

## Part B — Drift monitoring (validation vs test)

Reuses the Phase 3 adversarial machinery, now val-vs-test, reading **test feature values only** (never `isFraud`, never model predictions).

- **Overall val-vs-test adversarial AUC: 0.9999.** Phase 3's train-vs-val was 1.0 (all candidates) / 0.9995 (allowed set). Separability held steady (both ~perfectly separable) moving into the test period — the temporal drift Phase 3 identified persists val→test, it did not appear or vanish.

Top features by val-vs-test adversarial separability (with PSI):

| Feature | Adversarial AUC | PSI |
|---|---|---|
| D1_normalized *(dropped in Phase 3)* | 0.6692 | 1.2900 |
| P_emaildomain_prior_count *(dropped in Phase 3)* | 0.6353 | 5.1782 |
| id_31 | 0.5657 | 0.1028 |
| v_missing_count | 0.5497 | 0.1211 |
| card1_prior_count | 0.5281 | 0.0242 |
| V85 | 0.5267 | 0.0225 |
| V81 | 0.5265 | 0.0042 |
| V84 | 0.5264 | 0.0183 |
| V80 | 0.5263 | 0.0042 |
| V78 | 0.5260 | 0.0038 |
| V93 | 0.5260 | 0.0219 |
| V92 | 0.5259 | 0.0178 |

**PSI summary (val vs test):** 2 feature(s) significant (PSI ≥ 0.25, *alert*), 2 moderate (0.10–0.25, *investigate*).

**Crucially, both significant-PSI features (`D1_normalized`, `P_emaildomain_prior_count`) were already DROPPED in Phase 3** — they are not in the deployed manifest, so their (large) drift does not touch the model. This is a neat validation of Phase 3: the features flagged there as train-vs-val drift are exactly the ones that drift hardest val-vs-test. Among the 438 features the model actually uses, drift is mild — 0 significant, 2 moderate (`id_31`, `v_missing_count`).

History/count features (all deployed) and their PSI — note these are mostly **stable**, contradicting a naive 'the counts drift, so they drove the decline' story:

| History feature | Adversarial AUC | PSI |
|---|---|---|
| v_missing_count | 0.5497 | 0.1211 |
| card1_prior_count | 0.5281 | 0.0242 |
| has_identity | 0.5170 | 0.0074 |
| uid_prior_count | 0.5131 | 0.0080 |
| uid_amt_expanding_mean | 0.5029 | 0.0015 |
| uid_amt_expanding_std | 0.5009 | 0.0021 |

### Drift × importance cross-check (per-feature, not the category average)

"History is only 4.9%" is a category average; the sharp question is whether any *individual* feature is both **relied on** (high SHAP) and **drifting** (high PSI). Cross-referencing the two:

**Top SHAP drivers — are they stable?**

| Feature (top SHAP) | SHAP share | Rank | PSI |
|---|---|---|---|
| C13 | 3.80% | 1/438 | 0.0027 |
| TransactionAmt | 3.55% | 2/438 | 0.0005 |
| C14 | 3.42% | 3/438 | 0.0027 |
| C1 | 3.16% | 4/438 | 0.0021 |
| C5 | 3.13% | 5/438 | 0.0032 |
| card6 | 2.98% | 6/438 | 0.0529 |
| card1_prior_count | 2.89% | 7/438 | 0.0242 |
| card1 | 2.75% | 8/438 | 0.0022 |

**Deployed features that moderately+ drift — are they important?**

| Feature (PSI ≥ 0.10, deployed) | PSI | SHAP rank | SHAP share |
|---|---|---|---|
| v_missing_count | 0.1211 | 43/438 | 0.57% |
| id_31 | 0.1028 | 33/438 | 0.73% |

**The important × drifting quadrant is empty.** No feature in the top 20 by SHAP importance has PSI ≥ 0.10. The deployed features that do drift moderately (`id_31`, `v_missing_count`) rank 33 and 43 of 438 and each carry under 0.75% of importance — real but minor. The features the model leans on most (`C13`, `TransactionAmt`, `card1_prior_count`) are PSI-stable. So there is **no single named feature that is both a top driver and drifting** — the decline is diffuse, not attributable to one monitored signal.

### Would this monitoring have flagged the Phase 5 decline *before* test was scored?

**Yes — but only as a coarse warning, not a precise predictor.** Two things must be held together honestly:

- **The strong signal that WOULD have fired:** the overall val-vs-test adversarial AUC is 0.9999 — the input distribution is emphatically non-stationary. A label-free monitor computing this on incoming test-period features (no `isFraud`, as here) would have raised an unambiguous 'the world has shifted, do not assume stationary performance' flag *before* any labelled evaluation. That is real advance warning.
- **Why it is only coarse:** the loudest per-feature PSI alarms (`D1_normalized`, `P_emaildomain_prior_count`) are on features the model **doesn't use**; a careless operator could chase them fruitlessly. The features the model does rely on are mostly PSI-stable, and SHAP shows it leans 95% on transaction content — so per-feature PSI on the deployed set alone would have been mostly quiet and would NOT have cleanly predicted or attributed the specific −0.067 PR-AUC decline.

Net: monitoring gives a genuine, actionable *non-stationarity* alert, consistent with Phase 3's finding that separability is joint across many features — but it is a blunt early-warning instrument here, not a sharp forecaster of the exact drop.

*Read-only: no model, threshold, or manifest was modified in this phase.*
