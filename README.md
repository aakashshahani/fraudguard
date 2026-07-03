<p align="center">
  <img src="reports/banner.svg" alt="FraudGuard — fraud detection on the IEEE-CIS dataset" width="100%">
</p>

# FraudGuard

![CI](https://github.com/aakashshahani/fraudguard/actions/workflows/ci.yml/badge.svg)

An **end-to-end fraud-detection system** built on the **IEEE-CIS Fraud Detection**
dataset (Kaggle `ieee-fraud-detection`) — the real one that requires feature
engineering across a transaction/identity join, not the toy `creditcard.csv`. It
runs the full lifecycle in **7 phases**: data → features → adversarial validation
→ pre-registered modeling → cost-based thresholding → explainability & drift →
serving, testing, CI.

**Headline result:** final **test PR-AUC 0.5266** at a cost-minimising operating
point (precision 0.32 / recall 0.63), holding above a published paper's XGBoost
baseline (0.4692) *after* an honest temporal-decay hit — and every step is guarded
against leakage by tests that fail loudly.

The through-line is **leakage discipline and honest claims**: a strictly temporal
split, causal (point-in-time) feature aggregation, encoders fit on train only, a
pre-registered evaluation protocol, a single sealed test evaluation, and a
serving path proven free of training/serving skew by a fixture test.

## Architecture

```
                         IEEE-CIS raw CSVs (transaction ⨝ identity)
                                        │
     ┌──────────────────────────────────┴───────────────────────────────────┐
     │ OFFLINE  (train / validation only until the very end)                  │
     │                                                                        │
     │  P1  merge + downcast ──► strictly TEMPORAL split (70/15/15) ──┐       │
     │                                                                ▼       │
     │  P2  feature engineering: freq/label encoders (fit on TRAIN),          │
     │      missingness signals, −999 sentinels                               │
     │  P2.5 UID = card1+addr1+D1n ──► CAUSAL expanding aggregates            │
     │                                                                │       │
     │  P3  adversarial validation ──► drop drifting feats ──► feature_manifest
     │                                                                │       │
     │  P4  families (LogReg/RF/XGB) ──► imbalance bake-off ──► winner.joblib  │
     │      (pre-registered protocol; PR-AUC decides; SMOTE hypothesis tested) │
     │                                                                │       │
     │  P5  cost-min threshold on VAL ──► seal broken ONCE ──► TEST PR-AUC     │
     │  P6  SHAP (95% content) + val-vs-test drift monitor                     │
     └────────────────────────────────────┬───────────────────────────────────┘
                                           │ frozen: model + threshold + manifest + encoders
     ┌─────────────────────────────────────▼──────────────────────────────────┐
     │ ONLINE  (P7 serving)                                                     │
     │  POST /predict  →  feature store lookup (card1/uid latest state)         │
     │                 →  persisted encoders + sentinels + manifest order       │
     │                 →  XGBoost + threshold  →  {probability, decision}       │
     │  skew guardrail: assembled vector == offline features, column-for-column │
     │  Docker · GitHub Actions CI (runs on committed fixtures, no Kaggle data) │
     └──────────────────────────────────────────────────────────────────────── ┘
```

---

## Project structure

```
fraudguard/
├── data/{raw,processed}/          # gitignored (except the small feature_manifest.json)
├── models/                        # frozen artifacts (winner, threshold, encoders) — committed
├── docs/                          # per-phase results + the pre-registered protocol
├── reports/figures/               # plots (gitignored, regenerated)
├── src/
│   ├── data_prep.py               # P1  merge → downcast → temporal split
│   ├── eda.py                     # P1  class imbalance, missingness, TransactionDT range
│   ├── feature_engineering.py     # P2  encoders (fit on train), missingness, sentinels
│   ├── uid_features.py            # P2.5 UID reconstruction + causal expanding aggregates
│   ├── adversarial_validation.py  # P3  drift-drop features → feature_manifest.json
│   ├── modeling.py                # P4  family progression + imbalance bake-off (W&B)
│   ├── evaluation.py              # P5  cost-min threshold + one-time test evaluation
│   ├── monitoring.py              # P6  SHAP + val-vs-test drift (PSI, adversarial AUC)
│   └── serving/                   # P7  feature_store · assemble · FastAPI api
├── scripts/                       # build_serving_artifacts · benchmark_latency
├── tests/                         # all phases' guardrails (run in CI on fixtures)
├── Dockerfile · .github/workflows/ci.yml · LICENSE (MIT)
└── requirements.txt · requirements-serving.txt
```

`data/raw` and `data/processed` are **gitignored** — the dataset is hundreds of
MB and must never be committed. `.gitkeep` files preserve the empty folders.

---

## Data acquisition (manual step — required before anything runs)

The IEEE-CIS data is **not** redistributable, so it is not in this repo. Each
person must download it themselves from Kaggle:

1. **Create a free Kaggle account** at <https://www.kaggle.com>.
2. **Accept the competition rules.** Go to
   <https://www.kaggle.com/c/ieee-fraud-detection>, open the **Rules** tab, and
   click *"I Understand and Accept"*. The download **will fail with a 403 until
   you have accepted the rules for this specific competition** — an account
   alone is not enough.
3. **Create a Kaggle API token.** On your Kaggle account page
   (<https://www.kaggle.com/settings>) → *API* → **Create New Token**. This
   downloads `kaggle.json` (your username + key).
4. **Place the token where the CLI expects it:**
   - macOS / Linux: `~/.kaggle/kaggle.json` — then `chmod 600 ~/.kaggle/kaggle.json`
   - Windows: `C:\Users\<you>\.kaggle\kaggle.json`
5. **Install the CLI and download into `data/raw/`:**

   ```bash
   pip install kaggle
   kaggle competitions download -c ieee-fraud-detection -p data/raw
   cd data/raw && unzip ieee-fraud-detection.zip
   ```

   You need at least `train_transaction.csv` and `train_identity.csv` for
   Phase 1. (`kaggle.json` and everything under `data/raw/` are gitignored, so
   your credentials and the data will never be committed.)

   > **Why we don't use `test_transaction.csv` / `test_identity.csv`:** Kaggle's
   > official test set has **withheld labels** — it exists only for leaderboard
   > scoring, so no real metrics can be computed against it locally. All
   > evaluation in this project is reported on our own held-out **temporal test
   > split** carved from the labeled training data (the latest ~15%), which is
   > the correct choice, not a shortcut.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
# Phase 1 — merge + downcast + temporal split  (writes to data/processed/)
python -m src.data_prep

# Phase 1 — EDA figures  (writes to reports/figures/)
python -m src.eda

# Phase 2 — feature engineering  (reads Phase 1 artifacts, writes features.parquet)
python -m src.feature_engineering

# Phase 2.5 — UID pseudo-identity aggregation  (extends features.parquet in place)
python -m src.uid_features

# Phase 3 — adversarial validation + frozen feature manifest  (train+val only)
python -m src.adversarial_validation

# Phase 4 — model family progression + imbalance bake-off  (train+val only; W&B offline)
python -m src.modeling

# Phase 5 — threshold selection (val) + final test evaluation (loads Phase 4 winner)
python -m src.evaluation

# Phase 6 — SHAP explainability + val-vs-test drift monitoring (read-only)
python -m src.monitoring

# Guardrail tests (split + FE + UID + adversarial + modeling + sweep + drift)
pytest -q
```

### Outputs (in `data/processed/`)

| File | Contents |
|------|----------|
| `train_merged.parquet` | Transaction ⨝ identity, dtype-downcast |
| `split_indices.parquet` | `TransactionID`, `TransactionDT`, `split` (`train`/`val`/`test`) |
| `split_boundaries.json` | Auditable cut points in raw seconds + relative days (assumed calendar dates labelled as such) |
| `features.parquet` | Model-ready feature set (encoded categoricals, temporal-safe aggregates, missingness signals, and Phase 2.5 UID pseudo-identity aggregates), with the Phase 1 `split` preserved |
| `adversarial_validation_report.json` | Per-feature adversarial AUC, flags, and overall/residual separability (Phase 3) |
| `feature_manifest.json` | Frozen list of allowed modeling columns Phase 4 must use verbatim (Phase 3) |

The pre-registered Phase 4 evaluation protocol lives in
`docs/phase4_evaluation_protocol.md`.

`src/data_prep.py` also **logs the split boundaries** at runtime so the split is
auditable without opening any file.

---

## Why these choices (design notes)

- **Left join, nulls kept.** Identity data covers only a subset of transactions.
  Those nulls are a real signal (identity present vs. absent), so rows are never
  dropped at the join.
- **Chronological split, never shuffled.** Fraud is a time-ordered problem;
  training on the future to predict the past leaks. The split cuts on
  `TransactionDT` **quantiles** (not row position) so a single timestamp can
  never straddle a boundary. `tests/test_temporal_split.py` asserts
  `max(train) < min(val) < ... < min(test)` and is designed to **fail loudly**
  if the logic is ever replaced with a random split.
- **Validation reserved for tuning + drift.** The middle 15% is for threshold
  tuning and the later drift simulation; the final 15% (test) stays untouched
  until final evaluation.
- **Dtype downcasting.** float64→float32 and int64→int32 (where the range fits,
  via `pd.to_numeric(downcast=...)`) roughly halves the memory footprint.
- **Expanding counts, not global counts (Phase 2) — necessary, not optional.**
  Aggregate features like "how many times has this card been seen?" are only
  legitimate if computed *causally*: row N may count prior rows but never later
  ones. A naive `groupby.transform("count")` over the whole dataset leaks the
  future into every past row — the model would train on a count that, at
  inference time, doesn't yet exist. `expanding_prior_count` uses
  `rank(method="min") - 1` on `TransactionDT` within each key, giving the number
  of **strictly earlier** occurrences (ties don't count each other), independent
  of row order. A handcrafted test asserts the count matches hand-calculation
  and is unchanged under shuffling.
- **Encoders fit on train only (Phase 2) — necessary, not optional.**
  Frequency/label encoders are fit *strictly on the train split* and then applied
  to val/test; a category unseen in train maps to a designated unknown value
  (frequency → `0`, label → `-1`). Fitting the encoder on the full dataset would
  bleed val/test distribution into training — the frequency of a category would
  reflect rows the model shouldn't have seen, and threshold tuning on val would
  be optimistic. A test asserts a val/test-only category never appears in the
  fitted train map (confirmed on real data: e.g. new `id_30`/`id_31` device
  strings correctly encode to `-1`).
- **UID aggregation before modeling, not after (Phase 2.5) — a sequencing
  decision.** `card1 + addr1 + D1_normalized` reconstructs a persistent
  pseudo-client identity (a documented top-solution technique for this dataset),
  and per-client history is the strongest signal in IEEE-CIS. It is built *now*,
  as part of feature engineering, specifically so Phase 3's imbalance-handling
  comparison (class weights vs. resampling vs. …) runs against **one complete,
  frozen feature set**. If the UID features were added after a baseline existed,
  any lift would be confounded with the modeling change under test — you could no
  longer attribute an improvement to the technique rather than the feature set
  shifting underneath it.
  *Known bounded limitation:* `D1` is null for ~0.2% of rows, so those rows share
  a `__missing__` component in the key and a few distinct clients can blend into
  one pseudo-identity. The failure mode is *dilution* (a softer aggregate), not
  leakage or corruption, and it is bounded to that thin slice — documented and
  left as-is rather than rebuilt.
- **UID aggregates are full-dataset, categorical encoders are train-only — and
  that is not a contradiction.** These look opposite but answer different
  questions. An *encoder learns a parameter* (a category's frequency/code) from
  data; if it sees val/test while fitting, held-out distribution bleeds into the
  model — so it is fit on train only. An *expanding aggregate is a causal lookup*
  that reads only strictly-earlier rows; a validation row summarising earlier
  training rows is exactly the past→future direction that occurs at inference
  time, so it leaks nothing regardless of where the split boundary falls.
  Therefore `uid_prior_count` / `uid_amt_expanding_mean` / `uid_amt_expanding_std`
  are computed over the full time-ordered dataset, each excluding the current
  row (and tied timestamps). Rule of thumb: **fit parameters on train; window
  causally over everything.** The std of a client's prior amounts is undefined
  with < 2 prior transactions, so it is set to the `-999` sentinel (with
  `uid_prior_count` flagging history depth) rather than a misleading `0` — the
  model can then separate "genuinely low variance" from "not enough history".
- **Point-in-time correctness (the systems framing).** The causal-window
  discipline enforced by hand throughout this project — every aggregate reads
  only strictly-earlier rows, and the split cuts strictly on time — *is*
  **point-in-time correctness**: the guarantee that a feature's value for a row
  reflects only what was knowable at that row's timestamp. This is exactly what
  production feature stores (Feast, Tecton) exist to enforce at scale via
  point-in-time (as-of) joins. Building it explicitly here means the offline
  training features would match an online serving path without train/serve skew —
  the same property those systems provide, implemented from first principles.
- **Adversarial validation drops individually-drifting features, then freezes the
  set (Phase 3).** A classifier is trained to tell train rows from val rows (label
  0/1); a feature that *individually* separates them (univariate AUC > 0.70, a
  top-solution convention for this competition) is drifting hard enough that a
  fraud model could latch onto a train-only artifact. Those are dropped and the
  survivors frozen into `feature_manifest.json`, which Phase 4 loads verbatim — no
  ad hoc selection later. Two honest points: (1) `TransactionDay` (a Phase 2
  feature = `TransactionDT // 86400`) is a coarse copy of the temporal index, so
  it is excluded as a *candidate* by the same rule the spec applies to
  `TransactionDT`, not "discovered" as drift — otherwise it trivially drives the
  overall AUC to 1.0. (2) **The honest, precise reading of the residual:** only two
  features flag (`P_emaildomain_prior_count` 0.76, `D1_normalized` 0.72), yet
  dropping them barely moves the combined AUC (1.0000 → 0.9995). That means the
  separability comes from **many features acting jointly**, not one or two bad
  actors — which is exactly the blind spot of a per-feature *univariate* method
  (the tradeoff we chose over full-model importance). So the correct statement is:
  *adversarial validation caught the individually-strongest drifting features; the
  residual separability is an **expected** property of using inherently
  time-cumulative features (uid/card prior counts) by design — a live model sees
  the same daily growth — and is **not** evidence the feature set generalizes
  cleanly.* "Only 2 flagged" is **not** a clean bill of health. We deliberately do
  not try to erase this (the growth is the feature's whole point); whether the
  model leans on "how much history exists" as a shortcut vs. genuinely
  fraud-indicative patterns is deferred to SHAP analysis in a later phase. The
  central UID aggregates and raw `D1` are **not** flagged and survive.
- **Pre-registered evaluation protocol (`docs/phase4_evaluation_protocol.md`).**
  Written and committed in Phase 3, before any Phase 4 model exists: primary
  metric (validation PR-AUC), the family-progression-then-imbalance-bake-off
  design, and the exact winner + tie-break rule are fixed *before* results, so the
  comparison can't quietly become "whichever number looked best afterward." Same
  discipline as the split-boundary guardrail, applied to modeling.
- **Phase 4 outcome — imbalance handling didn't help *ranking*, and that's not a
  contradiction.** XGBoost won the family stage (val PR-AUC 0.5446 vs RF 0.5406 vs
  logreg 0.3623). In the bake-off, **no-handling won (0.5897)**, ahead of SMOTE
  (0.5752), class weights (0.5446), and both (0.5335). PR-AUC is a
  *threshold-independent ranking* metric; class weights and SMOTE reshape the
  score distribution to trade precision for recall, which doesn't improve ranking
  — their benefit lands at a chosen operating point, and choosing that point is
  deliberately **Phase 5**. The pre-registered SMOTE prediction (class weights ≥
  SMOTE) was **refuted** — SMOTE out-ranked the class-weight arm — but SMOTE did
  not "win": no-handling beat it, and stacking weights+SMOTE was worst. Reporting
  a refuted pre-registration honestly is the point of pre-registering. Full tables
  in `docs/phase4_results.md`; the winning `xgboost + none` (seed 42) model is
  frozen in `models/phase4_winner.joblib` for Phase 5/6 to reuse, not retrain.
- **Phase 5 — cost-based threshold, then the one-time test unseal.** The deployed
  model's operating threshold is chosen by **minimising the pre-registered 10:1
  cost on validation** (t≈0.078: precision 0.39, recall 0.66); a sensitivity sweep
  (5:1→50:1) shows the threshold falling and recall rising as missed fraud gets
  costlier — the expected direction, decision unchanged. The **Stage 1
  RF-vs-XGBoost divergence narrowed** (not fully closed): it existed only for the
  *balanced* XGBoost arm, and the deployed `none` model is cheaper than the RF
  *configuration tested* (RF+balanced) at each model's own best point (0.1517 vs
  0.1647) — a symmetric RF+none check wasn't run, so a reversal stays technically
  open, though XGBoost's decisive Stage-1 PR-AUC margin makes it unlikely. Then the
  test split was unsealed **once**: **test PR-AUC 0.5266 vs this artifact's seed-42
  validation 0.5937 (−0.067; the Phase 4 3-seed mean was 0.5897)** — a drop in the
  expected direction, consistent with Phase 3's ~0.9995 temporal separability
  (direction/qualitative match, not a derived magnitude). That the gap is a
  *decline* rather than an optimistic gain is consistent with — though not proof of
  — a leak-free evaluation. Full breakdown in `docs/phase5_results.md`.
- **Phase 6 — SHAP answers the history-shortcut question; drift monitoring is a
  coarse warning, not a sharp predictor.** SHAP on validation shows the model is
  **95.1% driven by transaction content, only 4.9% by history/count features** —
  directly answering the question left open since Phase 3: it is **not** leaning on
  "how much history exists" as a shortcut (`card1_prior_count` is the one prominent
  history feature; the rest are minor). Drift monitoring (val-vs-test adversarial
  AUC 0.9999, ~unchanged from train-vs-val) confirms strong, persistent
  non-stationarity. But the honest read on "would monitoring have caught the Phase 5
  decline early?" is **yes, coarsely**: the two *significant*-PSI features
  (`D1_normalized`, `P_emaildomain_prior_count`) were both already **dropped in
  Phase 3** (a neat validation that Phase 3 removed the right features) and aren't in
  the deployed model. A per-feature **drift × importance cross-check** makes the
  point sharp rather than category-average: the two deployed features that *do* drift
  moderately (`id_31`, `v_missing_count`) rank 33rd/43rd of 438 by SHAP and each carry
  <0.75% of importance, while the top drivers (`C13`, `TransactionAmt`,
  `card1_prior_count`) are PSI-stable — so the **"important × drifting" quadrant is
  empty**. The overall-separability signal gives a real "the world shifted" alert, but
  no single monitored feature would have sharply predicted the −0.067 drop; the decline
  is diffuse. Full breakdown in `docs/phase6_results.md`.
- **Phase 7 — serving with a skew guardrail, not just an endpoint.** The FastAPI
  `/predict` service reconstructs the feature vector at request time by reusing
  Phase 2's *persisted* encoders, the same missingness/sentinel logic, and an
  in-memory feature store (latest per-`card1`/`uid` aggregate state — a stand-in
  for what Feast/Tecton+Redis maintain live). The load-bearing piece is the
  **training/serving skew test**: 10 real transactions' raw fields go through the
  live assembly and the result is asserted equal, column-for-column, to their
  offline `features.parquet` vector. Unknown cards/uids fall back to the Phase 2.5
  insufficient-history sentinel rather than crashing or silently zeroing. Full
  breakdown in `docs/phase7_results.md`.

---

## Roadmap

1. **Phase 1 — Data acquisition, merge & temporal split** ✅
2. **Phase 2 — Feature engineering & encoding** ✅
   *(incl. Phase 2.5 — UID pseudo-identity aggregation)* ✅
3. **Phase 3 — Adversarial validation & pre-registered Phase 4 protocol** ✅
4. **Phase 4 — Baseline modeling & imbalance bake-off** ✅
   *(results in `docs/phase4_results.md`, winner in `models/`)*
5. **Phase 5 — Threshold selection & final evaluation** ✅
   *(results in `docs/phase5_results.md`; test split evaluated once)*
6. **Phase 6 — Explainability (SHAP) & drift monitoring** ✅
   *(results in `docs/phase6_results.md`)*
7. **Phase 7 — Serving, testing, CI & documentation** ✅ *(complete)*
   *(results in `docs/phase7_results.md`)*

> **TransactionDT note (important):** `TransactionDT` is a time delta in
> **seconds from a reference datetime that IEEE-CIS deliberately does not
> disclose** — specifically so competitors can't join external calendar data
> (holidays, weekends) against it. This project therefore reports the split in
> **relative days** (day 0 = the reference), which is exact and assumption-free.
> Any calendar date shown (e.g. in a figure caption) is an **assumed convention**
> — a common community guess of ~Dec 2017 for day 0 — and is always labelled
> "assumed". It is **not** an official date; treat the ~6-month span as *≈183
> relative days*, not a verified calendar range.

---

## Serving (Phase 7)

```bash
# Run the API locally (standalone, using the committed feature-store snapshot)
FRAUDGUARD_STORE=snapshot uvicorn src.serving.api:app --port 8000

# Or containerised (lean image; no Kaggle data required)
docker build -t fraudguard . && docker run -p 8000:8000 fraudguard

# Score a transaction
curl -s localhost:8000/predict -H 'content-type: application/json' \
  -d '{"card1": 10000, "TransactionAmt": 100.0, "TransactionDT": 15000000, "D1": 14, "addr1": 315, "ProductCD": "W"}'
# -> {"fraud_probability": ..., "decision": 0, "threshold": 0.0783, "uid_known": false}
```

`/predict` reconstructs the model's feature vector at request time from the raw
fields — feature-store lookup → **persisted** Phase 2 encoders → missingness /
`-999` sentinels → manifest column order → XGBoost + the Phase 5 threshold. The
CI-enforced **skew test** guarantees this assembly matches the offline features
column-for-column, and unknown cards/uids degrade to the insufficient-history
sentinel rather than crashing.

**Latency (measured, honest):** single-request **p50 343ms / p99 449ms** — which
does *not* yet fit a ~50–100ms real-time checkout budget. The breakdown shows why
and that it's cheaply fixable: the model + store lookup are ~22ms; the **pandas
per-row assembly is ~210ms** (pandas is built for batch, not single inference).
Replacing it with dict/numpy assembly targets <10ms (~35ms total). Documented as a
scoped optimisation rather than shipping a misleading number — see
`docs/phase7_results.md` and `scripts/benchmark_latency.py`.

## Scaling considerations

- **In-memory store → Feast/Tecton + Redis.** The `FeatureStore` here loads the
  whole dataset at startup and holds latest per-key state in a dict. At production
  scale that becomes: an **online store** (Redis) for millisecond key lookups, an
  **offline store** for point-in-time-correct training joins, and a **streaming**
  path that updates each key's aggregates as events arrive — so the "latest state"
  we snapshot statically is instead maintained live. The FastAPI app is stateless
  and horizontally scalable behind a load balancer; the feature store is the shared
  stateful tier. The `assemble.py` logic is exactly the transform a feature-store
  "on-demand transform" would host.
- **Retraining cadence, grounded in the measured drift.** This isn't hypothetical:
  Phase 5 measured a **−0.067 PR-AUC decline** from validation into a ~1-month
  future test window, and Phase 6 measured a **val-vs-test adversarial AUC of
  0.9999** — the input distribution shifts substantially month-over-month. That
  argues for **frequent retraining** (weekly–monthly) rather than set-and-forget,
  plus **trigger-based** retraining when the Phase 6 drift monitor's overall
  adversarial AUC / PSI crosses an alert threshold, with a shadow-evaluation gate
  before any new model is promoted.

## Future work

- **Real-time feature store** (Feast/Tecton + Redis) with streaming aggregate
  updates, replacing the static in-memory snapshot.
- **Dead-letter path** for malformed / incomplete requests: schema-validate at the
  edge, route rejects to a DLQ for inspection and replay instead of failing open.
- **Automated retraining trigger** wired to the Phase 6 drift monitor — when
  overall adversarial AUC or per-feature PSI crosses threshold, kick off a
  retrain + shadow-eval pipeline, closing the monitor→retrain loop.

## Beyond credit-card fraud

The hard part of this project isn't credit cards — it's the **rare-class,
adversarial, temporally-drifting** shape of the problem, and that shape is
everywhere abuse/integrity teams operate. **Ad click fraud**, **account takeover**,
and **fake engagement / bot detection** are all extreme class imbalance against an
adversary who actively adapts (so yesterday's features drift), where labels arrive
late and a fixed random split silently leaks the future. The machinery built here
transfers directly: strictly temporal splits, point-in-time-correct feature
aggregation, adversarial validation to drop drifting signal, cost-asymmetric
thresholding, drift monitoring, and a serving path proven free of training/serving
skew. Swap the dataset, keep the discipline.

## License

[MIT](LICENSE)
