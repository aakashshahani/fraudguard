# Phase 7 — Serving, Testing, CI & Documentation

The final phase wraps the frozen model into a service. It is **read-only** against
Phase 5's model + threshold and Phase 2's persisted encoders — nothing is
retrained, re-thresholded, or re-engineered.

## Part A — In-memory feature store

`src/serving/feature_store.py` loads the engineered data once at startup and holds,
per `card1` and per `uid`, the **latest** expanding-aggregate state as of the
dataset's final timestamp — a stand-in for the live state a real feature store
(Feast/Tecton + Redis) would maintain. Unknown `card1`/`uid` (no prior history)
fall back to the **Phase 2.5 insufficient-history sentinel** (`uid_prior_count=0`,
mean/std `= -999`), never a crash or a silent zero. Phase 2's frequency/label maps
are persisted (`models/phase2_encoders.json`) and reloaded verbatim — never refit
at serving time.

## Part B — FastAPI `/predict`

Raw pre-engineering fields → feature-store lookup by `card1`/`uid` → persisted
encoders → Phase 2 missingness/`-999` sentinel logic → assemble in
`feature_manifest.json`'s exact column order → score with the frozen XGBoost model +
threshold. Response: `{fraud_probability, decision, threshold, uid_known}`.

## Part C — Training/serving skew guardrail (the critical test)

10 real test-split transactions are committed as a fixture (raw fields **and** their
precomputed offline feature vectors). `test_serving_assembly_matches_offline_features_no_skew`
feeds each transaction's raw fields through the live serving assembly and asserts the
result equals the offline `features.parquet` vector **column-for-column (all 438)**.
This is the direct catch for training/serving skew — a wrong encoder, a mis-filled
sentinel, or a column-order slip fails it loudly. It runs in CI on the committed
fixtures with **no Kaggle data**.

> Design note: the fixtures are chosen with **distinct `card1` and `uid`** so each
> key's snapshot holds that row's own point-in-time aggregate state. This models a
> *point-in-time-correct* feature retrieval (the same guarantee discussed in
> Phase 6): the test validates the assembly transform exactly; aggregates are a
> lookup, not recomputed at serving.

## Part D — Latency & the honest budget

Measured **sequentially** (one request at a time — the number that matters for a
checkout budget; a high-concurrency in-process figure would just measure GIL
contention, not deployment latency):

| Metric | Value |
|---|---|
| End-to-end p50 | **343 ms** |
| End-to-end p99 | **449 ms** |
| — of which **assemble** p50 | **210 ms** |
| — of which **model** p50 | **22 ms** |
| Single-worker throughput | ~2.8 rps |

**Read honestly: this does not currently fit a real-time checkout budget.** A
synchronous fraud check in a checkout flow typically has a **~50–100 ms p99** budget
(it's one hop among many). At 449 ms p99 we are ~5–9× over.

But the breakdown says exactly where the cost is and that it's cheaply fixable:

- **The model + store lookup are fast (~22 ms).** The ML serving path itself is
  within budget.
- **The bottleneck is the pandas-per-row assembly (~210 ms).** Pandas has high
  fixed per-operation overhead on a 1-row, 433-column frame; it's built for batch,
  not single inference. **Fix (high-confidence):** replace the per-row pandas
  assembly with plain dict + preallocated-numpy operations — the encoders are dict
  lookups, the sentinels are conditionals, and the manifest is a fixed index order.
  That removes essentially all 210 ms, targeting **<10 ms assemble → ~35 ms total**,
  comfortably inside budget. (Alternatives: micro-batch requests to amortise pandas
  overhead, or move the fraud check off the synchronous path with a conservative
  fallback.)
- **Throughput** is CPU-bound under the GIL, so a single process serves at the
  sequential latency; scale out with multiple uvicorn workers / replicas (the app is
  stateless; the feature store is the shared tier).

This is left as a documented, scoped optimisation rather than silently shipping a
misleading "it's fast" number — consistent with the project's honesty discipline.
Plot + raw numbers: `reports/figures/phase7_latency.{png,json}`.

## Part E — Docker + CI

- **`Dockerfile`** — single lean serving image (`requirements-serving.txt`; no
  shap/wandb/matplotlib), runs standalone from the committed snapshot store.
- **`.github/workflows/ci.yml`** — installs deps, lints (ruff, error + undefined-name
  rules), and runs the **full pytest suite**. Every test runs against committed
  fixtures / small artifacts; tests needing the full Kaggle dataset self-skip, so CI
  never requires Kaggle credentials.
- **`LICENSE`** — MIT.

## Leakage / read-only summary

The test split's `isFraud` label is never referenced in serving; the drift monitor
reads test *features* only; and a guardrail asserts serving/monitoring never modify
the frozen model, threshold, or manifest. The skew test closes the loop from offline
training to online serving.
