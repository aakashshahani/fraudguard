<p align="center">
  <img src="reports/banner.svg" alt="FraudGuard: fraud detection on the IEEE-CIS dataset" width="100%">
</p>

# FraudGuard

![CI](https://github.com/aakashshahani/fraudguard/actions/workflows/ci.yml/badge.svg)

An end-to-end machine learning system that detects fraudulent card transactions on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset (590,540 transactions, 3.5% fraud). It runs the full lifecycle across seven phases: data preparation, feature engineering, adversarial validation, model selection, threshold tuning, explainability and drift monitoring, and a serving API with CI.

This uses the real IEEE-CIS competition data, not the toy `creditcard.csv`. It requires feature engineering across a transaction/identity join and careful handling of a strong temporal signal, which is where most of the interesting work is.

Two ideas run through the whole project. The first is leakage discipline: a fraud model that peeks at the future looks great offline and fails in production, so every phase is built to prevent that and is backed by a test that fails loudly if the discipline is ever broken. The second is honest reporting: the evaluation protocol was committed before any model was trained, the test set was scored exactly once, and results are written up as they are, including a hypothesis that turned out wrong.

## Results

| Metric | Value |
|--------|-------|
| Test PR-AUC (sealed hold-out, scored once) | **0.527** |
| Published paper XGBoost baseline, for comparison | 0.469 |
| Validation PR-AUC (used for model selection) | 0.594 |
| Operating point at the chosen threshold | precision 0.39, recall 0.66 |
| Features kept after adversarial validation | 438 |
| Automated guardrail tests, green in CI without the dataset | 24 |

PR-AUC (area under the precision-recall curve) is the right primary metric here because at a 3.5% fraud rate, ROC-AUC is inflated by the large legitimate class. The test score holds above a published paper baseline even after taking a real hit from temporal drift, which is discussed below.

## Architecture

```
IEEE-CIS raw CSVs (transaction joined with identity)
        |
  OFFLINE  (train and validation only, until the very last step)
        |
  P1   merge and downcast, then a strictly chronological 70/15/15 split
  P2   feature engineering: encoders fit on train only, missingness signals, sentinels
  P2.5 reconstruct a client identity (card + address + tenure), causal history features
  P3   adversarial validation drops drifting features, freezing feature_manifest.json
  P4   model bake-off (Logistic Regression, Random Forest, XGBoost) under a
       pre-registered protocol, producing the frozen winner model
  P5   cost-based threshold chosen on validation, then the test set is unsealed once
  P6   SHAP explainability and a validation-vs-test drift monitor
        |
        | frozen and versioned: model, threshold, feature manifest, encoders
        |
  ONLINE  (P7 serving)
        |
  POST /predict  ->  feature store lookup  ->  persisted encoders and sentinels
                 ->  manifest column order  ->  XGBoost + threshold
                 ->  { probability, decision }
  A skew test asserts the served feature vector equals the offline one, column by column.
  Packaged with Docker and a GitHub Actions CI pipeline that runs on committed fixtures.
```

## Tech stack

Python, pandas, XGBoost, scikit-learn, imbalanced-learn, SHAP, Weights and Biases, FastAPI, Docker, GitHub Actions, pyarrow, matplotlib.

## Project structure

```
fraudguard/
  data/{raw,processed}/          gitignored, except the small feature_manifest.json
  models/                        frozen artifacts (model, threshold, encoders), committed
  docs/                          per-phase results and the pre-registered protocol
  reports/figures/               plots (gitignored, regenerated)
  src/
    data_prep.py                 P1   merge, downcast, temporal split
    eda.py                       P1   class imbalance, missingness, time range
    feature_engineering.py       P2   encoders, missingness, sentinels
    uid_features.py              P2.5 client-identity reconstruction, causal aggregates
    adversarial_validation.py    P3   drift-based feature selection, feature manifest
    modeling.py                  P4   model bake-off with W&B tracking
    evaluation.py                P5   cost-based threshold, one-time test evaluation
    monitoring.py                P6   SHAP, drift monitoring (PSI, adversarial AUC)
    serving/                     P7   feature_store, assemble, FastAPI api
  scripts/                       build_serving_artifacts, benchmark_latency
  tests/                         guardrail tests for every phase, run in CI on fixtures
  Dockerfile, .github/workflows/ci.yml, LICENSE (MIT)
  requirements.txt, requirements-serving.txt
```

## Pipeline, phase by phase

### P1. Temporal split

Fraud is a time-ordered problem, so training on later data to predict earlier data leaks information that will not exist at inference time. The split is strictly chronological (70/15/15) and cuts on `TransactionDT` quantiles rather than row position, so a single timestamp cannot land on both sides of a boundary. A test asserts `max(train) < min(val) < min(test)` and there is a companion test proving that a shuffled split would fail that assertion, so the guardrail can actually catch a regression.

One honest detail carried through the whole project: `TransactionDT` is a time delta in seconds from a reference datetime that the competition deliberately does not disclose. The split is therefore reported in relative days, which is exact. Any calendar date shown in a figure is labeled as an assumed convention, not a real date.

### P2. Feature engineering

Categorical columns are frequency or label encoded, and the encoders are fit on the training split only, then applied to validation and test. A category that never appears in training maps to a designated unknown value (0 for frequency, -1 for label). Fitting an encoder on the full dataset would leak the held-out distribution into training, which is a subtle but real form of leakage that a test guards against.

Missingness is treated as signal rather than noise: an `has_identity` flag and a per-row count of null V-columns. V-column nulls are then filled with a `-999` sentinel so that tree models can split on "was missing" as its own value.

### P2.5. Client-identity reconstruction

The competition anonymizes the client, but `card1 + addr1 + D1_normalized` (card, billing region, and a time-stabilized tenure signal) reconstructs a persistent pseudo-identity. This is a documented technique from the top public solutions to this competition, not a guess, and per-client history is the strongest signal in the dataset.

The history features (prior transaction count, and the expanding mean and standard deviation of amount) are computed causally. Each row sees only rows for that client with a strictly earlier timestamp, never the current row or a later one. This is point-in-time correctness, the same guarantee production feature stores like Feast and Tecton provide with as-of joins, implemented here from first principles. When a client has fewer than two prior transactions the standard deviation is undefined, so it is set to the sentinel value and the prior-count feature records the history depth, which lets the model separate "genuinely low variance" from "not enough history to know."

There is a deliberate distinction between these two rules that looks contradictory but is not. Encoders are fit on train only, because an encoder learns a parameter from data and must not see held-out data while learning. Expanding aggregates are computed over the full time-ordered dataset, because each one is a causal lookup that reads only earlier rows, which is exactly the past-to-future direction that happens at inference time and leaks nothing. Fit parameters on train, window causally over everything.

### P3. Adversarial validation

A classifier is trained to tell training rows from validation rows. A feature that individually separates them past an AUC of 0.70 (a convention from top solutions to this competition) is drifting hard enough that a model could latch onto a training-only artifact, so it is dropped. The survivors are frozen into `feature_manifest.json`, which every later phase loads verbatim with no ad hoc selection.

Two honest points sit in the results. First, `TransactionDay` is a coarse copy of the time index and is excluded as a candidate up front, for the same reason `TransactionDT` is, rather than being "discovered" as drift. Second, only two features flag as high drift, but dropping them barely moves the combined adversarial AUC (from 1.0000 to 0.9995). That means the separability comes from many features acting together, not one or two bad actors, which is the known blind spot of a per-feature method. The correct reading is that adversarial validation removes the individually worst offenders, and the remaining separability is an expected property of using history features on a temporal split. "Only two flagged" is not a clean bill of health, and the writeup says so.

### P4. Model selection

The evaluation protocol was written and committed before any model was trained (`docs/phase4_evaluation_protocol.md`). It fixes the primary metric (validation PR-AUC), the comparison design (Logistic Regression, then Random Forest, then XGBoost, each averaged over three seeds), the imbalance-handling bake-off that runs only on the winning family, and the exact tie-break rules.

XGBoost won the family stage. In the bake-off, no imbalance handling beat the plain baseline on PR-AUC, because PR-AUC measures ranking quality across all thresholds while class weights and SMOTE mainly shift the operating point. Their benefit shows up at a chosen threshold, which is Phase 5's job. A pre-registered prediction, that class weights would match or beat SMOTE, was tested and reported as refuted (SMOTE out-ranked the class-weight arm, though no handling beat both). Reporting a refuted prediction as-is is the entire point of pre-registering it.

### P5. Threshold and final evaluation

The deployment threshold minimizes a pre-registered 10:1 false-negative to false-positive cost on validation, landing at about 0.078 with precision 0.39 and recall 0.66. A sensitivity analysis sweeps the cost ratio from 5:1 to 50:1 and shows the threshold falling and recall rising as missed fraud gets more expensive, which is the expected direction and does not change the decision.

The test set was then unsealed exactly once. Test PR-AUC came in at 0.527 against a validation PR-AUC of 0.594 for the same model, a decline of about 0.067. That drop is in the expected direction and is consistent with the strong temporal separability measured in Phase 3: the test period sits further in the future, so the features drift further and ranking degrades a little. A decline rather than an optimistic gain is what an honest, non-leaking evaluation should show. This result is final.

### P6. Explainability and drift monitoring

SHAP on the validation set shows the model draws about 95% of its importance from transaction content and under 5% from history features, which answers a question left open earlier: the model is not leaning on a "how much history exists" shortcut. A validation-vs-test drift monitor reports Population Stability Index and adversarial AUC using test feature values only, never the labels. A per-feature cross-check of drift against importance confirms that the two features which drift the most were already dropped in Phase 3, and that no feature the model actually relies on is drifting, so the important-and-drifting quadrant is empty. The monitor would flag the coming decline only coarsely, as an input-distribution shift, not as a sharp per-feature prediction, and the writeup says exactly that.

### P7. Serving

The FastAPI `/predict` service rebuilds the feature vector at request time using the persisted Phase 2 encoders, the same missingness and sentinel logic, and an in-memory feature store keyed by card and client. Unknown cards or clients fall back to the same insufficient-history sentinel rather than crashing or silently returning zero.

The load-bearing test feeds ten real transactions through the live pipeline and asserts that the assembled feature vector matches the offline `features.parquet` vector column for column across all 438 features. This is the direct catch for training-serving skew, where the online and offline feature computations quietly diverge. It runs in CI on committed fixtures with no dataset required.

## Running it

The dataset is not redistributable, so you download it from Kaggle (see "Getting the data"). Once it is in place:

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.data_prep             # P1   merge and temporal split
python -m src.eda                   # P1   EDA figures
python -m src.feature_engineering   # P2   features
python -m src.uid_features          # P2.5 client-identity aggregates
python -m src.adversarial_validation# P3   feature manifest
python -m src.modeling              # P4   model bake-off (W&B offline)
python -m src.evaluation            # P5   threshold and final test evaluation
python -m src.monitoring            # P6   SHAP and drift
pytest -q                           # all guardrail tests
```

### Serving API

```bash
# Standalone, using the committed feature-store snapshot
FRAUDGUARD_STORE=snapshot uvicorn src.serving.api:app --port 8000

# Or with Docker (lean image, no dataset required)
docker build -t fraudguard . && docker run -p 8000:8000 fraudguard

# Score a transaction
curl -s localhost:8000/predict -H 'content-type: application/json' \
  -d '{"card1": 10000, "TransactionAmt": 100.0, "TransactionDT": 15000000, "D1": 14, "addr1": 315, "ProductCD": "W"}'
# { "fraud_probability": ..., "decision": 0, "threshold": 0.0783, "uid_known": false }
```

### Getting the data

1. Create a free Kaggle account and accept the rules at [kaggle.com/c/ieee-fraud-detection](https://www.kaggle.com/c/ieee-fraud-detection) under the Rules tab. Downloads return a 403 until you accept.
2. Create an API token at [kaggle.com/settings](https://www.kaggle.com/settings) and place `kaggle.json` in `~/.kaggle/` (Windows: `C:\Users\<you>\.kaggle\`).
3. Download and unzip:
   ```bash
   pip install kaggle
   kaggle competitions download -c ieee-fraud-detection -p data/raw
   cd data/raw && unzip ieee-fraud-detection.zip
   ```

You need `train_transaction.csv` and `train_identity.csv`. Credentials and data are gitignored.

## Serving latency

Measured one request at a time, which is the number that matters for a checkout budget, p50 is 343 ms and p99 is 449 ms. That does not yet fit a typical 50 to 100 ms real-time budget. The breakdown shows where the cost is and that it is cheap to fix: the model and feature-store lookup take about 22 ms, while the pandas per-row assembly takes about 210 ms. Pandas has high fixed overhead on a single row because it is built for batch work, so replacing that assembly with plain dict and numpy operations would bring it to roughly 35 ms total. This is written up as a scoped optimization in `docs/phase7_results.md` rather than shipped as a misleading number. A high-concurrency in-process benchmark would only measure Python GIL contention, so throughput is scaled instead with more uvicorn workers and replicas.

## Scaling and next steps

The in-memory feature store loads the whole dataset at startup and holds the latest state per key. In production this becomes an online store (Redis) for fast lookups, an offline store for point-in-time-correct training joins, and a streaming path that updates aggregates as events arrive. The FastAPI app is stateless and scales horizontally behind a load balancer, with the feature store as the shared stateful tier.

On retraining cadence, this is not hypothetical. The measured 0.067 PR-AUC decline into a roughly one-month future test window, together with a validation-vs-test adversarial AUC near 1.0, argues for frequent retraining (weekly to monthly) plus trigger-based retraining when the drift monitor crosses an alert threshold, with a shadow-evaluation gate before any new model is promoted.

Planned next: a real-time feature store, a dead-letter path for malformed requests, and an automated retraining trigger wired to the Phase 6 drift monitor.

## Beyond credit-card fraud

The hard part of this project is not credit cards, it is the rare-class, adversarial, drifting shape of the problem, and that shape shows up wherever abuse and integrity teams work. Ad click fraud, account takeover, and fake-engagement detection all involve extreme class imbalance against an adversary who adapts, where labels arrive late and a random split silently leaks the future. The same machinery transfers: temporal splits, point-in-time feature aggregation, adversarial validation to drop drifting signal, cost-based thresholds, drift monitoring, and a serving path proven free of skew.

## License

[MIT](LICENSE)
