<p align="center">
  <img src="reports/banner.svg" alt="FraudGuard — fraud detection on the IEEE-CIS dataset" width="100%">
</p>

# FraudGuard

A fraud-detection project built on the **IEEE-CIS Fraud Detection** dataset
(Kaggle competition `ieee-fraud-detection`) — the real one that requires
feature engineering across a transaction/identity join, not the toy
`creditcard.csv`.

The project is delivered in **6 phases**. This repository currently implements:

> ### Phase 1 — Data acquisition, merge & temporal split ✅
> Load the raw CSVs, left-join identity onto transactions, downcast dtypes to
> fit a laptop, persist to Parquet, and build a **strictly chronological**
> train/validation/test split with a pytest guardrail against leakage.
>
> No encoding, no feature engineering, no models yet — those are later phases.

---

## Project structure

```
fraudguard/
├── data/
│   ├── raw/            # raw Kaggle CSVs  (gitignored — large)
│   └── processed/      # merged parquet + split artifacts (gitignored)
├── notebooks/          # exploratory notebooks
├── reports/figures/    # saved EDA plots (gitignored, regenerated)
├── src/
│   ├── data_prep.py    # load → merge → downcast → parquet → temporal split
│   └── eda.py          # class imbalance, missingness, TransactionDT range
├── tests/
│   └── test_temporal_split.py   # asserts the split is leak-free
├── requirements.txt
└── README.md
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

# Guardrail tests (temporal split + feature-engineering + UID leakage)
pytest -q
```

### Outputs (in `data/processed/`)

| File | Contents |
|------|----------|
| `train_merged.parquet` | Transaction ⨝ identity, dtype-downcast |
| `split_indices.parquet` | `TransactionID`, `TransactionDT`, `split` (`train`/`val`/`test`) |
| `split_boundaries.json` | Auditable cut points in raw seconds + relative days (assumed calendar dates labelled as such) |
| `features.parquet` | Model-ready feature set (encoded categoricals, temporal-safe aggregates, missingness signals, and Phase 2.5 UID pseudo-identity aggregates), with the Phase 1 `split` preserved |

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

---

## Roadmap

1. **Phase 1 — Data acquisition, merge & temporal split** ✅
2. **Phase 2 — Feature engineering & encoding** ✅
   *(incl. Phase 2.5 — UID pseudo-identity aggregation)* ✅ *(current)*
3. Phase 3 — Baseline modeling
4. Phase 4 — Model tuning & evaluation
5. Phase 5 — Drift simulation & monitoring
6. Phase 6 — Serving / deployment

> **TransactionDT note (important):** `TransactionDT` is a time delta in
> **seconds from a reference datetime that IEEE-CIS deliberately does not
> disclose** — specifically so competitors can't join external calendar data
> (holidays, weekends) against it. This project therefore reports the split in
> **relative days** (day 0 = the reference), which is exact and assumption-free.
> Any calendar date shown (e.g. in a figure caption) is an **assumed convention**
> — a common community guess of ~Dec 2017 for day 0 — and is always labelled
> "assumed". It is **not** an official date; treat the ~6-month span as *≈183
> relative days*, not a verified calendar range.
```
