# Summary Report — Customer Purchase Propensity: Data Cleaning & Feature Engineering

**Exam:** Red & White Skill Education — Practical (6 hours) | **Scope:** Preprocessing & feature engineering only (no model trained)

## Objective
Clean, merge, and engineer features from `customers.csv`, `transactions.json`, `products.sql`, and the
DummyJSON API so the result is ready for a *future* binary classifier predicting `purchased = 1/0`.

## Data Sources & Merge
Customers (CSV), transactions (JSON), and products (loaded into SQLite from `products.sql`) merged
cleanly on `customer_id` and `product_id` — 8 rows, 0 unmatched records, 0 missing values introduced.
The DummyJSON API (`id` 1–30, unrelated to real `customer_id` 101–108) was loaded and inspected but
**not merged**, since no valid join key exists — forcing one would fabricate a relationship.

## Techniques Used
Univariate/bivariate/multivariate EDA (14 charts + correlation heatmap); 6 missing-value techniques
(SimpleImputer, Most-Frequent, MissingIndicator+RandomSample, KNN, MICE, Complete Case Analysis) on a
clearly-labelled synthetic-missingness demo copy; 4 outlier methods (Z-score, IQR, Percentile,
Winsorization); datetime feature engineering (year/month/day/day-of-week/recency) with a fixed,
reproducible reference date (max transaction date = 2025-10-10); 3 encoding techniques (Label, One-Hot,
Ordinal — the latter on an explicitly-labelled demo column, since no genuine ordinal variable exists in
the raw data); equal-width + quantile income binning; 5 scaling techniques (Standard, MinMax, MaxAbs,
Robust, Normalizer) + `ColumnTransformer`; feature construction (recency, frequency/monetary aggregates,
age/income segments); `FunctionTransformer` (log1p, sqrt, reciprocal) and `PowerTransformer` (Box-Cox,
Yeo-Johnson); binarization (`frequent_buyer`, `high_value_purchase`).

## Problems Found in the Raw Data
- All 3 supplied files are genuinely clean: **zero missing values, zero duplicates**.
- Only **8 rows** survive the merge — every statistic here is real but illustrative of *method*, not
  a robust finding at scale.
- A real **₹200 price discrepancy** was found between transaction T007 and the product catalogue —
  captured as a `price_vs_catalogue_diff` feature rather than silently corrected.
- **Every supplied customer already has a transaction** — there is no negative class, so a genuine
  `purchased` target cannot be built from this data. A `high_value_purchase` (median-split) target is used
  instead purely to demonstrate binarization/bivariate mechanics, with its leakage implications documented.

## Best Imputation Method
**KNN Imputer**, conceptually — a very strong `age`↔`income` correlation (r≈0.98) was found in this
sample, which is exactly the multivariate structure KNN exploits (not applied to real data, which has no
missing values).

## Best Scaling Method
**RobustScaler for `amount`** (carries one legitimate high-value outlier, kept rather than deleted) and
**StandardScaler for `age`/`income`** (no outliers).

## Key Observations
Purchase amount is right-skewed (skew 0.81 → 0.03 after Box-Cox); `age`/`income` are very strongly
correlated in this sample — flagged as a small-sample artifact to re-check with more data; `amount` and
`price` are near-identical except for the one documented discrepancy.

## Conclusion
The pipeline produces a clean, reproducible, leakage-aware, ML-ready dataset (`processed_customer_data.csv`,
8 rows × 35 columns) with every preprocessing decision explained and justified against the actual data —
not invented. The main limitation is the small, purchase-only sample; the next step (out of scope for
this exam) is training and evaluating a classifier once a larger, longitudinal dataset with real
non-purchasing customers is available.
