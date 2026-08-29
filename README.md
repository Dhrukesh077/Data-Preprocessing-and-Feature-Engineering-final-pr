# 🧹 Customer Purchase Propensity
### Data Cleaning and Feature Engineering Pipeline

*A practical data-preprocessing project preparing raw multi-source e-commerce data for a future purchase-prediction model.*

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Pandas](https://img.shields.io/badge/Pandas-3.x-150458)
![NumPy](https://img.shields.io/badge/NumPy-2.x-013243)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-11557c)
![Seaborn](https://img.shields.io/badge/Seaborn-0.13-4c72b0)
![SciPy](https://img.shields.io/badge/SciPy-1.x-8CAAE6)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E)
![SQLite](https://img.shields.io/badge/SQLite-in--memory-003B57)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 1. Overview

This project cleans, merges, and engineers features from four raw data sources belonging to a small
e-commerce business, so the result is ready for a **future** machine learning model that predicts whether
a customer will make a purchase. **No model is trained here** — the entire scope is data understanding,
cleaning, EDA, and feature engineering.

## 🎯 2. Project Overview

An e-commerce company keeps customer, transaction, and product data in different systems (a CSV export, a
JSON transaction log, a SQL products table) and also has access to a general-purpose user API. Before any
predictive model can be built, this scattered, inconsistent data has to be understood, validated, merged,
and turned into clean, numeric, ML-ready features — that pipeline is what this project delivers.

## 💼 3. Business Problem

Predicting which customers are likely to purchase lets the business target offers more effectively. This
project builds the **data foundation** for that prediction — proper preprocessing is what determines
whether any model built on top of it can be trusted.

## 🤖 4. Machine Learning Problem

Framed as **binary classification**, target `purchased = 1` (customer purchased) / `0` (did not).
**This project does not train a model.** It also documents an important, genuine limitation: the supplied
data contains **no customers who failed to purchase**, so a true `purchased` label cannot be constructed
from it (see Section 19). A `high_value_purchase` demonstration target is used instead to show the
mechanics of binarization and bivariate analysis — see the notebook, Section 14.3, for the full reasoning.

## 📂 5. Data Sources

| Source | Format | Purpose | Loading Method |
|---|---|---|---|
| `customers.csv` | CSV | Customer demographics (age, gender, city, income) | `pandas.read_csv` |
| `transactions.json` | JSON | Purchase records (amount, payment mode, date) | `json` + `pandas.DataFrame` |
| `products.sql` | SQL script | Product catalogue (price, category, stock) | `sqlite3` in-memory DB → `pandas.read_sql` |
| DummyJSON API (`/users`) | REST/JSON | General user profiles (inspected, **not merged** — see §9) | `requests` → `pandas.json_normalize` |

## 🗂️ 6. Dataset Structure

**customers.csv** (8 rows)

| Column | Type | Description |
|---|---|---|
| customer_id | int | Unique customer identifier |
| name | text | Customer name (excluded from final ML features — PII) |
| age | int | Customer age |
| gender | text | Male / Female |
| city | text | Customer's city |
| income | int | Annual income (INR) |

**transactions.json** (8 rows): `transaction_id`, `customer_id`, `product_id`, `amount`, `payment_mode`, `date`

**products** (6 rows, via SQLite): `product_id`, `product_name`, `category`, `price`, `stock`

## 🔄 7. Project Workflow

```
Raw Data
  ↓
Data Import (CSV / JSON / SQLite / API)
  ↓
Data Understanding
  ↓
Data Integration (merge + validate)
  ↓
EDA (univariate / bivariate / multivariate)
  ↓
Missing Value Handling (demo)
  ↓
Outlier Handling
  ↓
Date & Mixed-Variable Engineering
  ↓
Categorical Encoding
  ↓
Binning
  ↓
Feature Scaling
  ↓
Feature Transformation
  ↓
Feature Construction & Binarization
  ↓
Final ML-Ready Dataset
```

## 📥 8. Data Import

Each of the four sources is loaded independently and profiled (`shape`, `dtypes`, `head`/`tail`,
`describe`, missing values, duplicates) before anything else happens — see notebook Sections 2–3.
`products.sql` is executed against a temporary **in-memory SQLite database**; the DummyJSON API is called
live via `requests`, with a cached snapshot (`data/api_users_snapshot.json`) as an offline fallback so the
notebook stays reproducible.

## 🔗 9. Data Merging

- **Join keys:** `customer_id` (customers ↔ transactions) and `product_id` (transactions ↔ products) — both
  validated with `indicator=True` merges: **0 unmatched rows** on either join.
- **API integration decision:** the DummyJSON API's `id` values (1–30) have **no relationship** to the real
  `customer_id` values (101–108). Forcing a merge would fabricate a relationship between a real customer
  and a random fictional profile, so the API is loaded and inspected on its own but **excluded** from the
  merged dataset — exactly as the exam's own instructions require.

## 🔍 10. Exploratory Data Analysis

### Univariate
![Income Distribution](outputs/charts/01_income_distribution.png)
![Age Distribution](outputs/charts/02_age_distribution.png)
![Purchase Amount Distribution](outputs/charts/03_purchase_amount_distribution.png)

Income and age are fairly compact and roughly symmetric; **purchase amount is right-skewed** (skew ≈ 0.81),
driven by two higher-priced electronics purchases — flagged for transformation later (§17).

![Categorical Distributions](outputs/charts/04_categorical_distributions.png)

Every categorical column is near-unique at this sample size (8 distinct cities, near-even gender split) —
a signal that One-Hot Encoding for high-cardinality columns like `city` needs care at this scale (§14).

### Bivariate
![Income vs Purchase Value](outputs/charts/05_income_vs_purchase_value.png)
![Age vs Amount](outputs/charts/06_age_vs_amount.png)
![Amount by Category and Payment](outputs/charts/07_amount_by_category_and_payment.png)

Customers in the above-median purchase group show a higher average income in this sample; `Audio` and
`Wearable` categories carry the highest average transaction value.

### Multivariate
![Correlation Heatmap](outputs/charts/heatmap_correlation.png)

Only genuinely numeric columns are correlated — categorical text is **never** blindly label-encoded just
to force it into a correlation matrix. Two real findings: `amount` and `price` are near-perfectly
correlated (expected — amount *is* the price paid), and `age`/`income` show a very strong r≈0.98
correlation in this sample, flagged as a likely small-sample artifact worth re-testing with more data.

![Pairplot](outputs/charts/08_pairplot_numeric.png)

## 🧩 11. Missing Data Handling

The real merged dataset has **zero missing values**. Per the exam's data-integrity requirement, missing
values are **not fabricated into the production data** — instead, all six required techniques are
demonstrated on a clearly-labelled, controlled `demo_df` copy (notebook §6), which is discarded afterward.

| Method | What it does | Advantages | Disadvantages | Used for final pipeline? |
|---|---|---|---|---|
| SimpleImputer (mean) | Fills with column mean | Fast, simple baseline | Ignores relationships between columns | No |
| Most Frequent | Fills with column mode | Works for categorical data | Poor for high-cardinality columns | No |
| MissingIndicator + Random Sample | Flags + samples from observed values | Preserves variance & missingness signal | Adds columns, needs a seed | No |
| KNN Imputer | Fills from k similar rows | Uses multivariate structure | Sensitive to scale, needs enough neighbours | No (real data has no gaps) |
| MICE / IterativeImputer | Chained regression imputation | Most flexible multivariate method | Unstable with very few rows, slower | No |
| Complete Case Analysis | Drops rows with any missing value | No invented values | Discards real data (38% here) | No |

**Final decision:** no imputation needed on the real data. If it were needed, **KNN Imputer** would be
the most defensible choice given the strong `age`↔`income` relationship found in EDA.

## 🚨 12. Outlier Handling

| Method | Feature | Outlier Count | Threshold | Treatment |
|---|---|---|---|---|
| Z-score (\|z\|>3) | income | 0 | ±3 SD | None needed |
| Z-score (\|z\|>3) | amount | 0 | ±3 SD | None needed |
| IQR | income | 0 | 30,375 – 75,375 | None needed |
| IQR | amount | 1 | -813.5 – 3,086.5 | Retained (genuine high-value item) |
| Percentile (5/95) | amount | 1 | flagged the ₹2,599 purchase | Winsorized (capped, not dropped) |

![Boxplot Before](outputs/charts/09_outliers_boxplot_before.png)
![Boxplot After Winsorization](outputs/charts/10_outliers_boxplot_after_winsorize.png)

**No rows are deleted.** The one flagged high-value transaction (₹2,599 Power Bank) matches the product
catalogue price exactly — it's a real purchase, not a data error, so deleting it would only shrink an
already tiny dataset.

## 📅 13. Date/Time Feature Engineering

`transactions.date` is converted to a real `datetime64` and used to derive `txn_year`, `txn_month`,
`txn_day`, `txn_day_of_week`, and `days_since_transaction` (recency), computed against a **fixed,
reproducible reference date = the maximum transaction date in the data (2025-10-10)** rather than
"today". `signup_date`/`last_purchase_date` are **not** processed because `customers.csv` does not
contain those columns — documented as a limitation rather than silently skipped or faked.

## 🔤 14. Categorical Encoding

| Method | Used on | Why |
|---|---|---|
| Label Encoding | `gender` | Genuinely binary — no false order implied |
| One-Hot Encoding | `city`, `category`, `payment_mode` | Nominal, multi-category, no inherent order |
| Ordinal Encoding | Demo column only (`satisfaction_level_DEMO`) | No genuinely ordinal column exists in the real data |

## 📏 15. Numerical Binning

Income is binned both by **equal-width** and **quantile** (`pd.qcut`) methods; **quantile binning is
selected** for the final feature (`income_group`) because it guarantees a balanced split regardless of the
underlying distribution — important with only 8 data points.

![Income Binning](outputs/charts/11_income_binning.png)

## 📐 16. Feature Scaling

| Scaler | Formula | Sensitive to outliers? | Used on |
|---|---|---|---|
| StandardScaler | (x−mean)/std | Yes | age, income |
| MinMaxScaler | (x−min)/(max−min) | Very | (demo only) |
| MaxAbsScaler | x/max\|x\| | Yes | (demo only) |
| RobustScaler | (x−median)/IQR | No | amount |
| Normalizer | row-wise unit norm | N/A (different axis) | (demo only) |

All five are demonstrated individually plus combined via **`ColumnTransformer`** (notebook §11.6), which
applies `RobustScaler` to `amount` (has a legitimate outlier) and `StandardScaler` to `age`/`income`
(no outliers) in a single, reusable, leakage-safe step.

![Scaling Comparison](outputs/charts/12_scaling_comparison_amount.png)

## ⚙️ 17. Feature Transformation

**FunctionTransformer:** `log1p` and `sqrt` on `amount` (right-skewed monetary values, safely handles a
future zero); a guarded `reciprocal` on `stock`.
**PowerTransformer:** `amount` is strictly positive, so **Box-Cox** is applied directly; **Yeo-Johnson** is
also shown as the safer default for any future zero/negative values.

![Transformation Comparison](outputs/charts/13_transformation_comparison_amount.png)

| Version | Skewness |
|---|---|
| Original | 0.81 |
| log1p | 0.08 |
| Box-Cox | 0.03 |
| Yeo-Johnson | 0.03 |

**Box-Cox is selected** for the final `amount_boxcox` feature.

## 🏗️ 18. Feature Construction

| Feature | Formula/Logic | Reason |
|---|---|---|
| `purchase_frequency`, `total_purchase_amount`, `average_purchase_amount` | `groupby("customer_id")` aggregates | Standard RFM-style monetary/frequency signals (correct general logic, even though constant in this 1-txn-per-customer sample) |
| `days_since_transaction` | reference_date − transaction date | Recency signal |
| `age_group`, `income_group` | Binned age / quantile-binned income | Simple, interpretable segments |
| `price_vs_catalogue_diff` | `amount − price` | Captures a real, discovered pricing discrepancy (T007) |
| `amount_boxcox` | Box-Cox transform of amount | Skew-reduced version of the main monetary feature |

## 🔢 19. Binarization

- **`frequent_buyer`** = 1 if `purchase_frequency` > median, else 0 — demonstrated exactly as specified,
  but **honestly reported as uninformative** on this dataset, since every customer has exactly one
  transaction (constant frequency).
- **`high_value_purchase`** = 1 if `amount` > median(amount), else 0 — the threshold is the sample
  median (data-driven, not arbitrary); this is the feature actually used to demonstrate bivariate
  analysis throughout the notebook, given that a genuine `purchased` target cannot be built (§4).

## ✅ 20. Final Dataset

- **Shape:** 8 rows × 35 columns
- **Missing values:** 0 (verified)
- **Duplicate rows:** 0 (verified)
- **Identifiers** (`transaction_id`, `customer_id`, `product_id`) kept for traceability, excluded from the
  numeric ML feature list
- PII (`name`) and zero-variance columns (`txn_year`, `txn_month` — every transaction is Oct 2025) dropped
- File: [`processed_customer_data.csv`](processed_customer_data.csv)

## 💡 21. Key Findings

- All 3 supplied files are genuinely clean — zero missing values, zero duplicates.
- A real ₹200 pricing discrepancy exists between transaction T007 and the product catalogue.
- `age` and `income` are very strongly correlated in this sample (r≈0.98) — likely a small-sample
  artifact, flagged for re-testing at scale.
- Purchase amount is right-skewed; Box-Cox reduces its skew from 0.81 to 0.03.
- Every supplied customer has already purchased — the data cannot support a genuine `purchased` target.

## 🧭 22. Preprocessing Decisions

| Problem | Method(s) Tested | Final Decision | Reason |
|---|---|---|---|
| Missing values (demo) | 6 techniques | None needed on real data | Real data has 0 missing values |
| Outliers in `amount` | Z-score, IQR, Percentile, Winsorization | Keep, don't delete | Legitimate high-value transaction |
| Income binning | Equal-width vs Quantile | Quantile | Balanced split despite small n |
| Scaling `amount` | Standard, MinMax, MaxAbs, Robust, Normalizer | RobustScaler | Resistant to the kept outlier |
| Transforming `amount` | log1p, sqrt, Box-Cox, Yeo-Johnson | Box-Cox | Strictly positive data; largest skew reduction |
| API merge | — | Not merged | No valid shared key with real customers |
| `purchased` target | — | Not constructed; `high_value_purchase` used as demo | No non-purchasing customers in the data |

## 📊 23. Before vs After

| Aspect | Before | After |
|---|---|---|
| `date` dtype | object (text) | datetime64 |
| `amount` skewness | 0.81 | 0.03 (Box-Cox) |
| Categorical columns | Raw text | Label / One-Hot encoded |
| `income` | Continuous | Continuous + `income_group` (binned) |
| Outlier in `amount` | Present, untouched | Retained (Winsorized version available separately) |

## 📁 24. Project Structure

```
customer-purchase-propensity/
│
├── README.md
├── DataPreprocessing.ipynb
├── feature_pipeline.py
├── processed_customer_data.csv
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── data/
│   ├── customers.csv
│   ├── transactions.json
│   ├── products.sql
│   └── api_users_snapshot.json
│
├── outputs/
│   └── charts/
│       ├── 01_income_distribution.png
│       ├── 02_age_distribution.png
│       ├── 03_purchase_amount_distribution.png
│       ├── 04_categorical_distributions.png
│       ├── 05_income_vs_purchase_value.png
│       ├── 06_age_vs_amount.png
│       ├── 07_amount_by_category_and_payment.png
│       ├── 08_pairplot_numeric.png
│       ├── 09_outliers_boxplot_before.png
│       ├── 10_outliers_boxplot_after_winsorize.png
│       ├── 11_income_binning.png
│       ├── 12_scaling_comparison_amount.png
│       ├── 13_transformation_comparison_amount.png
│       └── heatmap_correlation.png
│
└── report/
    └── summary_report.md
```

## 🚀 25. How to Run

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd customer-purchase-propensity

# 2. Create a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the notebook
jupyter notebook DataPreprocessing.ipynb
# (or run the standalone script)
python feature_pipeline.py

# 5. View the processed CSV
# -> processed_customer_data.csv
```

## 📦 26. Requirements

See [`requirements.txt`](requirements.txt): `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`,
`scikit-learn`, `requests`, `jupyter`.

## ⚠️ 27. Limitations

- **Very small sample** — 8 customers, 8 transactions, 6 products. Every statistic here is genuinely
  computed but should be read as demonstrating *method*, not as a robust finding.
- **No non-purchasing customers** in the supplied data, so a real `purchased` target could not be built.
- The DummyJSON API could not be safely merged (no shared key with the real customer data).
- No `signup_date`/`last_purchase_date` in the supplied data, so recency features are based on
  transaction date only.
- This project stops at feature engineering — **no model is trained or evaluated**.

## 🔮 28. Future Improvements

Larger, longitudinal dataset with repeat customers; genuine non-purchasing customers for a real
`purchased` label; real signup/last-purchase dates; train/test split; model training, cross-validation,
and evaluation; deployment.

## 📝 29. Conclusion

This project delivers a fully executed, genuinely reproducible data cleaning and feature engineering
pipeline across four real data sources, with every technique demonstrated, every chart generated from
actual data, and every limitation of the (very small) dataset stated honestly rather than papered over.
The resulting `processed_customer_data.csv` is clean, leakage-aware, and ready to serve as the foundation
for a future classification model.

## ☑️ 30. Academic / Exam Checklist

| Exam Requirement | Completed | Location |
|---|---|---|
| Project planning & problem framing | ✅ | Notebook §0, README §1–4 |
| Load CSV / JSON / SQL / API | ✅ | Notebook §2, `feature_pipeline.py` |
| Data understanding (shape, dtypes, describe, etc.) | ✅ | Notebook §3 |
| Merge on customer_id / product_id + validation | ✅ | Notebook §4 |
| API merge decision (no false relationship) | ✅ | Notebook §4.5, README §9 |
| Univariate / bivariate / multivariate EDA | ✅ | Notebook §5, README §10 |
| Correlation heatmap (numeric only) | ✅ | `outputs/charts/heatmap_correlation.png` |
| 6 missing-value techniques on labelled demo data | ✅ | Notebook §6 |
| 3+ outlier methods + Winsorization | ✅ | Notebook §7 |
| Datetime conversion + derived features + recency | ✅ | Notebook §8 |
| Mixed-variable / identifier handling | ✅ | Notebook §8 |
| Label / One-Hot / Ordinal encoding | ✅ | Notebook §9 |
| Numerical binning (equal-width + quantile) | ✅ | Notebook §10 |
| 5 scalers + ColumnTransformer | ✅ | Notebook §11 |
| Feature construction | ✅ | Notebook §12 |
| FunctionTransformer (log, sqrt, reciprocal) | ✅ | Notebook §13.1 |
| PowerTransformer (Box-Cox, Yeo-Johnson) | ✅ | Notebook §13.2 |
| Binarization with justified threshold | ✅ | Notebook §12.6 |
| Target leakage discussion | ✅ | Notebook §14.4 |
| Final CSV export, ML-ready | ✅ | `processed_customer_data.csv` |
| `feature_pipeline.py` reusable script | ✅ | Repo root |
| `DataPreprocessing.ipynb` fully executed | ✅ | Repo root |
| README with real generated charts | ✅ | This file |
| `report/summary_report.md` | ✅ | `report/summary_report.md` |
| GitHub-ready repo structure + LICENSE | ✅ | Repo root |
