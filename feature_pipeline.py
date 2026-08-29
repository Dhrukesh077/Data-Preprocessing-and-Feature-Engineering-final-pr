"""
feature_pipeline.py
--------------------
Reusable data cleaning + feature engineering pipeline for the
Customer Purchase Propensity project.

Loads customers.csv, transactions.json and products.sql, merges them,
engineers features, and writes processed_customer_data.csv.

This is the clean "production" version of the logic explored and
explained step-by-step in DataPreprocessing.ipynb. Run directly:

    python feature_pipeline.py

Requires: pandas, numpy, scikit-learn, scipy  (see requirements.txt)
"""

import json
import sqlite3

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder, PowerTransformer

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_DIR = "data"
OUTPUT_CSV = "processed_customer_data.csv"


def load_customers(path=f"{DATA_DIR}/customers.csv"):
    """Load customer demographics from CSV."""
    return pd.read_csv(path)


def load_transactions(path=f"{DATA_DIR}/transactions.json"):
    """Load transaction records from JSON."""
    with open(path) as f:
        data = json.load(f)
    return pd.DataFrame(data)


def load_products(path=f"{DATA_DIR}/products.sql"):
    """Create a temporary SQLite DB, run products.sql, and read the products table."""
    conn = sqlite3.connect(":memory:")
    with open(path) as f:
        conn.executescript(f.read())
    products = pd.read_sql("SELECT * FROM products;", conn)
    conn.close()
    return products


def merge_sources(customers, transactions, products):
    """Merge customers + transactions + products on customer_id / product_id."""
    df = transactions.merge(customers, on="customer_id", how="left")
    df = df.merge(products, on="product_id", how="left")
    assert df.isnull().sum().sum() == 0, "Merge introduced unexpected missing values"
    assert len(df) == len(transactions), "Merge changed row count unexpectedly"
    return df


def engineer_date_features(df):
    """Convert transaction date to datetime and derive year/month/day/day-of-week/recency.

    NOTE: signup_date / last_purchase_date are intentionally NOT processed here because
    they do not exist in the supplied customers.csv (see notebook Section 8.1).
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["txn_year"] = df["date"].dt.year
    df["txn_month"] = df["date"].dt.month
    df["txn_day"] = df["date"].dt.day
    df["txn_day_of_week"] = df["date"].dt.day_name()

    reference_date = df["date"].max()  # reproducible; documented in the notebook
    df["days_since_transaction"] = (reference_date - df["date"]).dt.days
    return df


def engineer_aggregate_features(df):
    """Customer-level purchase aggregates via groupby (generalises to repeat customers)."""
    df = df.copy()
    agg = df.groupby("customer_id").agg(
        purchase_frequency=("transaction_id", "count"),
        total_purchase_amount=("amount", "sum"),
        average_purchase_amount=("amount", "mean"),
    ).reset_index()
    return df.merge(agg, on="customer_id", how="left")


def engineer_bins_and_segments(df):
    """Quantile income binning + age segmentation."""
    df = df.copy()
    df["income_group"] = pd.qcut(df["income"], q=3, labels=["Low", "Medium", "High"])
    df["age_group"] = pd.cut(
        df["age"], bins=[0, 25, 35, 100],
        labels=["Young Adult (<=25)", "Adult (26-35)", "Mature Adult (36+)"]
    )
    return df


def engineer_quality_and_transform_features(df):
    """Data-quality diff feature + Box-Cox transform of amount (skew reduction)."""
    df = df.copy()
    df["price_vs_catalogue_diff"] = df["amount"] - df["price"]

    pt = PowerTransformer(method="box-cox")
    df["amount_boxcox"] = pt.fit_transform(df[["amount"]])
    return df


def encode_features(df):
    """Label-encode binary gender; one-hot encode nominal category/payment_mode."""
    df = df.copy()
    df["gender_label_encoded"] = LabelEncoder().fit_transform(df["gender"])

    category_ohe = pd.get_dummies(df["category"], prefix="category", dtype=int)
    payment_ohe = pd.get_dummies(df["payment_mode"], prefix="payment", dtype=int)
    return pd.concat([df, category_ohe, payment_ohe], axis=1)


def construct_binarized_features(df):
    """frequent_buyer (exam-specified) + high_value_purchase (data-driven demo target)."""
    df = df.copy()
    freq_threshold = df["purchase_frequency"].median()
    df["frequent_buyer"] = (df["purchase_frequency"] > freq_threshold).astype(int)

    amount_median = df["amount"].median()
    df["high_value_purchase"] = (df["amount"] > amount_median).astype(int)
    return df


def select_final_columns(df):
    """Assemble the final ML-ready dataset; drop PII / zero-variance / redundant columns.

    See notebook Section 14.1 for the full reasoning behind every drop decision.
    """
    identifier_cols = ["transaction_id", "customer_id", "product_id"]
    reference_cols = ["date", "gender", "city", "category", "payment_mode",
                       "product_name", "age_group", "income_group"]
    numeric_ml_cols = ["age", "income", "amount", "price", "stock", "days_since_transaction",
                       "purchase_frequency", "total_purchase_amount", "average_purchase_amount",
                       "price_vs_catalogue_diff", "amount_boxcox", "gender_label_encoded",
                       "txn_day", "frequent_buyer"]
    target_col = ["high_value_purchase"]

    ohe_cols = [c for c in df.columns
                if (c.startswith("category_")) or (c.startswith("payment_") and c != "payment_mode")]

    final_cols = identifier_cols + reference_cols + numeric_ml_cols + target_col + ohe_cols
    return df[final_cols].copy()


def run_pipeline():
    """End-to-end: load -> merge -> engineer -> encode -> select -> export."""
    print("Loading data sources...")
    customers = load_customers()
    transactions = load_transactions()
    products = load_products()

    print("Merging on customer_id / product_id...")
    df = merge_sources(customers, transactions, products)

    print("Engineering date, aggregate, binning, transform and encoding features...")
    df = engineer_date_features(df)
    df = engineer_aggregate_features(df)
    df = engineer_bins_and_segments(df)
    df = engineer_quality_and_transform_features(df)
    df = encode_features(df)
    df = construct_binarized_features(df)

    print("Selecting final ML-ready columns...")
    final_df = select_final_columns(df)

    assert final_df.isnull().sum().sum() == 0, "Unintended missing values in final dataset"
    assert final_df.duplicated().sum() == 0, "Unintended duplicate rows in final dataset"

    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Done. Final dataset shape: {final_df.shape} -> {OUTPUT_CSV}")
    return final_df


if __name__ == "__main__":
    run_pipeline()
