
"""Failure-risk baseline for Tawseel COD parcels."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

TARGET = "failed"

FEATURE_COLUMNS: list[str] = [
    "customer_id",
    "merchant_id",
    "area",
    "created_at",
    "promised_slot",
    "cod_amount",
    "n_items",
    "address_text",
]

HISTORY_COLUMNS = [
    "customer_prior_parcels",
    "customer_prior_fail_rate",
]



def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    grouped = (
        df.groupby(
            ["customer_id", "created_at"],
            dropna=False,
            sort=False,
        )["failed"]
        .agg(["count", "sum"])
        .sort_index()
    )

    customer_groups = grouped.groupby(
        level="customer_id",
        sort=False,
    )

    grouped["prior_count"] = (
        customer_groups["count"].cumsum() - grouped["count"]
    )

    grouped["prior_failures"] = (
        customer_groups["sum"].cumsum() - grouped["sum"]
    )

    keys = pd.MultiIndex.from_frame(
        df[["customer_id", "created_at"]]
    )

    history = grouped.reindex(keys)

    prior_count = history["prior_count"].to_numpy(dtype=float)
    prior_failures = history["prior_failures"].to_numpy(dtype=float)

    result["customer_prior_parcels"] = prior_count.astype(np.int64)

    result["customer_prior_fail_rate"] = np.divide(
        prior_failures,
        prior_count,
        out=np.full(len(df), np.nan),
        where=prior_count > 0,
    )

    return result



def time_split(
    df: pd.DataFrame,
    time_col: str = "created_at",
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")

    if len(df) == 0:
        return df.copy(), df.copy()

    n_test = int(np.ceil(len(df) * test_fraction))
    sorted_df = df.sort_values(time_col)
    boundary = sorted_df[time_col].iloc[len(df) - n_test]

    train = df.loc[df[time_col] < boundary].copy()
    test = df.loc[df[time_col] >= boundary].copy()

    return train, test


def _prepare_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.loc[:, FEATURE_COLUMNS + HISTORY_COLUMNS].copy()

    dates = pd.to_datetime(X["created_at"], errors="coerce")

    X["created_hour"] = dates.dt.hour
    X["created_dayofweek"] = dates.dt.dayofweek
    X["created_month"] = dates.dt.month

    return X.drop(columns=["created_at"])


def build_pipeline(seed: int = 0) -> Pipeline:

    numeric_features = [
        "cod_amount",
        "n_items",
        "customer_prior_parcels",
        "customer_prior_fail_rate",
        "created_hour",
        "created_dayofweek",
        "created_month",
    ]

    categorical_features = [
        "customer_id",
        "merchant_id",
        "area",
        "promised_slot",
        "address_text",
    ]

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                keep_empty_features=True,
            ),
        ),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="__MISSING__",
            ),
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ])

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    return Pipeline([
        (
            "feature_engineering",
            FunctionTransformer(
                _prepare_features,
                validate=False,
            ),
        ),
        ("preprocessing", preprocessing),
        (
            "classifier",
            LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=seed,
            ),
        ),
    ])


def evaluate(model, test_df: pd.DataFrame) -> dict:

    y_true = test_df[TARGET].to_numpy()
    probabilities = model.predict_proba(test_df)[:, 1]

    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "base_rate": float(y_true.mean()),
        "n": int(len(y_true)),
    }


def compare_to_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int = 0,
) -> dict:

    input_columns = FEATURE_COLUMNS + HISTORY_COLUMNS

    X_train = train_df.loc[:, input_columns]
    y_train = train_df[TARGET]

    model = build_pipeline(seed)
    model.fit(X_train, y_train)

    model_results = evaluate(model, test_df)

    dummy = DummyClassifier(strategy="prior")
    dummy.fit(X_train, y_train)

    class DummyAdapter:
        def predict_proba(self, X):
            return dummy.predict_proba(X.loc[:, input_columns])

    dummy_results = evaluate(DummyAdapter(), test_df)

    return {
        "model": model_results,
        "dummy": dummy_results,
        "lift_pr_auc": float(
            model_results["pr_auc"] - dummy_results["pr_auc"]
        ),
    }
