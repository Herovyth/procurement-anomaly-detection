import pandas as pd
import numpy as np
import umap

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore

import hdbscan


SUPPLIER_FEATURES = [
    "num_bids",
    "num_wins",
    "avg_competitors",
    "rejected_bids",
    "complaints",

    "avg_price_per_unit",
    "avg_contract",
    "max_contract",
    "min_contract",

    "contract_changes_share",
    "win_rate",

    "log_avg_contract",
    "log_max_contract",
    "log_min_contract",
    "log_avg_price_per_unit",

    # "avg_contract_missing",
    # "max_contract_missing",
    # "min_contract_missing",
    # "avg_price_per_unit_missing",
    # "win_rate_missing",
]


def load_supplier_data(path="data/supplier_level_features.csv"):
    df = pd.read_csv(path)
    df.set_index("supplier_id", inplace=True)
    return df


def prepare_features(df):
    X = df[SUPPLIER_FEATURES]

    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()

    X_imp = imputer.fit_transform(X)
    return scaler.fit_transform(X_imp)


def detect_supplier_anomalies(path="data/supplier_level_features.csv"):
    df = load_supplier_data(path)
    X = prepare_features(df)

    reducer = umap.UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.1,
        metric="euclidean",
        random_state=42
    )
    X_umap = reducer.fit_transform(X)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=15,   # ↓
        min_samples=15,         # ↓
        metric="euclidean"
    )
    df["cluster"] = clusterer.fit_predict(X_umap)
    df["cluster_outlier"] = df["cluster"] == -1

    iso = IsolationForest(
        n_estimators=500,
        contamination="auto",
        random_state=42
    )
    iso.fit(X)

    if_score = -iso.score_samples(X)
    df["iforest_score"] = zscore(if_score)

    df["risk_score"] = (
        1.5 * df["iforest_score"] +
        2.0 * df["cluster_outlier"].astype(int)
    )

    df["risk_rank"] = df["risk_score"].rank(ascending=False)

    return df.reset_index().sort_values("risk_score", ascending=False)


if __name__ == "__main__":
    result = detect_supplier_anomalies()
    result.to_csv("data/supplier_anomaly_results.csv", index=False)
