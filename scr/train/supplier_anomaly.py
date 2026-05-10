"""
supplier_anomaly.py

Що робить:
- Виявляє аномальних підрядників ансамблем з 3 підходів:
  1) K-Means + silhouette (колективні відхилення),
  2) Isolation Forest (точкові відхилення),
  3) LOF (локальні відхилення).

Коли використовується:
- Після підготовки `supplier_level_features.csv`.

Навіщо:
- Реалізує підхід із дипломної записки для unsupervised-аналізу профілю
  постачальників без еталонних міток.
"""
import pandas as pd
import numpy as np
import warnings

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.stats import zscore


SUPPLIER_FEATURES = [
    "num_bids",
    "num_wins",
    "avg_competitors",
    "rejected_bids",
    "complaints",

    "log_avg_contract",
    "log_avg_price_per_unit",

    "contract_changes_share",
    "win_rate",

    "log_max_contract",
    "log_min_contract",

    # "avg_contract_missing",
    # "max_contract_missing",
    # "min_contract_missing",
    # "avg_price_per_unit_missing",
    # "win_rate_missing",
]


def load_supplier_data(path="../../data/raw/supplier_level_features.csv"):
    """Завантажує ознаки підрядників і ставить `supplier_id` як індекс."""
    df = pd.read_csv(path)
    df.set_index("supplier_id", inplace=True)
    return df


def prepare_features(df):
    """Медіанна імпутація + RobustScaler для стійкості до викидів."""
    X = df[SUPPLIER_FEATURES]

    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()

    X_imp = imputer.fit_transform(X)
    return scaler.fit_transform(X_imp)


def _norm(series: pd.Series) -> pd.Series:
    """Min-max нормалізація в [0, 1]"""
    mn, mx = series.min(), series.max()
    if mx - mn < 1e-9:
        return pd.Series(0.0, index=series.index)
    return (series - mn) / (mx - mn)


def detect_supplier_anomalies(path="../../data/raw/supplier_level_features.csv"):
    """
    Рахує ризиковість підрядників і повертає ранжований DataFrame.

    Логіка ансамблю:
    - `ensemble_outlier=1`, якщо принаймні 2 з 3 методів позначили аномалію.
    - `risk_score` — середнє z-нормованих сил аномальності + невеликий бонус
      за консенсус ансамблю.
    """
    df = load_supplier_data(path)
    X = prepare_features(df)

    # --- 1) K-Means for "collective" outliers ---
    # Thesis: choose K using silhouette, then treat negative silhouette as suspicious.
    n = X.shape[0]
    if n < 3:
        raise ValueError(f"Not enough supplier samples for anomaly detection: n={n}")

    k_min = 2
    k_max = min(15, n - 1)
    if k_max < k_min:
        k_best = k_min
    else:
        # Silhouette can be expensive; for large n we score on a subset.
        score_idx = np.arange(n)
        if n > 2000:
            score_idx = np.random.choice(n, size=2000, replace=False)
        X_score = X[score_idx]

        k_best = k_min
        best_s = -np.inf
        for k in range(k_min, k_max + 1):
            try:
                km = KMeans(n_clusters=k, n_init=10, random_state=42)
                labels = km.fit_predict(X_score)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    s = silhouette_score(X_score, labels, metric="euclidean")
                if s > best_s:
                    best_s = s
                    k_best = k
            except Exception:
                continue

    kmeans = KMeans(n_clusters=k_best, n_init=10, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)
    sil_samples = silhouette_samples(X, kmeans_labels, metric="euclidean")
    df["cluster"] = kmeans_labels
    df["kmeans_silhouette"] = sil_samples
    kmeans_outlier_label = sil_samples < 0
    df["cluster_outlier"] = kmeans_outlier_label.astype(int)

    kmeans_strength_raw = -sil_samples  # higher => more anomalous
    df["kmeans_score"] = zscore(kmeans_strength_raw)

    # --- 2) Isolation Forest for "point" outliers ---
    iso = IsolationForest(
        n_estimators=500,
        contamination="auto",
        random_state=42,
    )
    iforest_pred = iso.fit_predict(X)  # -1 = outlier
    iforest_strength_raw = -iso.score_samples(X)  # higher => more anomalous
    df["iforest_score"] = zscore(iforest_strength_raw)
    df["iforest_outlier"] = (iforest_pred == -1).astype(int)

    # --- 3) LOF for local outliers ---
    n_neighbors = int(min(20, max(2, n - 1)))
    try:
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination="auto")
    except TypeError:
        # Older sklearn versions may not support contamination="auto"
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.1)

    lof_pred = lof.fit_predict(X)  # -1 = outlier
    lof_strength_raw = -lof.negative_outlier_factor_  # higher => more anomalous
    df["lof_score"] = zscore(lof_strength_raw)
    df["lof_outlier"] = (lof_pred == -1).astype(int)

    # --- Ensemble: anomaly if at least 2 out of 3 ---
    kmeans_out = df["cluster_outlier"].astype(int)
    iforest_out = df["iforest_outlier"].astype(int)
    lof_out = df["lof_outlier"].astype(int)
    df["ensemble_outlier"] = ((kmeans_out + iforest_out + lof_out) >= 2).astype(int)

    # Final ranking score: average z-scored strengths, with a small bonus for ensemble outliers.
    df["risk_score"] = (_norm(df["kmeans_score"])
                        + _norm(df["iforest_score"])
                        + _norm(df["lof_score"])
                        ) / 3.0 + 0.25 * df["ensemble_outlier"].astype(float)

    df["risk_score"] = _norm(df["risk_score"])

    df["risk_rank"] = df["risk_score"].rank(ascending=False, method="min")
    return df.reset_index().sort_values("risk_score", ascending=False)


if __name__ == "__main__":
    result = detect_supplier_anomalies()
    result.to_csv("../../data/prepared/supplier_anomaly_results.csv", index=False)
