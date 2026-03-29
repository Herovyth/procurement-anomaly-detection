from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import zscore

import umap
import hdbscan
from sentence_transformers import SentenceTransformer

import joblib
import os

CACHE_DIR = "data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

HIGH_SIGNAL_MISSING = [
    "description_missing",
    "winner_price_missing",
]

LOW_SIGNAL_MISSING = [
    "value_amount_missing",
    "price_per_unit_missing",
    "value_per_day_missing",
    "min_bid_missing",
    "winner_minus_min_missing",
    "tender_duration_days_missing",
    "completion_days_missing",
    "cpv_deviation_missing",
    "doc_nlp_pca_1_missing",
    "doc_nlp_pca_2_missing",
    "doc_nlp_pca_3_missing",
    "doc_nlp_pca_4_missing",
    "doc_nlp_pca_5_missing",
]

SMALL_CPV_GROUP_FALLBACK_K = 100  # fallback to pooled (global) normalization for small CPV groups
CPV2_MIN_TENDERS_FOR_MODEL = 100  # min tenders in CPV2 group to train a dedicated IF; else use fallback IF
SEMANTIC_GROUP_MIN_TENDERS = 100  # min tenders in group to compute semantic-outlier score


NUMERIC_FEATURES = [
    # Competition / counts
    "numberOfBids",
    "numberOfAdmitted",
    "bids_cpv_zscore",
    "admitted_cpv_zscore",
    "admitted_ratio",

    # CPV-normalized and price-related
    "cpv_deviation",

    # Log-valued signals (stable across CPV scale differences)
    "log_value_amount",
    "log_winner_price",
    "log_price_per_unit",
    "log_value_per_day",
    "log_min_bid",
    "price_drop_ratio",

    # Tender timing & differences
    "tender_duration_days",
    "completion_days",
    "winner_minus_min",

    # Behavioral / process
    "complaints",
    "tender_changes",
    "contract_changes",

    # Missingness: only high-signal flags + one aggregated counter
    "description_missing",
    "winner_price_missing",
    "total_missing_count",

    # Document features
    "doc_nlp_pca_1",
    "doc_nlp_pca_2",
    "doc_nlp_pca_3",
    "doc_nlp_pca_4",
    "doc_nlp_pca_5",
]


TEXT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Інтерпретований глобальний blend: semantic_outlier [0,1], semantic_pct [0,1], numeric_pct [0,1]
# Ваги узгоджені з відносним співвідношенням у risk_score (2 : 1.5 : 1), нормовані на суму 4.5.
RISK_BLEND_WEIGHTS = {
    "semantic_outlier": 2.0 / 4.5,
    "semantic_pct": 1.5 / 4.5,
    "numeric_pct": 1.0 / 4.5,
}

MODEL = None


def compute_embeddings_real(texts_real, suffix="real"):
    """Вираховує або завантажує embeddings лише для реальних тендерів."""
    cache_path = f"{CACHE_DIR}/embeddings_{suffix}.npy"

    if os.path.exists(cache_path):
        emb = np.load(cache_path)
        if len(emb) == len(texts_real):
            print(f"Завантажуємо embeddings для {suffix} з кешу...", flush=True)
            return emb
        print(f"{suffix} embeddings cache mismatch. Recomputing...")

    global MODEL
    if MODEL is None:
        print("Завантажуємо NLP модель...", flush=True)
        MODEL = SentenceTransformer(TEXT_MODEL)

    print(f"Рахуємо embeddings для {suffix}...", flush=True)
    emb = MODEL.encode(texts_real, show_progress_bar=True, normalize_embeddings=True)
    emb = emb.astype(np.float32)
    np.save(cache_path, emb)
    return emb


def compute_embeddings_synthetic(texts_synth, suffix="synth"):
    cache_path = f"{CACHE_DIR}/embeddings_{suffix}.npy"

    global MODEL
    if MODEL is None:
        print("Завантажуємо NLP модель...", flush=True)
        MODEL = SentenceTransformer(TEXT_MODEL)

    print(f"Рахуємо embeddings для {suffix}...", flush=True)
    emb = MODEL.encode(texts_synth, show_progress_bar=True, normalize_embeddings=True)
    emb = emb.astype(np.float32)
    np.save(cache_path, emb)
    return emb


def reduce_embeddings_with_real_and_synth(emb_real, emb_synth=None):
    """
    UMAP: тренуємо на real, потім трансформуємо synthetic.
    """
    model_path = f"{CACHE_DIR}/umap_model.joblib"
    reduced_real_path = f"{CACHE_DIR}/umap_reduced_real.npy"
    reduced_synth_path = f"{CACHE_DIR}/umap_reduced_synth.npy"

    # 1) UMAP для реальних
    if os.path.exists(reduced_real_path) and os.path.exists(model_path):
        reducer = joblib.load(model_path)
        reduced_real = np.load(reduced_real_path)
        if len(reduced_real) == len(emb_real):
            print("Завантажуємо UMAP для real з кешу...", flush=True)
        else:
            print("UMAP cache mismatch для real. Тренуємо заново...")
            reducer = umap.UMAP(
                n_neighbors=15, n_components=5, metric="cosine",
                random_state=42, low_memory=True
            )
            reduced_real = reducer.fit_transform(emb_real)
            np.save(reduced_real_path, reduced_real)
            joblib.dump(reducer, model_path)
    else:
        print("Тренуємо UMAP для real...", flush=True)
        reducer = umap.UMAP(
            n_neighbors=15, n_components=5, metric="cosine",
            random_state=42, low_memory=True
        )
        reduced_real = reducer.fit_transform(emb_real)
        np.save(reduced_real_path, reduced_real)
        joblib.dump(reducer, model_path)

    # 2) UMAP для synthetic
    reduced_synth = None
    if emb_synth is not None:
        print("Проєктуємо synthetic в той же простір UMAP...", flush=True)
        reduced_synth = reducer.transform(emb_synth)
        np.save(reduced_synth_path, reduced_synth)

    if reduced_synth is not None:
        return np.vstack([reduced_real, reduced_synth])
    return reduced_real


def load_tender_data(path="data/tender_level_features.csv"):
    return pd.read_csv(path)


def mean_cosine_similarity_sampled(embeddings, sample_size=500):
    idx = np.random.choice(len(embeddings), size=min(sample_size, len(embeddings)), replace=False)
    sample = embeddings[idx]
    sim = cosine_similarity(sample, embeddings)
    return sim.mean(axis=0)


def _to_bool_series(s: pd.Series) -> pd.Series:
    # CSV typically stores bools as "True"/"False" strings; normalize robustly.
    if s.dtype == bool:
        return s
    ss = s.astype(str).str.strip().str.lower()
    return ss.isin({"true", "1", "t", "yes", "y"})


def _ensure_missing_columns(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c not in df.columns:
            df[c] = False


def _cpv_capped_iqr_zscore(
    df: pd.DataFrame,
    value_col: str,
    out_col: str,
    cpv_col: str = "cpv",
    k: int = SMALL_CPV_GROUP_FALLBACK_K,
) -> None:
    """
    Z-score within CPV using (median, IQR) with fallback to pooled (global) stats
    when a CPV group is too small.
    """
    if cpv_col not in df.columns:
        df[cpv_col] = None

    sizes = df.groupby(cpv_col).size()
    sizes_map = df[cpv_col].map(sizes).fillna(0)
    mask_small = sizes_map.lt(k)

    # Pooled stats
    global_median = df[value_col].median()
    global_q25 = df[value_col].quantile(0.25)
    global_q75 = df[value_col].quantile(0.75)
    global_iqr = (global_q75 - global_q25) + 1e-9

    # CPV stats
    cpv_median = df.groupby(cpv_col)[value_col].median()
    cpv_q25 = df.groupby(cpv_col)[value_col].quantile(0.25)
    cpv_q75 = df.groupby(cpv_col)[value_col].quantile(0.75)
    cpv_iqr = (cpv_q75 - cpv_q25) + 1e-9

    median_per_row = df[cpv_col].map(cpv_median)
    iqr_per_row = df[cpv_col].map(cpv_iqr)

    z = (df[value_col] - median_per_row) / iqr_per_row

    # Fallback for small groups
    if mask_small.any():
        z.loc[mask_small] = (df.loc[mask_small, value_col] - global_median) / global_iqr

    df[out_col] = z.replace([np.inf, -np.inf], 0).fillna(0.0)


def _extract_cpv2(cpv: object) -> str:
    """
    CPV is expected in format '########-#'. We use the first two digits as the top-level section (CPV2).
    """
    if cpv is None or (isinstance(cpv, float) and np.isnan(cpv)):
        return "NA"
    s = str(cpv).strip()
    # Keep only digits for safety; CPV2 is first 2 digits of the 8-digit code.
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) < 2:
        return "NA"
    return digits[:2]


def _rank_pct(values: np.ndarray) -> np.ndarray:
    """
    Percentile rank in [0, 1], higher = more anomalous.
    Handles ties via average ranks.
    """
    if len(values) == 0:
        return values
    s = pd.Series(values)
    return s.rank(pct=True, method="average").to_numpy()


def semantic_outliers_by_group(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    group_col: str,
    percentile: float = 5,
    min_group_size: int = SEMANTIC_GROUP_MIN_TENDERS,
) -> np.ndarray:
    """
    Semantic outlier score per group via low mean cosine similarity.
    For small groups, falls back to a global score computed over the full dataset.
    Returns a z-scored vector aligned with df.index.
    """
    n = len(df)
    outlier_score = np.zeros(n, dtype=float)

    # Global baseline (fallback)
    mean_sim_global = mean_cosine_similarity_sampled(embeddings)
    thr_global = np.percentile(mean_sim_global, percentile)
    global_raw = thr_global - mean_sim_global

    # Group-specific where possible
    for g, group in df.groupby(group_col):
        idx = group.index.to_numpy()
        if len(idx) < min_group_size:
            outlier_score[idx] = global_raw[idx]
            continue

        emb = embeddings[idx]
        mean_sim = mean_cosine_similarity_sampled(emb)
        threshold = np.percentile(mean_sim, percentile)
        outlier_score[idx] = threshold - mean_sim

    return zscore(outlier_score)


def numeric_anomaly_score_by_group(
    X_scaled: np.ndarray,
    groups: pd.Series,
    min_group_size: int = CPV2_MIN_TENDERS_FOR_MODEL,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Train IsolationForest per group; for small groups use a pooled fallback model.
    Returns (numeric_zscore, numeric_percentile) aligned with the original row order.
    """
    n = X_scaled.shape[0]
    raw = np.zeros(n, dtype=float)

    # Fallback model trained on all data
    fallback_if = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=42,
    )
    fallback_if.fit(X_scaled)
    fallback_raw = -fallback_if.score_samples(X_scaled)

    # Group sizes
    sizes = groups.value_counts(dropna=False)

    # Fill per group
    for g, idx in groups.groupby(groups).groups.items():
        idx = np.asarray(list(idx), dtype=int)
        if sizes.get(g, 0) < min_group_size:
            raw[idx] = fallback_raw[idx]
            continue

        iso = IsolationForest(
            n_estimators=1000,
            contamination="auto",
            random_state=42,
        )
        iso.fit(X_scaled[idx])
        raw[idx] = -iso.score_samples(X_scaled[idx])

    # Normalize within group for global comparability
    z = np.zeros(n, dtype=float)
    pct = np.zeros(n, dtype=float)
    for g, idx in groups.groupby(groups).groups.items():
        idx = np.asarray(list(idx), dtype=int)
        z[idx] = zscore(raw[idx]) if len(idx) > 1 else 0.0
        pct[idx] = _rank_pct(raw[idx])

    return z, pct


def prepare_numeric_features(df):
    X = df[NUMERIC_FEATURES]

    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()

    X_imp = imputer.fit_transform(X)
    return scaler.fit_transform(X_imp)


def numeric_anomaly_score(X):
    """
    Backwards-compatible helper: global numeric score (z-scored IF raw scores).
    Prefer numeric_anomaly_score_by_group for CPV-hierarchical models.
    """
    iso = IsolationForest(
        n_estimators=1000,
        contamination="auto",
        random_state=42
    )
    iso.fit(X)
    scores = -iso.score_samples(X)
    return zscore(scores)


def detect_tender_anomalies(
    path: Optional[str] = "data/tender_level_features.csv",
    df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Якщо передано df — використовується він (наприклад, real + synthetic для оцінки).
    Інакше читається CSV за path.
    """

    if df is None:
        df = load_tender_data(path)
    else:
        df = df.copy()

    is_synth = df["tender_id"].astype(str).str.startswith("SYNTH_")

    df["text"] = (
        df["title"].fillna("") + ". " +
        df["description"].fillna("")
    )

    df_real = df[~is_synth].copy()
    df_synth = df[is_synth].copy()

    # ------------------------------
    # Feature engineering (CPV-robust + missingness consolidation)
    # ------------------------------
    _ensure_missing_columns(df, HIGH_SIGNAL_MISSING + LOW_SIGNAL_MISSING)
    for c in HIGH_SIGNAL_MISSING + LOW_SIGNAL_MISSING:
        df[c] = _to_bool_series(df[c])

    # 1) aggregated missingness counter (replaces most *_missing flags)
    df["total_missing_count"] = df[LOW_SIGNAL_MISSING].astype(int).sum(axis=1)

    # 2) CPV-normalized competition signals with small-CPV fallback
    _cpv_capped_iqr_zscore(df, value_col="numberOfBids", out_col="bids_cpv_zscore")
    _cpv_capped_iqr_zscore(df, value_col="numberOfAdmitted", out_col="admitted_cpv_zscore")

    # 3) admitted ratio (guard division by 0)
    bids = df["numberOfBids"].fillna(0)
    admitted = df["numberOfAdmitted"].fillna(0)
    df["admitted_ratio"] = np.where(bids > 0, admitted / bids, 0.0)

    # 4) price drop ratio vs CPV expected (expected = median_cpv)
    expected = df["median_cpv"].replace([np.inf, -np.inf], np.nan)
    winner = df["winner_price"].replace([np.inf, -np.inf], np.nan)
    denom = expected.replace(0, np.nan) + 1e-9
    df["price_drop_ratio"] = ((expected - winner) / denom).replace([np.inf, -np.inf], 0).fillna(0.0)

    # 5) gating / neutralization when winner_price is missing:
    #    suppress derived signals (price_drop_ratio) and the copied log signal (log_winner_price)
    winner_price_missing_mask = df["winner_price_missing"].astype(bool)
    df.loc[winner_price_missing_mask, "price_drop_ratio"] = 0.0
    df.loc[winner_price_missing_mask, "log_winner_price"] = 0.0

    # CPV2 (top-level section) for hierarchical modeling
    df["cpv2"] = df["cpv"].apply(_extract_cpv2)

    # ------------------------------
    # Обчислення embeddings і зниження розмірності
    # ------------------------------
    emb_real = compute_embeddings_real(df_real["text"].tolist())
    emb_synth = compute_embeddings_synthetic(df_synth["text"].tolist())

    embeddings = np.vstack([emb_real, emb_synth])
    # Проєкція у спільний простір через UMAP
    emb_reduced = reduce_embeddings_with_real_and_synth(emb_real, emb_synth)

    # HDBSCAN на спільному UMAP-просторі
    hdbscan_path = f"{CACHE_DIR}/hdbscan_labels.npy"
    clusters = None
    if os.path.exists(hdbscan_path):
        clusters = np.load(hdbscan_path)
        if len(clusters) != len(df):
            print("[WARN] HDBSCAN cache size mismatch. Recomputing...")
            clusters = None

    if clusters is None:
        print("Запускаємо HDBSCAN...", flush=True)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=50,
            min_samples=20,
            core_dist_n_jobs=1,
            metric="euclidean"
        )
        clusters = clusterer.fit_predict(emb_reduced)
        np.save(hdbscan_path, clusters)
        joblib.dump(clusterer, f"{CACHE_DIR}/hdbscan_model.joblib")

    df["semantic_cluster"] = clusters

    # ------------------------------
    # Semantic outlier score (за групами CPV2, fallback на глобальний)
    # ------------------------------
    semantic_group_score = semantic_outliers_by_group(df, embeddings, group_col="cpv2")
    df["semantic_score"] = semantic_group_score
    df["semantic_pct"] = df["semantic_score"].rank(pct=True, method="average")

    # ------------------------------
    # Numeric score: IsolationForest на всьому df (реальні + synthetic)
    # ------------------------------
    if_scores_path = f"{CACHE_DIR}/if_scores.npy"
    if os.path.exists(if_scores_path):
        print("Завантажуємо IsolationForest scores з кешу...", flush=True)
        scores = np.load(if_scores_path, allow_pickle=True).item()
        numeric_z = scores["z"]
        numeric_pct = scores["pct"]
        # Перевірка довжини
        if len(numeric_z) != len(df):
            print("[WARN] Cached numeric scores size mismatch! Recomputing...")
            X_num_scaled = prepare_numeric_features(df)
            numeric_z, numeric_pct = numeric_anomaly_score_by_group(X_num_scaled, groups=df["cpv2"])
            np.save(if_scores_path, {"z": numeric_z, "pct": numeric_pct})
    else:
        print("Тренуємо IsolationForest на всіх тендерах...", flush=True)
        X_num_scaled = prepare_numeric_features(df)
        numeric_z, numeric_pct = numeric_anomaly_score_by_group(X_num_scaled, groups=df["cpv2"])
        np.save(if_scores_path, {"z": numeric_z, "pct": numeric_pct})

    # Тепер точно match по довжині
    df["numeric_score"] = numeric_z
    df["numeric_pct"] = numeric_pct
    df["semantic_score"] = semantic_group_score

    # Глобальний percentile семантики [0, 1]: більше = семантично рідкісніший серед усіх тендерів
    df["semantic_pct"] = df["semantic_score"].rank(pct=True, method="average")

    # Інтерпретований blend у [0, 1]: числова частина як percentile у межах CPV2 (numeric_pct) +
    # семантика як semantic_pct + штраф за кластерний викид HDBSCAN
    threshold = 0.99  # верхній 1%
    df["semantic_outlier"] = (df["semantic_pct"] >= threshold).astype(int)
    so = df["semantic_outlier"].astype(float)
    df["risk_blend"] = (
        RISK_BLEND_WEIGHTS["semantic_outlier"] * so
        + RISK_BLEND_WEIGHTS["semantic_pct"] * df["semantic_pct"]
        + RISK_BLEND_WEIGHTS["numeric_pct"] * df["numeric_pct"]
    )

    df["risk_score"] = (
        2.0 * df["semantic_outlier"].astype(int) +
        1.5 * df["semantic_score"] +
        1.0 * df["numeric_score"]
    )

    df["risk_rank"] = df["risk_score"].rank(ascending=False, method="min")
    df["global_risk_rank"] = df["risk_blend"].rank(ascending=False, method="min")

    return df.sort_values("risk_score", ascending=False)


if __name__ == "__main__":
    result = detect_tender_anomalies()
    result.to_csv("data/tender_anomaly_results.csv", index=False)
