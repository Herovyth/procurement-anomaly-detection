from __future__ import annotations
"""
tender_anomaly.py

Що робить:
- Обчислює аномальність тендерів, комбінуючи:
  - numeric ризик (Isolation Forest + статистичні сигнали в межах CPV),
  - semantic ризик (SBERT-ембеддинги + подібність у групах).

Коли використовується:
- Після підготовки `tender_level_features.csv`.

Навіщо:
- Відповідає дипломній ідеї: ловити як фінансово-структурні, так і
  текстово-семантичні відхилення в умовах закупівель.
"""

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

CACHE_DIR = "../../data/cache"
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
]

SMALL_CPV_GROUP_FALLBACK_K = 100  # fallback to pooled (global) normalization for small CPV groups
CPV2_MIN_TENDERS_FOR_MODEL = 100  # min tenders in CPV2 group to train a dedicated IF; else use fallback IF
SEMANTIC_GROUP_MIN_TENDERS = 100  # min tenders in group to compute semantic-outlier score


NUMERIC_FEATURES = [
    "log_value_amount",
    "log_winner_price",
    "log_price_per_unit",
    "log_value_per_day",
    "numberOfBids",
    "numberOfAdmitted",
    "log_min_bid",
    "winner_minus_min",
    "tender_duration_days",
    "completion_days",
    "complaints",
    "tender_changes",
    "contract_changes",
    "cpv_deviation",
    "description_missing",
    "winner_price_missing",
    "total_missing_count",
    "bids_cpv_zscore",
    "admitted_cpv_zscore",
    "admitted_ratio",
    "price_drop_ratio",
]


TEXT_MODEL = "paraphrase-multilingual-mpnet-base-v2"
MODEL_CACHE_TAG = TEXT_MODEL.replace("/", "_").replace("-", "_")

RISK_BLEND_WEIGHTS = {
    "semantic_outlier": 2.0 / 4.5,
    "semantic_pct": 1.5 / 4.5,
    "numeric_pct": 1.0 / 4.5,
}

MODEL = None


def compute_embeddings_real(texts_real, suffix="real", cache_dir: Optional[str] = None):
    """Кодує тексти реальних тендерів у SBERT-вектори з кешуванням."""
    cdir = cache_dir if cache_dir is not None else CACHE_DIR
    os.makedirs(cdir, exist_ok=True)
    cache_path = f"{cdir}/embeddings_{MODEL_CACHE_TAG}_{suffix}.npy"

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


def compute_embeddings_synthetic(texts_synth, suffix="synth", cache_dir: Optional[str] = None):
    """Кодує synthetic-тексти в той самий семантичний простір."""
    if not texts_synth:
        return None

    cdir = cache_dir if cache_dir is not None else CACHE_DIR
    os.makedirs(cdir, exist_ok=True)
    cache_path = f"{cdir}/embeddings_{MODEL_CACHE_TAG}_{suffix}.npy"

    global MODEL
    if MODEL is None:
        print("Завантажуємо NLP модель...", flush=True)
        MODEL = SentenceTransformer(TEXT_MODEL)

    print(f"Рахуємо embeddings для {suffix}...", flush=True)
    emb = MODEL.encode(texts_synth, show_progress_bar=True, normalize_embeddings=True)
    emb = emb.astype(np.float32)
    np.save(cache_path, emb)
    return emb


def reduce_embeddings_with_real_and_synth(emb_real, emb_synth=None, cache_dir: Optional[str] = None):
    """Зменшує розмірність ембеддингів через UMAP і повторно використовує кеш."""
    cdir = cache_dir if cache_dir is not None else CACHE_DIR
    os.makedirs(cdir, exist_ok=True)
    model_path = f"{cdir}/umap_model_{MODEL_CACHE_TAG}.joblib"
    reduced_real_path = f"{cdir}/umap_reduced_real_{MODEL_CACHE_TAG}.npy"
    reduced_synth_path = f"{cdir}/umap_reduced_synth_{MODEL_CACHE_TAG}.npy"

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

    reduced_synth = None
    if emb_synth is not None:
        print("Проєктуємо synthetic в той же простір UMAP...", flush=True)
        reduced_synth = reducer.transform(emb_synth)
        np.save(reduced_synth_path, reduced_synth)

    if reduced_synth is not None:
        return np.vstack([reduced_real, reduced_synth])
    return reduced_real


def load_tender_data(path="../../data/raw/tender_level_features.csv"):
    return pd.read_csv(path)




def _to_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    ss = s.astype(str).str.strip().str.lower()
    return ss.isin({"true", "1", "t", "yes", "y"})


def _ensure_missing_columns(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c not in df.columns:
            df[c] = False


def _require_columns(df: pd.DataFrame, cols: list[str], context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{context}] Missing required columns: {missing}")


def _cpv_capped_iqr_zscore(
    df: pd.DataFrame,
    value_col: str,
    out_col: str,
    cpv_col: str = "cpv",
    k: int = SMALL_CPV_GROUP_FALLBACK_K,
) -> None:
    """
    Нормалізує ознаку в межах CPV через robust z-score (median/IQR).
    Для малих CPV-груп використовує глобальний fallback, щоб уникнути шуму.
    """
    if cpv_col not in df.columns:
        df[cpv_col] = None

    sizes = df.groupby(cpv_col).size()
    sizes_map = df[cpv_col].map(sizes).fillna(0)
    mask_small = sizes_map.lt(k)

    global_median = df[value_col].median()
    global_q25 = df[value_col].quantile(0.25)
    global_q75 = df[value_col].quantile(0.75)
    global_iqr = (global_q75 - global_q25) + 1e-9

    cpv_median = df.groupby(cpv_col)[value_col].median()
    cpv_q25 = df.groupby(cpv_col)[value_col].quantile(0.25)
    cpv_q75 = df.groupby(cpv_col)[value_col].quantile(0.75)
    cpv_iqr = (cpv_q75 - cpv_q25) + 1e-9

    median_per_row = df[cpv_col].map(cpv_median)
    iqr_per_row = df[cpv_col].map(cpv_iqr)

    z = (df[value_col] - median_per_row) / iqr_per_row

    if mask_small.any():
        z.loc[mask_small] = (df.loc[mask_small, value_col] - global_median) / global_iqr

    df[out_col] = z.replace([np.inf, -np.inf], 0).fillna(0.0)


def _extract_cpv2(cpv: object) -> str:
    if cpv is None or (isinstance(cpv, float) and np.isnan(cpv)):
        return "NA"
    s = str(cpv).strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) < 2:
        return "NA"
    return digits[:2]


def _rank_pct(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    s = pd.Series(values)
    return s.rank(pct=True, method="average").to_numpy()


def semantic_outliers_by_group(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    group_col: str,
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Виявляє семантичні аномалії всередині групи:
    низька середня cosine-подібність до k найближчих сусідів => підозра.
    """
    n = len(df)
    semantic_strength = np.zeros(n, dtype=float)
    semantic_outlier = np.zeros(n, dtype=int)

    max_exact_group = 2000

    for _, group in df.groupby(group_col):
        idx = group.index.to_numpy()
        m = len(idx)
        if m < 2:
            continue

        emb = embeddings[idx].astype(np.float32, copy=False)
        kk = min(k, m - 1)
        if kk <= 0:
            continue

        if m <= max_exact_group:
            sim = emb @ emb.T
            np.fill_diagonal(sim, -np.inf)
            topk = np.partition(sim, -kk, axis=1)[:, -kk:]
        else:
            subset_size = max_exact_group
            subset_idx = np.random.choice(m, size=subset_size, replace=False)
            emb_sub = emb[subset_idx]
            sim = emb @ emb_sub.T
            kk2 = min(kk, emb_sub.shape[0])
            topk = np.partition(sim, -kk2, axis=1)[:, -kk2:]

        mean_sim = topk.mean(axis=1)
        median = float(np.median(mean_sim))
        std = float(np.std(mean_sim))
        threshold = median - std

        mask = mean_sim < threshold
        semantic_outlier[idx] = mask.astype(int)
        semantic_strength[idx] = np.where(mask, threshold - mean_sim, 0.0)

    strength_std = float(np.std(semantic_strength))
    if strength_std < 1e-12:
        semantic_z = np.zeros_like(semantic_strength)
    else:
        semantic_z = (semantic_strength - float(np.mean(semantic_strength))) / (strength_std + 1e-9)

    return semantic_z, semantic_outlier


def numeric_anomaly_score_by_group(
    X_scaled: np.ndarray,
    groups: pd.Series,
    min_group_size: int = CPV2_MIN_TENDERS_FOR_MODEL,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Рахує numeric аномальність Isolation Forest-ом по групах CPV2.
    Для замалих груп використовує загальну fallback-модель.
    """
    n = X_scaled.shape[0]
    raw = np.zeros(n, dtype=float)

    fallback_if = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=42,
    )
    fallback_if.fit(X_scaled)
    fallback_raw = -fallback_if.score_samples(X_scaled)

    sizes = groups.value_counts(dropna=False)

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

    z = np.zeros(n, dtype=float)
    pct = np.zeros(n, dtype=float)
    for g, idx in groups.groupby(groups).groups.items():
        idx = np.asarray(list(idx), dtype=int)
        z[idx] = zscore(raw[idx]) if len(idx) > 1 else 0.0
        pct[idx] = _rank_pct(raw[idx])

    return z, pct


def prepare_numeric_features(df):
    """Імпутація + robust масштабування для числового блоку тендерних ознак."""
    X = df[NUMERIC_FEATURES]
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    X_imp = imputer.fit_transform(X)
    return scaler.fit_transform(X_imp)


def detect_tender_anomalies(
    path: Optional[str] = "../../data/raw/tender_level_features.csv",
    df: Optional[pd.DataFrame] = None,
    smoke_sample_size: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Основний pipeline оцінки ризику тендерів.

    Кроки:
    1) feature engineering (missing signals, CPV-нормалізації, ratio),
    2) semantic блок (SBERT + групові outlier-оцінки),
    3) numeric блок (IF + CPV-aware скоринг),
    4) blended risk score для підсумкового ранжування.
    """

    cdir = cache_dir if cache_dir is not None else CACHE_DIR
    os.makedirs(cdir, exist_ok=True)

    if df is None:
        df = load_tender_data(path)
    else:
        df = df.copy()

    df = df.reset_index(drop=True)

    if smoke_sample_size is not None:
        df = df.head(smoke_sample_size)

    _require_columns(
        df,
        cols=[
            "tender_id", "cpv", "status", "title", "description",
            "log_value_amount", "log_winner_price", "log_price_per_unit",
            "log_value_per_day", "numberOfBids", "numberOfAdmitted",
            "log_min_bid", "winner_minus_min", "tender_duration_days",
            "completion_days", "complaints", "tender_changes",
            "contract_changes", "median_cpv",
        ],
        context="detect_tender_anomalies",
    )

    is_synth = df["tender_id"].astype(str).str.startswith("SYNTH_")

    df["text"] = (
        df["title"].fillna("") + ". " +
        df["description"].fillna("")
    )

    df_real = df[~is_synth].copy()
    df_synth = df[is_synth].copy()

    # ── Feature engineering ───────────────────────────────────────────────────
    _ensure_missing_columns(df, HIGH_SIGNAL_MISSING + LOW_SIGNAL_MISSING)
    for c in HIGH_SIGNAL_MISSING + LOW_SIGNAL_MISSING:
        df[c] = _to_bool_series(df[c])

    df["total_missing_count"] = df[LOW_SIGNAL_MISSING].astype(int).sum(axis=1)

    # 2) CPV-normalized competition signals with small-CPV fallback
    _cpv_capped_iqr_zscore(df, value_col="numberOfBids", out_col="bids_cpv_zscore")
    _cpv_capped_iqr_zscore(df, value_col="numberOfAdmitted", out_col="admitted_cpv_zscore")

    # 3) admitted ratio (guard division by 0)
    bids = df["numberOfBids"].fillna(0)
    admitted = df["numberOfAdmitted"].fillna(0)
    df["admitted_ratio"] = np.where(bids > 0, admitted / bids, 0.0)

    # Відновлюємо winner_price із log_winner_price (в extraction зберігається лише log версія)
    df["winner_price"] = np.expm1(df["log_winner_price"])

    expected = df["median_cpv"].replace([np.inf, -np.inf], np.nan)
    winner = df["winner_price"].replace([np.inf, -np.inf], np.nan)
    denom = expected.replace(0, np.nan) + 1e-9
    df["price_drop_ratio"] = ((expected - winner) / denom).replace([np.inf, -np.inf], 0).fillna(0.0)

    winner_price_missing_mask = df["winner_price_missing"].astype(bool)
    df.loc[winner_price_missing_mask, "price_drop_ratio"] = 0.0
    df.loc[winner_price_missing_mask, "log_winner_price"] = 0.0

    # Використовуємо winner_price CPV z-score як єдину версію cpv_deviation
    winner_price_num = pd.to_numeric(df["winner_price"], errors="coerce")
    cpv_means = winner_price_num.groupby(df["cpv"]).transform("mean")
    cpv_stds = winner_price_num.groupby(df["cpv"]).transform("std")
    denom = cpv_stds.replace(0, np.nan) + 1e-9
    df["cpv_deviation"] = (winner_price_num - cpv_means) / denom
    df["cpv_deviation"] = df["cpv_deviation"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df.loc[winner_price_missing_mask, "cpv_deviation"] = 0.0

    df["cpv2"] = df["cpv"].apply(_extract_cpv2)

    # ── Embeddings ────────────────────────────────────────────────────────────
    emb_real = compute_embeddings_real(df_real["text"].tolist(), cache_dir=cdir)
    emb_synth = compute_embeddings_synthetic(df_synth["text"].tolist(), cache_dir=cdir)

    # Align embeddings (real first, then synth) back to the original df row order.
    real_idx = df_real.index.to_numpy()
    synth_idx = df_synth.index.to_numpy()

    emb_dim = emb_real.shape[1]
    embeddings = np.empty((len(df), emb_dim), dtype=np.float32)
    embeddings[real_idx] = emb_real
    if emb_synth is not None:
        embeddings[synth_idx] = emb_synth

    # ── UMAP + HDBSCAN ────────────────────────────────────────────────────────
    emb_reduced_stacked = reduce_embeddings_with_real_and_synth(emb_real, emb_synth, cache_dir=cdir)
    reduced_dim = emb_reduced_stacked.shape[1]
    emb_reduced = np.empty((len(df), reduced_dim), dtype=np.float32)
    n_real = len(df_real)
    emb_reduced[real_idx] = emb_reduced_stacked[:n_real]
    if emb_synth is not None:
        emb_reduced[synth_idx] = emb_reduced_stacked[n_real:]

    hdbscan_path = f"{cdir}/hdbscan_labels_{MODEL_CACHE_TAG}.npy"
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
        joblib.dump(clusterer, f"{cdir}/hdbscan_model_{MODEL_CACHE_TAG}.joblib")

    df["semantic_cluster"] = clusters

    # ── Semantic outlier score ────────────────────────────────────────────────
    semantic_score, semantic_outlier = semantic_outliers_by_group(
        df, embeddings, group_col="cpv2", k=5
    )
    df["semantic_score"] = semantic_score
    df["semantic_outlier"] = semantic_outlier
    df["semantic_pct"] = df["semantic_score"].rank(pct=True, method="average")

    # ── Numeric score: IsolationForest ────────────────────────────────────────
    if_scores_path = f"{cdir}/if_scores_cpv_z_v2_{MODEL_CACHE_TAG}.npy"
    if os.path.exists(if_scores_path):
        print("Завантажуємо IsolationForest scores з кешу...", flush=True)
        scores = np.load(if_scores_path, allow_pickle=True).item()
        numeric_z = scores["z"]
        numeric_pct = scores["pct"]
        if len(numeric_z) != len(df):
            print("[WARN] Cached numeric scores size mismatch! Recomputing...")
            X_num_scaled = prepare_numeric_features(df)
            numeric_z, numeric_pct = numeric_anomaly_score_by_group(
                X_num_scaled, groups=df["cpv2"]
            )
            np.save(if_scores_path, {"z": numeric_z, "pct": numeric_pct})
    else:
        print("Тренуємо IsolationForest на всіх тендерах...", flush=True)
        X_num_scaled = prepare_numeric_features(df)
        numeric_z, numeric_pct = numeric_anomaly_score_by_group(
            X_num_scaled, groups=df["cpv2"]
        )
        np.save(if_scores_path, {"z": numeric_z, "pct": numeric_pct})

    # ── Numeric score adjustment (cpv_deviation boost) ────────────────────────
    df["numeric_screen"]        = (df["cpv_deviation"].abs() > 2).astype(int)
    df["numeric_score_adj_raw"] = (
        numeric_z
        + df["numeric_screen"].astype(float) * df["cpv_deviation"].abs()
    )

    numeric_score_adj = np.zeros(len(df), dtype=float)
    numeric_pct_adj = np.zeros(len(df), dtype=float)
    for _, idx in df.groupby("cpv2").groups.items():
        idx = np.asarray(list(idx), dtype=int)
        vals = df.loc[idx, "numeric_score_adj_raw"].to_numpy(dtype=float)
        if len(vals) > 1 and np.nanstd(vals) > 0:
            z = (vals - np.nanmean(vals)) / (np.nanstd(vals) + 1e-9)
        else:
            z = np.zeros_like(vals, dtype=float)
        numeric_score_adj[idx] = z
        numeric_pct_adj[idx] = _rank_pct(vals)

    df["numeric_score"] = numeric_score_adj
    df["numeric_pct"] = numeric_pct_adj

    so = df["semantic_outlier"].astype(float)
    df["risk_score"] = (
        RISK_BLEND_WEIGHTS["semantic_outlier"] * so
        + RISK_BLEND_WEIGHTS["semantic_pct"] * df["semantic_pct"]
        + RISK_BLEND_WEIGHTS["numeric_pct"] * df["numeric_pct"]
    )
    df["risk_rank"] = df["risk_score"].rank(ascending=False, method="min")

    return df.sort_values("risk_score", ascending=False)


if __name__ == "__main__":
    smoke_flag = os.environ.get("SMOKE_TENDER_ANOMALIES", "0") == "1"
    smoke_n    = int(os.environ.get("SMOKE_TENDER_ANOMALIES_N", "200"))
    if smoke_flag:
        result = detect_tender_anomalies(smoke_sample_size=smoke_n)
    else:
        result = detect_tender_anomalies()
    result.to_csv("../../data/prepared/tender_anomaly_results.csv", index=False)