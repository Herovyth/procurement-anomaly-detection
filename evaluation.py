"""
Оцінка якості детекції аномалій.

Тендери: повний пайплайн з tender_anomaly.py (CPV2 IF, semantic, risk_blend),
         Recall@K за інтерпретованим risk_blend та за попереднім risk_score.

Постачальники: попередня схема (глобальний IF на числових колонках).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from tender_anomaly import detect_tender_anomalies

# ----------------------------
# Параметри
# ----------------------------
K = 50  # top-K для Recall@K

SYNTH_TENDER_PREFIX = "SYNTH_"


# ----------------------------
# Тендери: злиття real + synthetic з узгодженими колонками
# ----------------------------
def merge_tender_eval_set(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Повертає merged df та булеву серію міток (True = синтетичний аномальний)."""
    synth = synthetic_df.copy()
    for col in real_df.columns:
        if col not in synth.columns:
            synth[col] = np.nan
    synth = synth[real_df.columns]
    merged = pd.concat([real_df, synth], ignore_index=True)
    labels = merged["tender_id"].astype(str).str.startswith(SYNTH_TENDER_PREFIX)
    return merged, labels


def recall_at_k_unified(
    scores: np.ndarray,
    labels: np.ndarray,
    K: int,
    higher_is_more_anomalous: bool = True,
) -> float:
    """
    labels[i]==1 — позитив (синтетична аномалія).
    Recall@K = частка позитивів серед top-K за score.
    """
    n_pos = int(labels.sum())
    if n_pos == 0:
        return 0.0
    order = np.argsort(scores)
    if higher_is_more_anomalous:
        order = order[::-1]
    topk = order[: min(K, len(scores))]
    hits = labels[topk].sum()
    return hits / n_pos


# ----------------------------
# Завантаження даних
# ----------------------------
tender_features = pd.read_csv("data/tender_level_features.csv")
supplier_features = pd.read_csv("data/supplier_level_features.csv")

supplier_numeric_cols = [
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
]


def prepare_data(real_df, synthetic_df, numeric_cols):
    X_real = real_df[numeric_cols].fillna(0)
    X_synth = synthetic_df[numeric_cols].fillna(0)
    scaler = StandardScaler()
    X_real_scaled = scaler.fit_transform(X_real)
    X_synth_scaled = scaler.transform(X_synth)
    return X_real_scaled, X_synth_scaled, real_df, synthetic_df


def run_iforest(X_real, X_synth):
    model = IsolationForest(n_estimators=200, contamination="auto", random_state=42)
    model.fit(X_real)
    score_real = -model.score_samples(X_real)
    score_synth = -model.score_samples(X_synth)
    return score_real, score_synth


def recall_at_k(score_real, score_synth, K_val):
    all_scores = np.concatenate([score_real, score_synth])
    labels = np.array([0] * len(score_real) + [1] * len(score_synth))
    return recall_at_k_unified(all_scores, labels, K_val, higher_is_more_anomalous=True)


# ----------------------------
# Тендери: detect_tender_anomalies(full pipeline)
# ----------------------------
tender_synth_path = "data/synthetic_tender_anomalies.csv"
scored_tenders_for_plot: pd.DataFrame | None = None
if os.path.isfile(tender_synth_path):
    synthetic_tender = pd.read_csv(tender_synth_path)
    tender_merged, tender_labels = merge_tender_eval_set(tender_features, synthetic_tender)
    print(
        f"[Tender eval] Реальних: {len(tender_features)}, синтетики: {len(synthetic_tender)}, "
        f"разом: {len(tender_merged)}, позитивів: {tender_labels.sum()}"
    )
    print("[Tender eval] Запуск повного tender_anomaly pipeline (ембединги + IF + semantic)...")

    scored_tenders = detect_tender_anomalies(df=tender_merged)
    scored_tenders_for_plot = scored_tenders
    # IMPORTANT:
    # detect_tender_anomalies() повертає df, відсортований за risk_score (descending).
    # Тому labels з tender_merged можуть не відповідати рядкам scored_tenders.
    # Оцінюємо labels в тій самій таблиці/порядку, що й scores.
    labels_arr = (
        scored_tenders["tender_id"]
        .astype(str)
        .str.startswith(SYNTH_TENDER_PREFIX)
        .to_numpy(dtype=int)
    )

    rec_blend = recall_at_k_unified(
        scored_tenders["risk_blend"].to_numpy(),
        labels_arr,
        K,
        higher_is_more_anomalous=True,
    )
    rec_risk = recall_at_k_unified(
        scored_tenders["risk_score"].to_numpy(),
        labels_arr,
        K,
        higher_is_more_anomalous=True,
    )
    rec_numeric_only = recall_at_k_unified(
        scored_tenders["numeric_pct"].to_numpy(),
        labels_arr,
        K,
        higher_is_more_anomalous=True,
    )

    print(f"Recall@{K} Tender (risk_blend, інтерпретований): {rec_blend:.4f}")
    print(f"Recall@{K} Tender (risk_score, z-score blend):    {rec_risk:.4f}")
    print(f"Recall@{K} Tender (numeric_pct only):             {rec_numeric_only:.4f}")
else:
    print(f"[Tender eval] Пропущено: немає файлу {tender_synth_path}")

# ----------------------------
# Постачальники (як раніше)
# ----------------------------
supplier_synth_path = "data/synthetic_supplier_anomalies.csv"
if os.path.isfile(supplier_synth_path):
    synthetic_supplier = pd.read_csv(supplier_synth_path)
    supplier_real, supplier_synth, supplier_df_real, supplier_df_synth = prepare_data(
        supplier_features,
        synthetic_supplier,
        supplier_numeric_cols,
    )
    supplier_score_real, supplier_score_synth = run_iforest(supplier_real, supplier_synth)
    recall_supplier = recall_at_k(supplier_score_real, supplier_score_synth, K)
    n_pos_supplier = len(supplier_score_synth)
    print(
        f"Recall@{K} Supplier (legacy global IF): {recall_supplier:.4f} "
        f"(positives={n_pos_supplier})"
    )
else:
    print(f"[Supplier eval] Пропущено: немає файлу {supplier_synth_path}")

# ----------------------------
# Візуалізація (опційно; без інтерактивного display)
# ----------------------------
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if (
        scored_tenders_for_plot is not None
        and os.path.isfile(tender_synth_path)
        and os.path.isfile(supplier_synth_path)
    ):
        scored_tenders = scored_tenders_for_plot
        lbl = scored_tenders["tender_id"].astype(str).str.startswith(SYNTH_TENDER_PREFIX)
        plt.figure(figsize=(8, 4))
        plt.boxplot(
            [
                scored_tenders.loc[~lbl, "risk_blend"],
                scored_tenders.loc[lbl, "risk_blend"],
            ],
            tick_labels=["Real", "Synthetic"],
        )
        plt.title("Tender risk_blend (full pipeline)")
        plt.ylabel("risk_blend [0,1]")
        plt.tight_layout()
        plt.savefig("evaluation_tender_risk_blend.png", dpi=150)
        plt.close()

        synthetic_supplier = pd.read_csv(supplier_synth_path)
        supplier_real, supplier_synth, _, _ = prepare_data(
            supplier_features,
            synthetic_supplier,
            supplier_numeric_cols,
        )
        supplier_score_real, supplier_score_synth = run_iforest(supplier_real, supplier_synth)
        plt.figure(figsize=(8, 4))
        plt.boxplot([supplier_score_real, supplier_score_synth], tick_labels=["Real", "Synthetic"])
        plt.title("Supplier Anomaly Scores (legacy)")
        plt.ylabel("IF score")
        plt.tight_layout()
        plt.savefig("evaluation_supplier_scores.png", dpi=150)
        plt.close()
        print("[Plots] Збережено evaluation_tender_risk_blend.png, evaluation_supplier_scores.png")
except Exception as e:
    print(f"[Plots] Пропущено: {e}")
