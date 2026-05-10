"""
Оцінювання моделей (методика з дипломної записки).

Оскільки задача unsupervised (немає істинних міток аномалій), оцінювання базується на:
- метриках якості кластеризації (Silhouette, Davies–Bouldin) там, де є кластери,
- аналізі розподілу anomaly score (форма розподілу + percentile-поріг),
- візуальній валідації через PCA/UMAP у 2D.

Скрипт генерує метрики та зберігає графіки у PNG.

Коли запускати:
- Після формування `supplier_anomaly_results.csv` / `tender_anomaly_results.csv`,
  щоб перевірити якість кластеризації та форму розподілу risk score.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
import umap

# Allow running this file directly via `python scr/test/evaluation.py`.
SCR_ROOT = Path(__file__).resolve().parents[1]
if str(SCR_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(SCR_ROOT.parent))

from scr.train.supplier_anomaly import SUPPLIER_FEATURES, prepare_features, detect_supplier_anomalies
from scr.train.tender_anomaly import (
    NUMERIC_FEATURES,
    prepare_numeric_features,
    detect_tender_anomalies,
    compute_embeddings_real,
    compute_embeddings_synthetic,
    reduce_embeddings_with_real_and_synth,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "../../results/evaluation_results"

# Percentile threshold for anomaly cut-off (percentile-based in unsupervised setting)
ANOMALY_PCTL = 0.99


def _safe_metric(fn, X: np.ndarray, labels: np.ndarray) -> float | None:
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return None
    try:
        return float(fn(X, labels))
    except Exception:
        return None


def _percentile_threshold(values: np.ndarray, p: float) -> float:
    v = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(v) == 0:
        return float("nan")
    return float(np.quantile(v, p))


def _mode_or_first(values: pd.Series) -> int | float | str | None:
    mode = values.mode(dropna=False)
    if not mode.empty:
        return mode.iloc[0]
    if len(values) == 0:
        return None
    return values.iloc[0]


def _plot_hist(values: np.ndarray, title: str, path: str, threshold: float | None = None) -> None:
    v = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(v) == 0:
        return
    plt.figure(figsize=(8, 4))
    plt.hist(v, bins=60, alpha=0.85)
    if threshold is not None and np.isfinite(threshold):
        plt.axvline(threshold, linestyle="--", linewidth=2)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_2d_scatter(points: np.ndarray, is_anom: np.ndarray, title: str, path: str) -> None:
    if len(points) == 0:
        return
    plt.figure(figsize=(7, 6))
    mask = is_anom.astype(bool)
    plt.scatter(points[~mask, 0], points[~mask, 1], s=6, alpha=0.35, label="normal")
    plt.scatter(points[mask, 0], points[mask, 1], s=10, alpha=0.85, label="anomaly")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def evaluate_suppliers(path: str = "../../data/raw/supplier_level_features.csv") -> None:
    print("[Supplier] Scoring anomalies (K-Means + IF + LOF ensemble)...", flush=True)
    scored = detect_supplier_anomalies(path=path)
    scored.to_csv("../../data/raw/supplier_anomaly_results.csv", index=False)

    # rebuild feature matrix for metrics/visualizations
    df = pd.read_csv(path)
    X = prepare_features(df)
    cluster_by_id = scored.groupby("supplier_id", dropna=False)["cluster"].agg(_mode_or_first)
    labels = df["supplier_id"].map(cluster_by_id).to_numpy()

    sil = _safe_metric(lambda a, b: silhouette_score(a, b, metric="euclidean"), X, labels)
    db = _safe_metric(davies_bouldin_score, X, labels)
    print(f"[Supplier] silhouette={sil if sil is not None else 'NA'}  davies_bouldin={db if db is not None else 'NA'}")

    # distribution + percentile threshold
    thr = _percentile_threshold(scored["risk_score"].to_numpy(), ANOMALY_PCTL)
    scored["is_anomaly_pctl"] = (scored["risk_score"] >= thr).astype(int)
    print(f"[Supplier] risk_score threshold (p={ANOMALY_PCTL:.2f}): {thr:.4f}; anomalies={scored['is_anomaly_pctl'].sum()}")
    _plot_hist(
        scored["risk_score"].to_numpy(),
        title=f"Supplier risk_score distribution (p={ANOMALY_PCTL:.2f} threshold)",
        path=f"{OUT_DIR}/eval_supplier_risk_score_hist.png",
        threshold=thr,
    )

    # PCA / UMAP for visual validation
    is_anom_map = scored.groupby("supplier_id", dropna=False)["is_anomaly_pctl"].max()
    is_anom_by_id = df["supplier_id"].map(is_anom_map).fillna(0).to_numpy(dtype=int)
    pca2 = PCA(n_components=2, random_state=42).fit_transform(X)
    _plot_2d_scatter(
        pca2,
        is_anom=is_anom_by_id,
        title="Suppliers PCA(2) — anomalies highlighted",
        path=f"{OUT_DIR}/eval_supplier_pca2.png",
    )
    um2 = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.1, metric="euclidean", random_state=42).fit_transform(X)
    _plot_2d_scatter(
        um2,
        is_anom=is_anom_by_id,
        title="Suppliers UMAP(2) — anomalies highlighted",
        path=f"{OUT_DIR}/eval_supplier_umap2.png",
    )


def evaluate_tenders(path: str = "../../data/raw/tender_level_features.csv") -> None:
    print("[Tender] Scoring anomalies (numeric + semantic)...", flush=True)
    df_raw = pd.read_csv(path)
    scored = detect_tender_anomalies(df=df_raw)
    scored.to_csv("../../data/prepared/tender_anomaly_results.csv", index=False)

    risk_col = "risk_score" if "risk_score" in scored.columns else "risk_blend"

    thr = _percentile_threshold(scored[risk_col].to_numpy(), ANOMALY_PCTL)
    scored["is_anomaly_pctl"] = (scored[risk_col] >= thr).astype(int)
    print(f"[Tender] {risk_col} threshold (p={ANOMALY_PCTL:.2f}): {thr:.4f}; anomalies={scored['is_anomaly_pctl'].sum()}")

    _plot_hist(
        scored[risk_col].to_numpy(),
        title=f"Tender {risk_col} distribution (p={ANOMALY_PCTL:.2f} threshold)",
        path=f"{OUT_DIR}/eval_tender_risk_blend_hist.png",
        threshold=thr,
    )
    _plot_hist(
        scored["numeric_score"].to_numpy(),
        title="Tender numeric_score distribution",
        path=f"{OUT_DIR}/eval_tender_numeric_score_hist.png",
        threshold=None,
    )
    _plot_hist(
        scored["semantic_score"].to_numpy(),
        title="Tender semantic_score distribution",
        path=f"{OUT_DIR}/eval_tender_semantic_score_hist.png",
        threshold=None,
    )

    # clustering quality in semantic space (UMAP-reduced SBERT embeddings)
    if {"tender_id", "title", "description", "semantic_cluster"}.issubset(set(df_raw.columns) | set(scored.columns)):
        df_sem = df_raw.copy()
        df_sem["text"] = df_sem["title"].fillna("") + ". " + df_sem["description"].fillna("")

        is_synth = df_sem["tender_id"].astype(str).str.startswith("SYNTH_")
        df_real = df_sem.loc[~is_synth]
        df_synth = df_sem.loc[is_synth]

        emb_real = compute_embeddings_real(df_real["text"].tolist())
        emb_synth = compute_embeddings_synthetic(df_synth["text"].tolist())
        emb_reduced_stacked = reduce_embeddings_with_real_and_synth(emb_real, emb_synth)

        sem_dim = emb_reduced_stacked.shape[1]
        X_sem = np.empty((len(df_sem), sem_dim), dtype=np.float32)
        real_idx = df_real.index.to_numpy()
        synth_idx = df_synth.index.to_numpy()
        n_real = len(df_real)
        X_sem[real_idx] = emb_reduced_stacked[:n_real]
        if emb_synth is not None:
            X_sem[synth_idx] = emb_reduced_stacked[n_real:]

        if "tender_id" in scored.columns and "tender_id" in df_sem.columns:
            cluster_map = scored.groupby("tender_id", dropna=False)["semantic_cluster"].agg(_mode_or_first)
            tender_labels = pd.to_numeric(df_sem["tender_id"].map(cluster_map), errors="coerce").fillna(-1).to_numpy(dtype=int)
        else:
            tender_labels = pd.to_numeric(scored.get("semantic_cluster"), errors="coerce").reindex(range(len(df_sem))).fillna(-1).to_numpy(dtype=int)

        metric_mask = tender_labels != -1
        if metric_mask.sum() >= 2 and len(np.unique(tender_labels[metric_mask])) >= 2:
            sil = _safe_metric(
                lambda a, b: silhouette_score(a, b, metric="euclidean"),
                X_sem[metric_mask],
                tender_labels[metric_mask],
            )
            db = _safe_metric(
                davies_bouldin_score,
                X_sem[metric_mask],
                tender_labels[metric_mask],
            )
            print(f"[Tender] semantic silhouette={sil if sil is not None else 'NA'}  davies_bouldin={db if db is not None else 'NA'}")
        else:
            print("[Tender] semantic silhouette=NA  davies_bouldin=NA (insufficient non-noise clusters)")
    else:
        print("[Tender] semantic silhouette=NA  davies_bouldin=NA (missing required semantic columns)")

    avail = [c for c in NUMERIC_FEATURES if c in scored.columns]
    if len(avail) == len(NUMERIC_FEATURES):
        X_scaled = prepare_numeric_features(scored)

        if "tender_id" in scored.columns and "tender_id" in df_raw.columns:
            is_anom_map = scored.groupby("tender_id", dropna=False)["is_anomaly_pctl"].max()
            is_anom = df_raw["tender_id"].map(is_anom_map).fillna(0).to_numpy(dtype=int)
        else:
            # fallback: scored і df_raw мають однакову довжину та порядок
            is_anom = scored["is_anomaly_pctl"].reindex(range(len(df_raw))).fillna(0).to_numpy(dtype=int)

        pca2 = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
        _plot_2d_scatter(
            pca2,
            is_anom=is_anom,
            title="Tenders PCA(2) over numeric features — anomalies highlighted",
            path=f"{OUT_DIR}/eval_tender_pca2_numeric.png",
        )
        um2 = umap.UMAP(
            n_neighbors=15, n_components=2, min_dist=0.1,
            metric="euclidean", random_state=42
        ).fit_transform(X_scaled)
        _plot_2d_scatter(
            um2,
            is_anom=is_anom,
            title="Tenders UMAP(2) over numeric features — anomalies highlighted",
            path=f"{OUT_DIR}/eval_tender_umap2_numeric.png",
        )
    else:
        print(f"Skip PCA/UMAP numeric viz: missing {len(NUMERIC_FEATURES) - len(avail)} numeric feature columns.")


def main() -> None:
    evaluate_suppliers()
    evaluate_tenders()
    print(
        "Saved plots: "
        "eval_supplier_risk_score_hist.png, eval_supplier_pca2.png, eval_supplier_umap2.png, "
        "eval_tender_risk_blend_hist.png, eval_tender_numeric_score_hist.png, eval_tender_semantic_score_hist.png "
        "(and PCA/UMAP numeric plots if possible)."
    )


if __name__ == "__main__":
    main()
