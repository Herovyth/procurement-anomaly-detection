from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import davies_bouldin_score, silhouette_samples, silhouette_score
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM
from sklearn.impute import SimpleImputer


RANDOM_STATE = 42
SAMPLE_SIZES = [500, 1000, 2000, 4000, 7000, 11000]
TARGET_ANOMALY_RATE = 5.0

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
]

METHOD_ORDER = [
    "ensemble_kmeans_if_lof",
    "kmeans",
    "iforest",
    "lof",
    "ocsvm",
]

METHOD_LABELS = {
    "ensemble_kmeans_if_lof": "Ensemble (K-Means + IF + LOF)",
    "kmeans": "K-Means",
    "iforest": "Isolation Forest",
    "lof": "LOF",
    "ocsvm": "One-Class SVM",
}

METHOD_COLORS = {
    "ensemble_kmeans_if_lof": "#1B4F8A",
    "kmeans": "#2E86AB",
    "iforest": "#E84855",
    "lof": "#F4A261",
    "ocsvm": "#6B4D8A",
}


def load_data(csv_path: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Завантаження даних, імпутація та масштабування."""
    df = pd.read_csv(csv_path)
    missing = [c for c in SUPPLIER_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"У CSV відсутні потрібні колонки: {missing}")

    X_raw = df[SUPPLIER_FEATURES].copy()
    for col in SUPPLIER_FEATURES:
        X_raw[col] = pd.to_numeric(X_raw[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    X_imp = imputer.fit_transform(X_raw)
    X = scaler.fit_transform(X_imp)
    return df, X


def _kmeans_labels_scores(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """K-Means: аномалія, якщо silhouette < 0 (як у дипломному підході)."""
    n = len(X)
    if n < 3:
        return np.zeros(n, dtype=int), np.zeros(n, dtype=float)

    k_min = 2
    k_max = min(15, n - 1)
    if k_max < k_min:
        k_best = k_min
    else:
        score_idx = np.arange(n)
        if n > 2000:
            rng = np.random.default_rng(RANDOM_STATE)
            score_idx = rng.choice(n, size=2000, replace=False)
        X_score = X[score_idx]

        k_best = k_min
        best_s = -np.inf
        for k in range(k_min, k_max + 1):
            try:
                km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
                labels_tmp = km.fit_predict(X_score)
                s = silhouette_score(X_score, labels_tmp, metric="euclidean")
                if s > best_s:
                    best_s = s
                    k_best = k
            except Exception:
                continue

    km = KMeans(n_clusters=k_best, n_init=10, random_state=RANDOM_STATE)
    cluster_labels = km.fit_predict(X)
    sil = silhouette_samples(X, cluster_labels, metric="euclidean")
    labels = (sil < 0).astype(int)
    scores = -sil
    return labels, scores


def method_ensemble(X: np.ndarray, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Ансамбль 2 з 3: K-Means + Isolation Forest + LOF."""
    _ = df
    km_labels, km_scores = _kmeans_labels_scores(X)

    iforest = IsolationForest(n_estimators=500, contamination="auto", random_state=RANDOM_STATE)
    if_labels = (iforest.fit_predict(X) == -1).astype(int)
    if_scores = -iforest.score_samples(X)

    n = len(X)
    n_neighbors = min(20, max(2, n - 1))
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination="auto")
    lof_labels = (lof.fit_predict(X) == -1).astype(int)
    lof_scores = -lof.negative_outlier_factor_

    labels = ((km_labels + if_labels + lof_labels) >= 2).astype(int)
    scores = (km_scores + if_scores + lof_scores) / 3.0 + 0.25 * labels.astype(float)
    return labels, scores


def method_kmeans(X: np.ndarray, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    _ = df
    return _kmeans_labels_scores(X)


def method_iforest(X: np.ndarray, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    _ = df
    model = IsolationForest(n_estimators=500, contamination="auto", random_state=RANDOM_STATE)
    labels = (model.fit_predict(X) == -1).astype(int)
    scores = -model.score_samples(X)
    return labels, scores


def method_lof(X: np.ndarray, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    _ = df
    n = len(X)
    if n < 3:
        return np.zeros(n, dtype=int), np.zeros(n, dtype=float)
    n_neighbors = min(20, max(2, n - 1))
    model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination="auto")
    labels = (model.fit_predict(X) == -1).astype(int)
    scores = -model.negative_outlier_factor_
    return labels, scores


def method_ocsvm(X: np.ndarray, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    _ = df
    n = len(X)
    model = OneClassSVM(gamma="scale")
    if n > 3000:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(n, size=3000, replace=False)
        model.fit(X[idx])
    else:
        model.fit(X)
    labels = (model.predict(X) == -1).astype(int)
    scores = -model.decision_function(X)
    return labels, scores


def compute_metrics(X: np.ndarray, labels: np.ndarray, elapsed_sec: float) -> dict[str, float]:
    """Розрахунок Silhouette, DB-index, anomaly rate, time."""
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    if len(X) < 2 or len(uniq) < 2:
        sil = np.nan
        dbi = np.nan
    else:
        try:
            sil = float(silhouette_score(X, labels))
        except Exception:
            sil = np.nan
        try:
            dbi = float(davies_bouldin_score(X, labels))
        except Exception:
            dbi = np.nan

    anomaly_rate = float(np.mean(labels == 1) * 100.0) if len(labels) else np.nan
    return {
        "silhouette": sil,
        "davies_bouldin": dbi,
        "anomaly_rate_pct": anomaly_rate,
        "time_sec": float(elapsed_sec),
    }


def plot_stability(metrics_df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    for method in METHOD_ORDER:
        sub = metrics_df[metrics_df["method"] == method].sort_values("sample_size")
        lw = 3.0 if method == "ensemble_kmeans_if_lof" else 1.5
        ls = "-" if method == "ensemble_kmeans_if_lof" else "--"
        axes[0].plot(
            sub["sample_size"], sub["silhouette"],
            label=METHOD_LABELS[method], color=METHOD_COLORS[method],
            linewidth=lw, linestyle=ls, marker="o"
        )
        axes[1].plot(
            sub["sample_size"], sub["davies_bouldin"],
            label=METHOD_LABELS[method], color=METHOD_COLORS[method],
            linewidth=lw, linestyle=ls, marker="o"
        )

    axes[0].set_title("Silhouette Score")
    axes[0].set_ylabel("Score")
    axes[1].set_title("Davies-Bouldin Index")
    for ax in axes:
        ax.set_xlabel("Розмір вибірки")
        ax.set_xscale("log")
        ax.grid(alpha=0.25)
    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle("Стабільність методів для аномальних підрядників")
    fig.tight_layout()
    fig.savefig(out_dir / "supplier_stability_plot.png", dpi=180)
    plt.close(fig)


def plot_time(metrics_df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for method in METHOD_ORDER:
        sub = metrics_df[metrics_df["method"] == method].sort_values("sample_size")
        lw = 3.0 if method == "ensemble_kmeans_if_lof" else 1.5
        ls = "-" if method == "ensemble_kmeans_if_lof" else "--"
        ax.plot(
            sub["sample_size"], sub["time_sec"],
            label=METHOD_LABELS[method], color=METHOD_COLORS[method],
            linewidth=lw, linestyle=ls, marker="o"
        )
    ax.set_xscale("log")
    ax.set_xlabel("Розмір вибірки")
    ax.set_ylabel("Час, сек")
    ax.set_title("Час виконання методів (підрядники)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "supplier_time_plot.png", dpi=180)
    plt.close(fig)


def plot_anomaly_rate(metrics_df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for method in METHOD_ORDER:
        sub = metrics_df[metrics_df["method"] == method].sort_values("sample_size")
        lw = 3.0 if method == "ensemble_kmeans_if_lof" else 1.5
        ls = "-" if method == "ensemble_kmeans_if_lof" else "--"
        ax.plot(
            sub["sample_size"], sub["anomaly_rate_pct"],
            label=METHOD_LABELS[method], color=METHOD_COLORS[method],
            linewidth=lw, linestyle=ls, marker="o"
        )
    ax.axhline(TARGET_ANOMALY_RATE, color="black", linestyle="--", linewidth=1.2, label="Ціль 5%")
    ax.set_xscale("log")
    ax.set_xlabel("Розмір вибірки")
    ax.set_ylabel("Anomaly rate, %")
    ax.set_title("Частка аномалій (підрядники)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "supplier_anomaly_rate_plot.png", dpi=180)
    plt.close(fig)


def plot_pca_grid(
    X_full: np.ndarray,
    labels_by_method: dict[str, np.ndarray],
    metrics_full_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    pca2 = KMeans(n_clusters=2, random_state=RANDOM_STATE)  # заглушка для random_state-послідовності
    _ = pca2
    from sklearn.decomposition import PCA
    p2 = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_full)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes_flat = axes.flatten()
    method_axes = axes_flat[:5]
    info_ax = axes_flat[5]

    for ax, method in zip(method_axes, METHOD_ORDER):
        labels = labels_by_method.get(method)
        if labels is None or len(labels) != len(p2):
            ax.text(0.5, 0.5, "Немає даних", ha="center", va="center")
            ax.set_title(METHOD_LABELS[method])
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        mask = labels.astype(bool)
        ax.scatter(p2[~mask, 0], p2[~mask, 1], s=8, alpha=0.35, c="#95a5a6", label="normal")
        ax.scatter(p2[mask, 0], p2[mask, 1], s=10, alpha=0.9, c=METHOD_COLORS[method], label="anomaly")
        ax.set_title(METHOD_LABELS[method], color=METHOD_COLORS[method])
        ax.grid(alpha=0.2)
        ax.set_xticks([])
        ax.set_yticks([])

        if method == "ensemble_kmeans_if_lof":
            for spine in ax.spines.values():
                spine.set_edgecolor("#2ca02c")
                spine.set_linewidth(3)

    info_ax.axis("off")
    lines = ["Інформаційна панель (повна вибірка):", ""]
    for method in METHOD_ORDER:
        sub = metrics_full_df[metrics_full_df["method"] == method]
        if sub.empty:
            lines.append(f"- {METHOD_LABELS[method]}: немає даних")
            continue
        r = sub.iloc[0]
        lines.append(
            f"- {METHOD_LABELS[method]}: "
            f"sil={r['silhouette']:.3f} | db={r['davies_bouldin']:.3f} | "
            f"anom={r['anomaly_rate_pct']:.2f}% | t={r['time_sec']:.2f}s"
        )
    info_ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=10)

    fig.suptitle("PCA-візуалізація аномалій для підрядників", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "supplier_pca_visualization.png", dpi=180)
    plt.close(fig)


def _minmax_norm(values: np.ndarray, reverse: bool = False) -> np.ndarray:
    v = values.astype(float).copy()
    valid = np.isfinite(v)
    if not np.any(valid):
        return np.full_like(v, np.nan)
    vmin = np.nanmin(v[valid])
    vmax = np.nanmax(v[valid])
    if np.isclose(vmax, vmin):
        out = np.zeros_like(v, dtype=float)
        out[~valid] = np.nan
        return out
    out = (v - vmin) / (vmax - vmin)
    if reverse:
        out = 1.0 - out
    out[~valid] = np.nan
    return out


def plot_summary_heatmap(metrics_full_df: pd.DataFrame, out_dir: Path) -> None:
    metrics_cols = ["silhouette", "davies_bouldin", "anomaly_rate_pct", "time_sec"]
    table = metrics_full_df.set_index("method")[metrics_cols].reindex(METHOD_ORDER)
    raw = table.to_numpy(dtype=float)
    norm = np.column_stack(
        [
            _minmax_norm(raw[:, 0], reverse=False),
            _minmax_norm(raw[:, 1], reverse=True),
            _minmax_norm(raw[:, 2], reverse=False),
            _minmax_norm(raw[:, 3], reverse=True),
        ]
    )
    norm_plot = np.nan_to_num(norm, nan=0.0)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(norm_plot, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03, label="Нормалізований score")

    ax.set_xticks(np.arange(len(metrics_cols)))
    ax.set_xticklabels(["Silhouette", "DB-index", "Anomaly rate %", "Time sec"])
    ax.set_yticks(np.arange(len(METHOD_ORDER)))
    ax.set_yticklabels([METHOD_LABELS[m] for m in METHOD_ORDER])
    ax.set_title("Summary heatmap (підрядники, повна вибірка)")

    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            val = raw[i, j]
            txt = "NaN" if not np.isfinite(val) else f"{val:.4f}" if j < 2 else f"{val:.2f}"
            ax.text(j, i, txt, ha="center", va="center", color="black", fontsize=9)

    selected_idx = METHOD_ORDER.index("ensemble_kmeans_if_lof")
    rect = plt.Rectangle(
        (-0.5, selected_idx - 0.5), len(metrics_cols), 1.0,
        fill=False, edgecolor="#2ca02c", linewidth=3
    )
    ax.add_patch(rect)

    fig.tight_layout()
    fig.savefig(out_dir / "supplier_summary_heatmap.png", dpi=180)
    plt.close(fig)


def run_experiment(df_full: pd.DataFrame, X_full: np.ndarray, out_dir: Path) -> pd.DataFrame:
    methods: dict[str, Callable[[np.ndarray, pd.DataFrame], tuple[np.ndarray, np.ndarray]]] = {
        "ensemble_kmeans_if_lof": method_ensemble,
        "kmeans": method_kmeans,
        "iforest": method_iforest,
        "lof": method_lof,
        "ocsvm": method_ocsvm,
    }

    rng = np.random.default_rng(RANDOM_STATE)
    n_total = len(df_full)
    records: list[dict[str, float | int | str]] = []
    labels_on_full: dict[str, np.ndarray] = {}

    for size in SAMPLE_SIZES:
        if size > n_total:
            continue

        idx = rng.choice(n_total, size=size, replace=False)
        df_part = df_full.iloc[idx].reset_index(drop=True)
        X_part = X_full[idx]
        print(f"\n=== Sample size: {size} ===")

        for method_key in METHOD_ORDER:
            fn = methods[method_key]
            started = time.perf_counter()
            try:
                labels, scores = fn(X_part, df_part)
                _ = scores
                elapsed = time.perf_counter() - started
                metrics = compute_metrics(X_part, labels, elapsed_sec=elapsed)
                records.append(
                    {
                        "sample_size": size,
                        "method": method_key,
                        "method_label": METHOD_LABELS[method_key],
                        **metrics,
                    }
                )
                print(
                    f"[OK] {METHOD_LABELS[method_key]} | "
                    f"sil={metrics['silhouette']:.4f} | db={metrics['davies_bouldin']:.4f} | "
                    f"anom={metrics['anomaly_rate_pct']:.2f}% | t={metrics['time_sec']:.3f}s"
                )
            except Exception as exc:
                elapsed = time.perf_counter() - started
                print(f"[ERR] {METHOD_LABELS[method_key]}: {exc}")
                records.append(
                    {
                        "sample_size": size,
                        "method": method_key,
                        "method_label": METHOD_LABELS[method_key],
                        "silhouette": np.nan,
                        "davies_bouldin": np.nan,
                        "anomaly_rate_pct": np.nan,
                        "time_sec": elapsed,
                    }
                )

    full_size = n_total
    print(f"\n=== Full sample (for PCA/heatmap): {full_size} ===")
    for method_key in METHOD_ORDER:
        fn = methods[method_key]
        started = time.perf_counter()
        try:
            labels, scores = fn(X_full, df_full)
            _ = scores
            labels_on_full[method_key] = labels
            elapsed = time.perf_counter() - started
            metrics = compute_metrics(X_full, labels, elapsed_sec=elapsed)
            existing = [r for r in records if (r["sample_size"] == full_size and r["method"] == method_key)]
            if not existing:
                records.append(
                    {
                        "sample_size": full_size,
                        "method": method_key,
                        "method_label": METHOD_LABELS[method_key],
                        **metrics,
                    }
                )
        except Exception as exc:
            print(f"[ERR][FULL] {METHOD_LABELS[method_key]}: {exc}")
            labels_on_full[method_key] = np.zeros(n_total, dtype=int)

    metrics_df = pd.DataFrame(records).sort_values(["sample_size", "method"]).reset_index(drop=True)
    metrics_df.to_csv(out_dir / "supplier_metrics_table.csv", index=False)

    metrics_full_df = metrics_df[metrics_df["sample_size"] == n_total].copy()
    if metrics_full_df.empty:
        full_records = []
        for method_key in METHOD_ORDER:
            labels = labels_on_full.get(method_key, np.zeros(n_total, dtype=int))
            metrics = compute_metrics(X_full, labels, elapsed_sec=np.nan)
            full_records.append(
                {
                    "sample_size": n_total,
                    "method": method_key,
                    "method_label": METHOD_LABELS[method_key],
                    **metrics,
                }
            )
        metrics_full_df = pd.DataFrame(full_records)

    plot_stability(metrics_df, out_dir)
    plot_time(metrics_df, out_dir)
    plot_anomaly_rate(metrics_df, out_dir)
    plot_pca_grid(X_full, labels_on_full, metrics_full_df, out_dir)
    plot_summary_heatmap(metrics_full_df, out_dir)
    return metrics_df


def main() -> None:
    csv_path = "../../data/prepared/supplier_anomaly_results.csv"
    out_dir = Path("supplier_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    df, X = load_data(csv_path)
    metrics_df = run_experiment(df, X, out_dir)

    print("\nФінальна таблиця метрик:")
    with pd.option_context("display.max_rows", 200, "display.max_columns", 20, "display.width", 160):
        print(metrics_df.to_string(index=False))

    print("\nЗбережено у папку results/:")
    print("- supplier_metrics_table.csv")
    print("- supplier_stability_plot.png")
    print("- supplier_time_plot.png")
    print("- supplier_anomaly_rate_plot.png")
    print("- supplier_pca_visualization.png")
    print("- supplier_summary_heatmap.png")


if __name__ == "__main__":
    main()
