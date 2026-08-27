"""
Аналіз результатів rule-based оцінки зв'язків замовник–постачальник.

Читає `relationship_anomaly_results.csv` (або перераховує з raw features)
і формує таблиці та графіки для розділу дипломної про виявлення ризикових пар.

Запуск з кореня репозиторію:
    python scr/test/analyze_relationship.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SCR_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SCR_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scr.train.buyer_supplier_relationship import (
    MAX_RISK_SCORE,
    RULE_COLUMNS,
    RULE_LABELS_UA,
    RULE_WEIGHTS,
    classify_risk,
    score_relationships,
)

PREPARED_CSV = PROJECT_ROOT / "data/prepared/relationship_anomaly_results.csv"
RAW_CSV = PROJECT_ROOT / "data/raw/relationship_level_features.csv"
OUT_DIR = PROJECT_ROOT / "results/relationship_results"

NUMERIC_FEATURES = [
    "num_tenders",
    "avg_price",
    "avg_competitors",
    "single_bid_share",
    "buyer_win_share",
    "supplier_income_share",
]

RISK_LEVEL_ORDER = ["low", "medium", "high"]
RISK_COLORS = {"low": "#2ecc71", "medium": "#f39c12", "high": "#e74c3c"}


def load_relationship_results() -> pd.DataFrame:
    if PREPARED_CSV.exists():
        return pd.read_csv(PREPARED_CSV)
    if RAW_CSV.exists():
        print(f"Файл {PREPARED_CSV.name} не знайдено — перераховую з raw features.")
        return score_relationships(pd.read_csv(RAW_CSV))
    raise FileNotFoundError(
        f"Не знайдено ні {PREPARED_CSV}, ні {RAW_CSV}. "
        "Спочатку запустіть extraction та buyer_supplier_relationship.py."
    )


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    level_counts = df["risk_level"].value_counts().reindex(RISK_LEVEL_ORDER, fill_value=0)
    level_pct = (level_counts / len(df) * 100).round(2)

    return pd.DataFrame(
        [
            {
                "n_pairs": len(df),
                "n_buyers": df["buyer_id"].nunique(),
                "n_suppliers": df["supplier_id"].nunique(),
                "risk_score_mean": round(df["risk_score"].mean(), 3),
                "risk_score_median": float(df["risk_score"].median()),
                "risk_score_std": round(df["risk_score"].std(ddof=0), 3),
                "risk_score_min": int(df["risk_score"].min()),
                "risk_score_max": int(df["risk_score"].max()),
                "max_possible_score": MAX_RISK_SCORE,
                "pct_low": float(level_pct["low"]),
                "pct_medium": float(level_pct["medium"]),
                "pct_high": float(level_pct["high"]),
                "n_high": int(level_counts["high"]),
                "n_medium": int(level_counts["medium"]),
                "n_low": int(level_counts["low"]),
            }
        ]
    )


def compute_rule_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for rule in RULE_COLUMNS:
        triggered = df[rule].astype(int) == 1
        n_triggered = int(triggered.sum())
        rows.append(
            {
                "rule": rule,
                "rule_label_ua": RULE_LABELS_UA[rule],
                "weight": RULE_WEIGHTS[rule],
                "n_triggered": n_triggered,
                "pct_triggered": round(n_triggered / len(df) * 100, 2),
                "mean_risk_when_triggered": round(df.loc[triggered, "risk_score"].mean(), 3)
                if n_triggered
                else np.nan,
                "mean_risk_when_not_triggered": round(df.loc[~triggered, "risk_score"].mean(), 3)
                if (~triggered).any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def compute_rule_cooccurrence(df: pd.DataFrame) -> pd.DataFrame:
    rule_mat = df[RULE_COLUMNS].astype(int)
    corr = rule_mat.corr()
    corr.index = [RULE_LABELS_UA[c] for c in corr.index]
    corr.columns = [RULE_LABELS_UA[c] for c in corr.columns]
    return corr.round(3)


def compute_feature_stats_by_risk(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for level in RISK_LEVEL_ORDER:
        sub = df[df["risk_level"] == level]
        if sub.empty:
            continue
        for feat in NUMERIC_FEATURES:
            rows.append(
                {
                    "risk_level": level,
                    "feature": feat,
                    "count": len(sub),
                    "mean": round(sub[feat].mean(), 4),
                    "median": round(sub[feat].median(), 4),
                    "p25": round(sub[feat].quantile(0.25), 4),
                    "p75": round(sub[feat].quantile(0.75), 4),
                }
            )
    return pd.DataFrame(rows)


def compute_entity_summaries(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    buyer = (
        df.groupby("buyer_id", as_index=False)
        .agg(
            n_relationships=("supplier_id", "count"),
            n_high_risk=("risk_level", lambda s: int((s == "high").sum())),
            n_medium_risk=("risk_level", lambda s: int((s == "medium").sum())),
            max_risk_score=("risk_score", "max"),
            mean_risk_score=("risk_score", "mean"),
        )
        .sort_values(["n_high_risk", "max_risk_score", "mean_risk_score"], ascending=False)
    )
    buyer["pct_high_risk"] = (buyer["n_high_risk"] / buyer["n_relationships"] * 100).round(2)

    supplier = (
        df.groupby("supplier_id", as_index=False)
        .agg(
            n_relationships=("buyer_id", "count"),
            n_high_risk=("risk_level", lambda s: int((s == "high").sum())),
            n_medium_risk=("risk_level", lambda s: int((s == "medium").sum())),
            max_risk_score=("risk_score", "max"),
            mean_risk_score=("risk_score", "mean"),
        )
        .sort_values(["n_high_risk", "max_risk_score", "mean_risk_score"], ascending=False)
    )
    supplier["pct_high_risk"] = (supplier["n_high_risk"] / supplier["n_relationships"] * 100).round(2)
    return buyer, supplier


def top_suspicious_pairs(df: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    cols = [
        "buyer_id",
        "supplier_id",
        "num_tenders",
        "avg_price",
        "avg_competitors",
        "single_bid_share",
        "buyer_win_share",
        "supplier_income_share",
        *RULE_COLUMNS,
        "risk_score",
        "risk_level",
    ]
    return df.sort_values("risk_score", ascending=False).head(n)[cols].reset_index(drop=True)


def plot_risk_score_hist(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(-0.5, MAX_RISK_SCORE + 1.5, 1)
    ax.hist(df["risk_score"], bins=bins, color="#3498db", edgecolor="white", alpha=0.9)
    ax.axvline(4, color="#f39c12", linestyle="--", linewidth=1.5, label="Поріг medium (≥4)")
    ax.axvline(6, color="#e74c3c", linestyle="--", linewidth=1.5, label="Поріг high (≥6)")
    ax.set_xlabel("Risk score (сума ваг правил)")
    ax.set_ylabel("Кількість пар")
    ax.set_title("Розподіл risk score для пар замовник–постачальник")
    ax.set_xticks(range(0, MAX_RISK_SCORE + 1))
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "risk_score_hist.png", dpi=180)
    plt.close(fig)


def plot_risk_level_bar(df: pd.DataFrame, out_dir: Path) -> None:
    counts = df["risk_level"].value_counts().reindex(RISK_LEVEL_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [RISK_COLORS[l] for l in counts.index]
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white")
    for bar, val in zip(bars, counts.values):
        pct = val / len(df) * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Кількість пар")
    ax.set_title("Розподіл категорій ризику (low / medium / high)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "risk_level_bar.png", dpi=180)
    plt.close(fig)


def plot_rule_frequency(rule_stats: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    labels = [f"{row.rule}\n(w={row.weight})" for row in rule_stats.itertuples()]
    ax.barh(labels, rule_stats["pct_triggered"], color="#5dade2", edgecolor="white")
    ax.set_xlabel("Частка пар, %")
    ax.set_title("Частота спрацювання правил rule-based scoring")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "rule_frequency_bar.png", dpi=180)
    plt.close(fig)


def plot_rule_cooccurrence_heatmap(corr: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdYlBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
        cbar_kws={"label": "Кореляція Пірсона"},
    )
    ax.set_title("Кореляція між бінарними правилами ризику")
    fig.tight_layout()
    fig.savefig(out_dir / "rule_cooccurrence_heatmap.png", dpi=180)
    plt.close(fig)


def plot_features_by_risk_level(df: pd.DataFrame, out_dir: Path) -> None:
    plot_df = df[["risk_level", *NUMERIC_FEATURES]].copy()
    plot_df["risk_level"] = pd.Categorical(plot_df["risk_level"], categories=RISK_LEVEL_ORDER, ordered=True)
    melted = plot_df.melt(id_vars="risk_level", var_name="feature", value_name="value")
    melted["value"] = pd.to_numeric(melted["value"], errors="coerce")
    melted = melted.replace([np.inf, -np.inf], np.nan).dropna(subset=["value"])

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, feat in zip(axes.flatten(), NUMERIC_FEATURES):
        sub = melted[melted["feature"] == feat]
        sns.boxplot(
            data=sub,
            x="risk_level",
            y="value",
            hue="risk_level",
            order=RISK_LEVEL_ORDER,
            hue_order=RISK_LEVEL_ORDER,
            palette=RISK_COLORS,
            legend=False,
            ax=ax,
            fliersize=2,
        )
        ax.set_title(feat)
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Розподіл ознак за категорією ризику", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "features_by_risk_level.png", dpi=180)
    plt.close(fig)


def plot_pca_by_risk(df: pd.DataFrame, out_dir: Path) -> None:
    X = df[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = X.notna().all(axis=1)
    if valid.sum() < 10:
        print("Пропущено PCA: замало повних рядків.")
        return

    X_scaled = StandardScaler().fit_transform(X.loc[valid])
    pca2 = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    levels = df.loc[valid, "risk_level"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    for level in RISK_LEVEL_ORDER:
        mask = levels == level
        if not mask.any():
            continue
        ax.scatter(
            pca2[mask, 0],
            pca2[mask, 1],
            s=12,
            alpha=0.55,
            c=RISK_COLORS[level],
            label=f"{level} (n={mask.sum()})",
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA(2) пар за числовими ознаками, колір = risk_level")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "pca_by_risk_level.png", dpi=180)
    plt.close(fig)


def plot_top_entities(
    entity_df: pd.DataFrame,
    id_col: str,
    title: str,
    filename: str,
    out_dir: Path,
    top_n: int = 15,
) -> None:
    sub = entity_df[entity_df["n_high_risk"] > 0].head(top_n)
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(sub[id_col].astype(str), sub["n_high_risk"], color="#e74c3c", edgecolor="white")
    ax.set_xlabel("Кількість high-risk зв'язків")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=180)
    plt.close(fig)


def run_analysis(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = compute_summary(df)
    rule_stats = compute_rule_stats(df)
    cooccurrence = compute_rule_cooccurrence(df)
    feature_stats = compute_feature_stats_by_risk(df)
    buyer_summary, supplier_summary = compute_entity_summaries(df)
    top_pairs = top_suspicious_pairs(df)

    summary.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8")
    rule_stats.to_csv(out_dir / "rule_stats.csv", index=False, encoding="utf-8")
    cooccurrence.to_csv(out_dir / "rule_cooccurrence.csv", encoding="utf-8")
    feature_stats.to_csv(out_dir / "feature_stats_by_risk_level.csv", index=False, encoding="utf-8")
    buyer_summary.to_csv(out_dir / "buyer_risk_summary.csv", index=False, encoding="utf-8")
    supplier_summary.to_csv(out_dir / "supplier_risk_summary.csv", index=False, encoding="utf-8")
    top_pairs.to_csv(out_dir / "top_suspicious_pairs.csv", index=False, encoding="utf-8")

    plot_risk_score_hist(df, out_dir)
    plot_risk_level_bar(df, out_dir)
    plot_rule_frequency(rule_stats, out_dir)
    plot_rule_cooccurrence_heatmap(cooccurrence, out_dir)
    plot_features_by_risk_level(df, out_dir)
    plot_pca_by_risk(df, out_dir)
    plot_top_entities(
        buyer_summary,
        "buyer_id",
        "Топ замовників за кількістю high-risk зв'язків",
        "top_buyers_high_risk.png",
        out_dir,
    )
    plot_top_entities(
        supplier_summary,
        "supplier_id",
        "Топ постачальників за кількістю high-risk зв'язків",
        "top_suppliers_high_risk.png",
        out_dir,
    )

    row = summary.iloc[0]
    print("\n=== Підсумок для дипломної (зв'язки замовник–постачальник) ===")
    print(f"Пар у вибірці: {int(row.n_pairs)} (замовників: {int(row.n_buyers)}, постачальників: {int(row.n_suppliers)})")
    print(
        f"Категорії ризику: low {row.pct_low:.1f}% | medium {row.pct_medium:.1f}% | high {row.pct_high:.1f}%"
    )
    print(f"Risk score: mean={row.risk_score_mean}, median={row.risk_score_median}, max={int(row.risk_score_max)}")
    print("\nНайчастіші правила:")
    for r in rule_stats.sort_values("pct_triggered", ascending=False).itertuples():
        print(f"  - {r.rule}: {r.pct_triggered}% пар (вага {r.weight})")
    print(f"\nРезультати збережено у: {out_dir}")


def main() -> None:
    df = load_relationship_results()
    missing_rules = [c for c in RULE_COLUMNS if c not in df.columns]
    if missing_rules:
        if not RAW_CSV.exists():
            raise ValueError(f"У CSV відсутні колонки правил: {missing_rules}")
        df = score_relationships(pd.read_csv(RAW_CSV))

    if "risk_level" not in df.columns and "risk_score" in df.columns:
        df["risk_level"] = df["risk_score"].apply(classify_risk)

    run_analysis(df, OUT_DIR)


if __name__ == "__main__":
    main()
