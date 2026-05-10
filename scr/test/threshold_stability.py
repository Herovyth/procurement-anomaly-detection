from __future__ import annotations

import argparse
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns



def _validate_inputs(
    df: pd.DataFrame,
    score_col: str,
    thresholds: list[float] | tuple[float, ...],
    id_col: str,
) -> list[float]:
    if df.empty:
        raise ValueError("`df` is empty.")
    if score_col not in df.columns:
        raise ValueError(f"Column `{score_col}` not found in df.")
    if id_col not in df.columns:
        raise ValueError(f"Column `{id_col}` not found in df.")
    if not thresholds:
        raise ValueError("`thresholds` must contain at least one value.")

    thresholds_sorted = sorted({float(t) for t in thresholds})
    for t in thresholds_sorted:
        if not (0.0 < t < 1.0):
            raise ValueError(f"Threshold {t} is out of bounds. Use values in (0, 1).")

    return thresholds_sorted


def _jaccard_similarity(a: set[Any], b: set[Any]) -> float:
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def analyze_threshold_stability(
    df: pd.DataFrame,
    score_col: str,
    thresholds: list[float] | tuple[float, ...],
    id_col: str,
) -> dict[str, Any]:
    """
    Analyze anomaly set stability across percentile thresholds.

    Steps:
    1) For each threshold p, define top anomalies as records with score > percentile(p).
    2) Build Jaccard similarity matrix between anomaly ID sets.
    3) Define stable core as IDs present in all thresholds >= 0.95.
    4) Output:
       - stats table (threshold | anomalies count | sample % | overlap with p=0.99),
       - Jaccard heatmap,
       - stable core list with risk_score,
       - rank stability plot for top-50 IDs at p=0.99.

    Returns:
        dict with keys: `jaccard_matrix`, `stable_core_ids`, `threshold_stats_df`.
    """
    thresholds_sorted = _validate_inputs(df, score_col, thresholds, id_col)

    work_df = df[[id_col, score_col]].copy()
    work_df[score_col] = pd.to_numeric(work_df[score_col], errors="coerce")
    work_df = work_df.dropna(subset=[score_col, id_col]).drop_duplicates(subset=[id_col], keep="first")

    if work_df.empty:
        raise ValueError("No valid rows remain after cleaning NaN values and duplicate IDs.")

    id_sets_by_threshold: dict[float, set[Any]] = {}
    threshold_cutoffs: dict[float, float] = {}

    for p in thresholds_sorted:
        cutoff = float(work_df[score_col].quantile(p))
        top_ids = set(work_df.loc[work_df[score_col] > cutoff, id_col].tolist())
        threshold_cutoffs[p] = cutoff
        id_sets_by_threshold[p] = top_ids

    # Jaccard matrix
    t_labels = [f"{t:.3f}" for t in thresholds_sorted]
    jaccard_matrix = pd.DataFrame(index=t_labels, columns=t_labels, dtype=float)
    for t1 in thresholds_sorted:
        for t2 in thresholds_sorted:
            jaccard_matrix.loc[f"{t1:.3f}", f"{t2:.3f}"] = _jaccard_similarity(
                id_sets_by_threshold[t1], id_sets_by_threshold[t2]
            )

    # Stable core (all thresholds >= 0.95)
    high_thresholds = [t for t in thresholds_sorted if t >= 0.95]
    if high_thresholds:
        stable_core_ids = set.intersection(*(id_sets_by_threshold[t] for t in high_thresholds))
    else:
        stable_core_ids = set()

    # Stats table with overlap vs p=0.99
    ref_threshold = 0.99
    closest_ref = min(thresholds_sorted, key=lambda x: abs(x - ref_threshold))
    ref_set = id_sets_by_threshold[closest_ref]

    stats_rows: list[dict[str, Any]] = []
    n_total = len(work_df)
    for t in thresholds_sorted:
        current_set = id_sets_by_threshold[t]
        overlap = _jaccard_similarity(current_set, ref_set)
        stats_rows.append(
            {
                "threshold": t,
                "cutoff_score": threshold_cutoffs[t],
                "anomaly_count": len(current_set),
                "sample_pct": (len(current_set) / n_total) * 100.0,
                f"overlap_vs_p={closest_ref:.3f}": overlap,
            }
        )
    threshold_stats_df = pd.DataFrame(stats_rows).sort_values("threshold").reset_index(drop=True)

    print("\nThreshold stats")
    print(threshold_stats_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # Jaccard heatmap
    plt.figure(figsize=(8, 6))
    if sns is not None:
        sns.heatmap(jaccard_matrix, annot=True, fmt=".3f", cmap="Blues", vmin=0.0, vmax=1.0)
    else:
        plt.imshow(jaccard_matrix.values, cmap="Blues", vmin=0.0, vmax=1.0)
        plt.colorbar(label="Jaccard similarity")
        plt.xticks(range(len(t_labels)), t_labels)
        plt.yticks(range(len(t_labels)), t_labels)
        for i in range(len(t_labels)):
            for j in range(len(t_labels)):
                plt.text(j, i, f"{jaccard_matrix.values[i, j]:.3f}", ha="center", va="center", color="black")
    plt.title("Jaccard Similarity Across Thresholds")
    plt.xlabel("Threshold")
    plt.ylabel("Threshold")
    plt.tight_layout()
    plt.show()

    # Stable core with scores
    stable_core_df = (
        work_df[work_df[id_col].isin(stable_core_ids)]
        .sort_values(score_col, ascending=False)
        .reset_index(drop=True)
    )
    print(f"\nStable core (thresholds >= 0.95): {len(stable_core_df)} IDs")
    if not stable_core_df.empty:
        print(stable_core_df[[id_col, score_col]].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    else:
        print("No stable core IDs found.")

    # Rank stability plot for top-50 at p=0.99 (or nearest available)
    ref_cutoff = threshold_cutoffs[closest_ref]
    ref_top_df = work_df.loc[work_df[score_col] > ref_cutoff, [id_col, score_col]].copy()
    ref_top_df = ref_top_df.sort_values(score_col, ascending=False).head(50)
    top50_ids = ref_top_df[id_col].tolist()

    if top50_ids:
        plot_rows: list[dict[str, Any]] = []
        for t in thresholds_sorted:
            cutoff = threshold_cutoffs[t]
            current_top = work_df.loc[work_df[score_col] > cutoff, [id_col, score_col]].copy()
            if current_top.empty:
                continue

            current_top = current_top.sort_values(score_col, ascending=False).reset_index(drop=True)
            current_top["rank_pct"] = (current_top.index + 1) / len(current_top)
            rank_map = dict(zip(current_top[id_col], current_top["rank_pct"]))

            for entity_id in top50_ids:
                plot_rows.append(
                    {
                        "id": entity_id,
                        "threshold": t,
                        "rank_pct": rank_map.get(entity_id, np.nan),
                    }
                )

        rank_stability_df = pd.DataFrame(plot_rows)

        plt.figure(figsize=(12, 7))
        for entity_id, entity_sub in rank_stability_df.groupby("id"):
            plt.plot(
                entity_sub["threshold"],
                entity_sub["rank_pct"],
                alpha=0.45,
                linewidth=1.0,
            )
        plt.gca().invert_yaxis()
        plt.title(f"Rank Stability for Top-50 at p={closest_ref:.3f}")
        plt.xlabel("Threshold")
        plt.ylabel("Percentile rank within threshold top-set (lower is better)")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.show()
    else:
        print(f"\nNo entities found above p={closest_ref:.3f}; rank stability plot skipped.")

    return {
        "jaccard_matrix": jaccard_matrix,
        "stable_core_ids": sorted(stable_core_ids),
        "threshold_stats_df": threshold_stats_df,
    }


def _parse_thresholds(raw: str) -> list[float]:
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("No thresholds provided.")
    return [float(v) for v in values]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Threshold stability analysis for anomaly risk scores."
    )
    parser.add_argument(
        "--input-csv",
        default="../../data/prepared/tender_anomaly_results.csv",
        help="Path to input dataframe CSV.",
    )
    parser.add_argument(
        "--score-col",
        default="risk_score",
        help="Column with anomaly score (default: risk_score).",
    )
    parser.add_argument(
        "--id-col",
        required=True,
        help="Entity ID column (e.g., supplier_id, tender_id).",
    )
    parser.add_argument(
        "--thresholds",
        default="0.90,0.95,0.99,0.995,0.999",
        help="Comma-separated percentiles in (0,1).",
    )

    args = parser.parse_args()
    thresholds = _parse_thresholds(args.thresholds)
    df = pd.read_csv(args.input_csv)

    result = analyze_threshold_stability(
        df=df,
        score_col=args.score_col,
        thresholds=thresholds,
        id_col=args.id_col,
    )

    print("\nReturned keys:", list(result.keys()))
    print("Stable core size:", len(result["stable_core_ids"]))


if __name__ == "__main__":
    main()
