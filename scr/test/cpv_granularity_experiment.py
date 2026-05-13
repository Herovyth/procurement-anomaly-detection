"""
Експеримент для пояснення, чому CPV2 (2 перші цифри CPV) зазвичай кращий
компроміс для групування в tender anomaly pipeline, ніж CPV1 або CPV3.

Що робить:
1) бере семпл з tender_anomaly_results.csv,
2) формує групи CPV1 / CPV2 / CPV3,
3) запускає IsolationForest всередині груп,
4) оцінює:
   - розмір груп,
   - coverage окремих моделей,
   - внутрішньогрупову variance,
   - стабільність anomaly scores,
   - runtime.

ВАЖЛИВО:
Silhouette / Davies-Bouldin спеціально НЕ використовуються,
бо CPV-коди — це taxonomy labels, а не ML-clusters.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


# Allow running directly via:
# python scr/test/cpv_granularity_experiment.py
SCR_ROOT = Path(__file__).resolve().parents[1]

if str(SCR_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(SCR_ROOT.parent))

from scr.train.tender_anomaly import (
    NUMERIC_FEATURES,
    prepare_numeric_features,
)


def _extract_cpv_prefix(cpv: object, n_digits: int) -> str:
    if cpv is None or (isinstance(cpv, float) and np.isnan(cpv)):
        return "NA"

    digits = "".join(ch for ch in str(cpv).strip() if ch.isdigit())

    if len(digits) < n_digits:
        return "NA"

    return digits[:n_digits]


def _within_group_distance(
    X_scaled: np.ndarray,
    groups: pd.Series,
) -> float:
    """
    Average normalized within-group distance.

    Lower value =>
    tenders inside groups are more homogeneous.
    """

    distances = []

    for _, idx in groups.groupby(groups).groups.items():
        idx = np.asarray(list(idx), dtype=int)

        if len(idx) < 2:
            continue

        group_X = X_scaled[idx]

        centroid = np.mean(group_X, axis=0)

        dists = np.linalg.norm(
            group_X - centroid,
            axis=1,
        )

        distances.extend(dists.tolist())

    if not distances:
        return float("nan")

    return float(np.mean(distances))


def numeric_anomaly_score_by_custom_group(
    X_scaled: np.ndarray,
    groups: pd.Series,
    min_group_size: int,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    IsolationForest anomaly scoring:
    - окремі моделі для достатньо великих груп,
    - fallback global model для малих груп.
    """

    n = X_scaled.shape[0]

    raw_scores = np.zeros(n, dtype=float)
    percentile_scores = np.zeros(n, dtype=float)

    # Global fallback model
    fallback_if = IsolationForest(
        n_estimators=250,
        contamination="auto",
        random_state=random_state,
        n_jobs=-1,
    )

    fallback_if.fit(X_scaled)

    fallback_raw = -fallback_if.score_samples(X_scaled)

    sizes = groups.value_counts(dropna=False)

    for g, idx in groups.groupby(groups).groups.items():
        idx = np.asarray(list(idx), dtype=int)

        # Small group => fallback model
        if sizes.get(g, 0) < min_group_size:
            raw_scores[idx] = fallback_raw[idx]
            continue

        # Dedicated group model
        iso = IsolationForest(
            n_estimators=400,
            contamination="auto",
            random_state=random_state,
            n_jobs=-1,
        )

        iso.fit(X_scaled[idx])

        raw_scores[idx] = -iso.score_samples(X_scaled[idx])

    # Percentile normalization inside groups
    for _, idx in groups.groupby(groups).groups.items():
        idx = np.asarray(list(idx), dtype=int)

        s = pd.Series(raw_scores[idx])

        percentile_scores[idx] = (
            s.rank(pct=True, method="average").to_numpy()
        )

    return raw_scores, percentile_scores


def _evaluate_granularity(
    df: pd.DataFrame,
    X_scaled: np.ndarray,
    cpv_digits: int,
    min_group_size: int,
) -> dict[str, float | int | str]:
    group_col = f"cpv{cpv_digits}"

    groups = df[group_col].fillna("NA").astype(str)

    sizes = groups.value_counts(dropna=False)

    # Share of rows using dedicated IF models
    rows_in_dedicated = int(
        sizes[sizes >= min_group_size].sum()
    )

    dedicated_share = (
        rows_in_dedicated / len(df)
        if len(df)
        else 0.0
    )

    # Runtime
    t0 = time.perf_counter()

    raw_scores, percentile_scores = (
        numeric_anomaly_score_by_custom_group(
            X_scaled=X_scaled,
            groups=groups,
            min_group_size=min_group_size,
        )
    )

    elapsed = time.perf_counter() - t0

    # Main grouping quality metric
    within_group_var = _within_group_distance(
        X_scaled,
        groups,
    )

    # Score stability
    raw_std = float(np.std(raw_scores))
    pct_std = float(np.std(percentile_scores))

    return {
        "cpv_level": group_col,
        "n_groups": int(sizes.shape[0]),
        "median_group_size": float(sizes.median()),
        "min_group_size_threshold": int(min_group_size),
        "dedicated_model_share": float(dedicated_share),
        "within_group_variance": float(within_group_var),
        "raw_score_std": raw_std,
        "pct_score_std": pct_std,
        "runtime_sec": float(elapsed),
    }


def run_cpv_granularity_experiment(
    path: str = "../../data/prepared/tender_anomaly_results.csv",
    sample_size: int = 10000,
    random_state: int = 42,
) -> pd.DataFrame:

    df = pd.read_csv(path)

    if "cpv" not in df.columns:
        raise ValueError(
            "Column 'cpv' is required"
        )

    # Reproducible sample
    n = min(sample_size, len(df))

    df = (
        df.sample(
            n=n,
            random_state=random_state,
        )
        .reset_index(drop=True)
    )

    # CPV prefixes
    df["cpv1"] = df["cpv"].apply(
        lambda x: _extract_cpv_prefix(x, 1)
    )

    df["cpv2"] = df["cpv"].apply(
        lambda x: _extract_cpv_prefix(x, 2)
    )

    df["cpv3"] = df["cpv"].apply(
        lambda x: _extract_cpv_prefix(x, 3)
    )

    # Validate numeric columns
    required = [
        c for c in NUMERIC_FEATURES
        if c in df.columns
    ]

    if len(required) != len(NUMERIC_FEATURES):
        missing = sorted(
            set(NUMERIC_FEATURES) - set(required)
        )

        raise ValueError(
            f"Missing required numeric columns: {missing}"
        )

    # Feature scaling
    X_scaled = prepare_numeric_features(df)

    # Adaptive threshold
    min_group_size = max(
        30,
        int(np.sqrt(len(df))),
    )

    rows = [
        _evaluate_granularity(
            df=df,
            X_scaled=X_scaled,
            cpv_digits=1,
            min_group_size=min_group_size,
        ),
        _evaluate_granularity(
            df=df,
            X_scaled=X_scaled,
            cpv_digits=2,
            min_group_size=min_group_size,
        ),
        _evaluate_granularity(
            df=df,
            X_scaled=X_scaled,
            cpv_digits=3,
            min_group_size=min_group_size,
        ),
    ]

    out = pd.DataFrame(rows)

    out_path = Path(
        "../../results/cpv_granularity_comparison.csv"
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_csv(out_path, index=False)

    print("\nCPV granularity comparison:\n")

    print(out.to_string(index=False))

    print(f"\nSaved table: {out_path}")

    print(
        "\nЯк читати результат:\n"
        "- within_group_variance нижче -> групи більш однорідні;\n"
        "- dedicated_model_share низьке -> занадто дрібне grouping;\n"
        "- n_groups дуже мале -> grouping занадто грубе;\n"
        "- CPV2 зазвичай дає найкращий баланс;\n"
        "- raw_score_std / pct_score_std показують стабільність anomaly scoring."
    )

    return out


if __name__ == "__main__":
    run_cpv_granularity_experiment()