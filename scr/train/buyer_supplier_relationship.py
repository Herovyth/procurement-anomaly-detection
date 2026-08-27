"""
buyer_supplier_relationship.py

Що робить:
- Обчислює rule-based risk score для пари "замовник–постачальник".

Коли використовується:
- Після формування `relationship_level_features.csv`, як швидкий інтерпретований
  етап оцінки ризику зв'язків у стилі дипломної записки.

Навіщо:
- Виявити повторювані/малоконкурентні взаємодії (ознаки можливих узгоджених
  дій), навіть без складної ML-моделі.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RULE_COLUMNS = [
    "high_win_share",
    "low_competition",
    "repeated_pair",
    "high_supplier_income",
    "high_single_bid_share",
    "high_avg_price",
]

RULE_WEIGHTS: dict[str, int] = {
    "high_win_share": 2,
    "low_competition": 1,
    "repeated_pair": 2,
    "high_supplier_income": 1,
    "high_single_bid_share": 2,
    "high_avg_price": 1,
}

RULE_LABELS_UA: dict[str, str] = {
    "high_win_share": "Висока частка перемог постачальника у замовника",
    "low_competition": "Низька конкуренція (avg_competitors ≤ 2)",
    "repeated_pair": "Повторювана пара (num_tenders ≥ 5)",
    "high_supplier_income": "Висока залежність доходу постачальника від замовника",
    "high_single_bid_share": "Висока частка процедур з 1 учасником",
    "high_avg_price": "Висока середня ціна (верхній квартиль)",
}

MAX_RISK_SCORE = sum(RULE_WEIGHTS.values())


def classify_risk(score: int | float) -> str:
    """Перетворює сумарний бал правил у категорію ризику для інтерфейсу."""
    if score >= 6:
        return "high"
    if score >= 4:
        return "medium"
    return "low"


def score_relationships(rel_df: pd.DataFrame) -> pd.DataFrame:
    """Додає бінарні правила, risk_score та risk_level до датафрейму пар."""
    out = rel_df.copy()

    out["high_win_share"] = (out["buyer_win_share"] > 0.7).astype(int)
    out["low_competition"] = (out["avg_competitors"] <= 2).astype(int)
    out["repeated_pair"] = (out["num_tenders"] >= 5).astype(int)
    out["high_supplier_income"] = (out["supplier_income_share"] > 0.5).astype(int)
    out["high_single_bid_share"] = (out["single_bid_share"] > 0.5).astype(int)

    avg_price_thr = out["avg_price"].quantile(0.75)
    out["high_avg_price"] = (out["avg_price"] >= avg_price_thr).astype(int)

    out["risk_score"] = sum(out[col] * weight for col, weight in RULE_WEIGHTS.items())
    out["risk_level"] = out["risk_score"].apply(classify_risk)
    return out


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    input_path = project_root / "data/raw/relationship_level_features.csv"
    output_path = project_root / "data/prepared/relationship_anomaly_results.csv"

    try:
        rel_df = pd.read_csv(input_path)
    except FileNotFoundError:
        print("relationship_level_features.csv не знайдено")
        raise SystemExit(1)

    scored = score_relationships(rel_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False, encoding="utf-8")

    print("Готово! Top-10 підозрілих зв'язків:")
    print(scored.sort_values("risk_score", ascending=False).head(10))


if __name__ == "__main__":
    main()
