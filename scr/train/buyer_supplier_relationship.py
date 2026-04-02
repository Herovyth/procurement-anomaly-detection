import pandas as pd


try:
    rel_df = pd.read_csv("data/relationship_level_features.csv")
except FileNotFoundError:
    print("relationship_level_features.csv не знайдено")
    exit()


rel_df["high_win_share"] = (rel_df["buyer_win_share"] > 0.7).astype(int)

rel_df["low_competition"] = (rel_df["avg_competitors"] <= 2).astype(int)

rel_df["repeated_pair"] = (rel_df["num_tenders"] >= 5).astype(int)

rel_df["high_supplier_income"] = (rel_df["supplier_income_share"] > 0.5).astype(int)

# rel_df["long_streak"] = (rel_df["win_streak"] >= 3).astype(int)

rel_df["high_single_bid_share"] = (rel_df["single_bid_share"] > 0.5).astype(int)

# missing_wr = rel_df["win_regular_months"].isna()
# if "win_regular_months_missing" in rel_df.columns:
#     missing_wr = missing_wr | (rel_df["win_regular_months_missing"].astype(int) != 0)
# rel_df["tight_win_spacing"] = (
#     (~missing_wr)
#     & (rel_df["num_tenders"] >= 2)
#     & (rel_df["win_regular_months"] >= 0)
#     & (rel_df["win_regular_months"] <= 4.0)
# ).astype(int)

loc_b = rel_df["buyer_locality"].astype(str).str.strip().str.lower()
loc_s = rel_df["supplier_locality"].astype(str).str.strip().str.lower()
unknown = {"", "nan", "невідомо", "none"}

# Project requirement: use avg_price signal instead of same_locality.
# Mark relationship as high-price if avg_price is in top quartile.
avg_price_thr = rel_df["avg_price"].quantile(0.75)
rel_df["high_avg_price"] = (rel_df["avg_price"] >= avg_price_thr).astype(int)

rel_df["risk_score"] = (
    2 * rel_df["high_win_share"] +
    1 * rel_df["low_competition"] +
    2 * rel_df["repeated_pair"] +
    1 * rel_df["high_supplier_income"] +
    # 1 * rel_df["long_streak"] +
    2 * rel_df["high_single_bid_share"] +
    # 1 * rel_df["tight_win_spacing"] +
    1 * rel_df["high_avg_price"]
)


def classify_risk(score):
    # Max raw score = 9 in current active rules.
    if score >= 6:
        return "high"
    elif score >= 4:
        return "medium"
    else:
        return "low"


rel_df["risk_level"] = rel_df["risk_score"].apply(classify_risk)

rel_df.to_csv("data/relationship_anomaly_results.csv", index=False, encoding="utf-8")

print("Готово! Top-10 підозрілих зв'язків:")
print(rel_df.sort_values("risk_score", ascending=False).head(10))
