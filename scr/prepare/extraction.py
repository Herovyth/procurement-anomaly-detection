from dateutil.parser import isoparse
from pymongo import MongoClient
import pandas as pd
from datetime import datetime
import numpy as np

import requests

from document_extraction import (
    DOCUMENT_FETCH_ENABLED,
    DOC_NLP_PCA_COMPONENTS,
    TEXT_MODEL,
    summarize_tier0,
    summarize_tier1_for_documents,
    tier1_placeholder_meta_and_blob,
)

FIXED_NOW = datetime(2025, 12, 11)


def extract_tender_features(t, http_session=None):
    tender_id = t.get("tenderID")
    status = t.get("status")
    title = t.get("title")
    description = t.get("description")

    value_amount = t.get("value", {}).get("amount")

    # Дати тендеру
    start = t.get("tenderPeriod", {}).get("startDate")
    end = t.get("tenderPeriod", {}).get("endDate")

    tender_duration_days = None
    completion_days = None

    if start:
        start_dt = isoparse(start)
        if end:
            end_dt = isoparse(end)
            tender_duration_days = (end_dt - start_dt).days
            completion_days = tender_duration_days
        else:
            tender_duration_days = (FIXED_NOW - start_dt).days
            completion_days = tender_duration_days

    bids = t.get("bids", [])
    active_bids = [b for b in bids if b.get("status") == "active"]

    bid_prices = [
        b.get("value", {}).get("amount")
        for b in active_bids
        if b.get("value", {}).get("amount") is not None
    ]

    awards = [a for a in t.get("awards", []) if a.get("status") == "active"]
    winner_price = awards[0]["value"]["amount"] if awards else None

    price_per_unit = None
    items = t.get("items", [])
    if items:
        prices = []
        for it in items:
            q = it.get("quantity")
            v = it.get("unit", {}).get("value", {}).get("amount")
            if q and v:
                prices.append(v / q)
        if prices:
            price_per_unit = sum(prices) / len(prices)
    if price_per_unit is None:
        price_per_unit = value_amount

    docs = t.get("documents") or []
    doc_t0 = summarize_tier0(docs)
    if DOCUMENT_FETCH_ENABLED:
        doc_t1, doc_blob = summarize_tier1_for_documents(docs, session=http_session)
    else:
        doc_t1, doc_blob = tier1_placeholder_meta_and_blob()

    row = {
        "tender_id": tender_id,
        "cpv": t.get("items")[0].get("classification", {}).get("id") if items else None,
        "status": status,
        "title": title,
        "description": description,
        "value_amount": value_amount,
        "winner_price": winner_price,
        "price_per_unit": price_per_unit,
        "value_per_day": value_amount / tender_duration_days if value_amount and tender_duration_days else None,
        "numberOfBids": len(bids),
        "numberOfAdmitted": len(active_bids),
        "min_bid": min(bid_prices) if bid_prices else None,
        "winner_minus_min": winner_price - min(bid_prices) if winner_price and bid_prices else None,
        "tender_duration_days": tender_duration_days,
        "completion_days": completion_days,
        "complaints": len(t.get("complaints", [])),
        "tender_changes": len(t.get("amendments", [])),
        "contract_changes": sum(len(c.get("changes", [])) for c in t.get("contracts", [])),
    }
    row.update(doc_t0)
    row.update(doc_t1)
    row["_doc_blob"] = doc_blob
    return row


def add_document_nlp_pca_columns(tender_df: pd.DataFrame, doc_blobs: list[str]) -> None:
    """SentenceTransformer embeddings of extracted document text → PCA columns (in-place)."""
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA

    n = len(tender_df)
    for i in range(1, DOC_NLP_PCA_COMPONENTS + 1):
        tender_df[f"doc_nlp_pca_{i}"] = 0.0

    if n == 0 or len(doc_blobs) != n:
        return

    if not DOCUMENT_FETCH_ENABLED:
        return

    texts = [b if (b and str(b).strip()) else " " for b in doc_blobs]
    n_comp = min(DOC_NLP_PCA_COMPONENTS, max(1, n))
    try:
        model = SentenceTransformer(TEXT_MODEL)
        emb = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        pca = PCA(n_components=n_comp, random_state=42)
        pcs = pca.fit_transform(emb)
    except Exception:
        pcs = np.zeros((n, min(DOC_NLP_PCA_COMPONENTS, max(1, n))))

    for i in range(DOC_NLP_PCA_COMPONENTS):
        col = f"doc_nlp_pca_{i + 1}"
        if i < pcs.shape[1]:
            tender_df[col] = pcs[:, i]
        else:
            tender_df[col] = 0.0


def extract_supplier_rows(t):
    rows = []
    tender_id = t.get("tenderID") or t.get("id")
    bids = t.get("bids", [])
    awards = t.get("awards", [])
    contracts = t.get("contracts", [])

    award_suppliers = {
        a["suppliers"][0]["identifier"]["id"]: a
        for a in awards if a.get("status") == "active" and a.get("suppliers")
    }

    for b in bids:
        tenderers = b.get("tenderers") or []
        if not tenderers:
            continue
        supplier = tenderers[0]
        supplier_id = supplier.get("identifier", {}).get("id")
        if not supplier_id:
            continue

        awards_for_supplier = [a.get("value", {}).get("amount") for a in awards
                               if a.get("status") == "active"
                               and a.get("suppliers") and a["suppliers"][0]["identifier"]["id"] == supplier_id]

        contracts_for_supplier = [c.get("value", {}).get("amount") for c in contracts
                                  if c.get("suppliers") and c["suppliers"][0]["identifier"]["id"] == supplier_id]

        contract_changes = [len(c.get("changes", [])) for c in contracts
                            if c.get("suppliers") and c["suppliers"][0]["identifier"]["id"] == supplier_id]
        contract_changes_share = sum(1 for ch in contract_changes if ch > 0) / len(contract_changes) if contract_changes else 0

        items = t.get("items", [])
        price_per_unit = None
        if items:
            prices = []
            for it in items:
                q = it.get("quantity")
                v = it.get("unit", {}).get("value", {}).get("amount")
                if q and v:
                    prices.append(v / q)
            if prices:
                price_per_unit = sum(prices) / len(prices)
        if price_per_unit is None:
            price_per_unit = t.get("value", {}).get("amount")

        is_winner = supplier_id in award_suppliers
        contract_value = awards_for_supplier[0] if awards_for_supplier else None

        rows.append({
            "supplier_id": supplier_id,
            "tender_id": tender_id,
            "bid_price": b.get("value", {}).get("amount"),
            "awards_amount": awards_for_supplier,
            "contracts_amount": contracts_for_supplier,
            "contract_value": contract_value,
            "price_per_unit": price_per_unit,
            "is_winner": is_winner,
            "num_competitors": max(len(bids) - 1, 0),
            "complaints": len(t.get("complaints", [])),
            "rejected": b.get("status") == "invalid",
            "contract_changes_share": contract_changes_share
        })
    return rows


def extract_relationship_rows(t):
    rows = []
    buyer = t.get("procuringEntity") or {}
    buyer_id = buyer.get("identifier", {}).get("id")
    buyer_locality = buyer.get("address", {}).get("locality")

    for a in t.get("awards", []):
        if a.get("status") != "active" or not a.get("suppliers"):
            continue
        supplier = a["suppliers"][0]
        supplier_id = supplier.get("identifier", {}).get("id")
        supplier_locality = supplier.get("address", {}).get("locality")

        rows.append({
            "buyer_id": buyer_id,
            "supplier_id": supplier_id,
            "tender_id": t.get("tenderID") or t.get("id"),
            "cpv": t.get("classification", {}).get("id"),
            "price": a["value"]["amount"],
            "num_bids": len(t.get("bids", [])),
            "single_bid": len(t.get("bids", [])) == 1,
            "buyer_locality": buyer_locality,
            "supplier_locality": supplier_locality,
            "date": t.get("date")
        })
    return rows


if __name__ == "__main__":
    client = MongoClient("mongodb://127.0.0.1:27017/")
    db = client["prozorro_mcp"]
    tenders = db["tenders_cpv_09310000_5"]

    all_supplier_rows = []
    all_tender_rows = []
    all_relationship_rows = []
    all_doc_blobs: list[str] = []

    _http_session = requests.Session() if DOCUMENT_FETCH_ENABLED else None
    if _http_session is not None:
        _http_session.headers.setdefault(
            "User-Agent",
            "DyplomProcurementResearch/1.0 (+educational; extraction)",
        )

    for t in tenders.find():
        all_supplier_rows.extend(extract_supplier_rows(t))
        row = extract_tender_features(t, http_session=_http_session)
        all_doc_blobs.append(row.pop("_doc_blob"))
        all_tender_rows.append(row)
        all_relationship_rows.extend(extract_relationship_rows(t))

    df_sup = pd.DataFrame(all_supplier_rows)
    supplier_ds = df_sup.groupby("supplier_id").agg(
        num_bids=("tender_id", "count"),
        num_wins=("is_winner", "sum"),
        avg_competitors=("num_competitors", "mean"),
        rejected_bids=("rejected", "sum"),
        complaints=("complaints", "sum"),
        avg_price_per_unit=("price_per_unit", "mean"),
        avg_contract=("contract_value", "mean"),
        max_contract=("contract_value", "max"),
        min_contract=("contract_value", "min"),
        contract_changes_share=("contract_changes_share", "mean")
    )
    supplier_ds["win_rate"] = supplier_ds["num_wins"] / supplier_ds["num_bids"]

    tender_df = pd.DataFrame(all_tender_rows)
    cpv_median = tender_df.groupby("cpv")["value_amount"].median().to_dict()
    tender_df["median_cpv"] = tender_df["cpv"].map(cpv_median)
    tender_df["cpv_deviation"] = (tender_df["value_amount"] - tender_df["median_cpv"]).abs()

    add_document_nlp_pca_columns(tender_df, all_doc_blobs)

    rel_df = pd.DataFrame(all_relationship_rows)
    rel_ds = rel_df.groupby(["buyer_id", "supplier_id"]).agg(
        num_tenders=("tender_id", "count"),
        avg_price=("price", "mean"),
        avg_competitors=("num_bids", "mean"),
        single_bid_share=("single_bid", "mean"),
        buyer_locality=("buyer_locality", "first"),
        supplier_locality=("supplier_locality", "first")
    )

    buyer_total_tenders = rel_df.groupby("buyer_id")["tender_id"].count()
    supplier_totals = rel_df.groupby("supplier_id")["price"].sum()
    rel_ds["buyer_win_share"] = rel_ds["num_tenders"] / rel_ds.index.get_level_values(0).map(buyer_total_tenders)
    rel_ds["supplier_income_share"] = (rel_ds["avg_price"] * rel_ds["num_tenders"]) / rel_ds.index.get_level_values(1).map(supplier_totals)


    def compute_win_stats(df):
        df = df.sort_values("date")
        dates = [isoparse(d) for d in df["date"] if d]
        if not dates:
            return pd.Series({"win_regular_months": None, "win_streak": 0})
        months_between = [(dates[i] - dates[i-1]).days / 30 for i in range(1, len(dates))]
        avg_months = sum(months_between)/len(months_between) if months_between else None

        streak = 1
        max_streak = 1
        for i in range(1, len(dates)):
            if (dates[i] - dates[i-1]).days <= 60:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 1
        return pd.Series({"win_regular_months": avg_months, "win_streak": max_streak})


    win_stats = rel_df.groupby(["buyer_id", "supplier_id"]).apply(compute_win_stats)
    rel_ds = rel_ds.join(win_stats)

    for col in ["avg_contract", "max_contract", "min_contract", "avg_price_per_unit"]:
        supplier_ds[f"{col}_missing"] = supplier_ds[col].isna()
        median_value = supplier_ds[col].median()
        supplier_ds[col] = supplier_ds[col].fillna(median_value)

    supplier_ds["win_rate_missing"] = supplier_ds["win_rate"].isna()
    supplier_ds["win_rate"] = supplier_ds["win_rate"].fillna(0)

    for col in ["avg_contract", "max_contract", "min_contract", "avg_price_per_unit"]:
        supplier_ds[f"log_{col}"] = np.log1p(supplier_ds[col])

    tender_numeric_cols = [
        "value_amount", "winner_price", "price_per_unit",
        "value_per_day", "min_bid", "winner_minus_min",
        "tender_duration_days", "completion_days", "cpv_deviation",
        "doc_nlp_pca_1", "doc_nlp_pca_2", "doc_nlp_pca_3", "doc_nlp_pca_4", "doc_nlp_pca_5",
    ]

    tender_df["description_missing"] = tender_df["description"].isna() | (tender_df["description"].str.strip() == "")
    tender_df["description"] = tender_df["description"].fillna("НЕВІДОМО")
    tender_df["description"] = tender_df["description"].replace("", "НЕВІДОМО")

    for col in tender_numeric_cols:
        tender_df[f"{col}_missing"] = tender_df[col].isna()
        median_value = tender_df[col].median()
        tender_df[col] = tender_df[col].fillna(median_value)

    for col in ["value_amount", "winner_price", "price_per_unit", "value_per_day", "min_bid"]:
        tender_df[f"log_{col}"] = np.log1p(tender_df[col])

    tender_df["doc_text_features_missing"] = tender_df["doc_text_features_missing"].fillna(1.0)

    for col in ["buyer_locality", "supplier_locality"]:
        rel_ds[col] = rel_ds[col].fillna("НЕВІДОМО")
        rel_ds[col] = rel_ds[col].replace("", "НЕВІДОМО")


    for col in ["win_regular_months", "win_streak"]:
        rel_ds[f"{col}_missing"] = rel_ds[col].isna()
        rel_ds[col] = rel_ds[col].fillna(0)

    tender_df.to_csv(f"data/tender_level_features.csv", index=False, encoding="utf-8")
    supplier_ds.reset_index().to_csv(f"data/supplier_level_features.csv", index=False, encoding="utf-8")
    rel_ds.reset_index().to_csv(f"data/relationship_level_features.csv", index=False, encoding="utf-8")
