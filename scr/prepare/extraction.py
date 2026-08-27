"""
extraction.py — витягування та конструювання ознак із сирих JSON-тендерів Prozorro.

Принцип інкапсуляції:
  - extract_tender_features(t, cpv_medians)  → повністю готовий рядок тендеру
  - extract_supplier_rows(t)                 → сирі рядки по кожному біду (для агрегації)
  - aggregate_supplier_features(rows)        → повністю готовий DataFrame підрядників
  - extract_relationship_rows(t)             → рядки зв'язків замовник–підрядник

Постобробка (fillna, log, missing flags) НЕ виноситься назовні —
вона виконується всередині кожної функції.

Контекст дипломної методики:
  - Тендерний рівень: фінансові/часові/структурні ознаки для numeric anomaly detection.
  - Рівень підрядника: агрегати поведінки постачальника для K-Means/IF/LOF.
  - Рівень зв'язків: індикатори концентрації перемог і конкуренції buyer-supplier.
"""

from dateutil.parser import isoparse
from pymongo import MongoClient
import pandas as pd
from datetime import datetime
import numpy as np

import requests

try:
    from .document_extraction import (
        DOCUMENT_FETCH_ENABLED,
        summarize_tier1_for_documents,
        tier1_placeholder_meta_and_blob,
    )
except ImportError:
    from document_extraction import (
        DOCUMENT_FETCH_ENABLED,
        summarize_tier1_for_documents,
        tier1_placeholder_meta_and_blob,
    )

FIXED_NOW = datetime(2025, 12, 11)

# ──────────────────────────────────────────────────────────────────────────────
# Внутрішні допоміжні функції
# ──────────────────────────────────────────────────────────────────────────────

def _safe_log(value) -> float:
    """log1p із захистом від None та від'ємних значень."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    return float(np.log1p(max(float(value), 0.0)))


def _missing(value) -> bool:
    """True якщо значення відсутнє."""
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return False


def _fillna(value, default=0.0):
    """Повертає default якщо value відсутнє."""
    return default if _missing(value) else value


# ──────────────────────────────────────────────────────────────────────────────
# Тендерний рівень
# ──────────────────────────────────────────────────────────────────────────────

def extract_tender_features(
    t: dict,
    cpv_medians: dict | None = None,
    http_session=None,
) -> dict:
    """
    Повертає повністю готовий рядок ознак тендеру:
      - сирі числові поля
      - log-трансформації
      - missing-прапори
      - cpv_deviation (якщо передано cpv_medians)
      - description_missing

    Parameters
    ----------
    t           : сирий JSON-документ тендеру з MongoDB
    cpv_medians : dict {cpv_code: median_value} — потрібен для cpv_deviation;
                  рахується один раз зовні по всій/батч-вибірці
    http_session: requests.Session для завантаження документів (опційно)
    """

    # ── Ідентифікатори та текстові поля ──────────────────────────────────────
    tender_id   = t.get("tenderID")
    status      = t.get("status")
    title       = t.get("title")
    description = t.get("description")

    description_missing = _missing(description) or str(description).strip() == ""
    description = "НЕВІДОМО" if description_missing else description

    items = t.get("items", [])
    cpv = items[0].get("classification", {}).get("id") if items else None

    # ── Фінансові поля ───────────────────────────────────────────────────────
    value_amount = t.get("value", {}).get("amount")
    value_amount_missing = _missing(value_amount)
    value_amount = _fillna(value_amount)

    # ── Переможець ───────────────────────────────────────────────────────────
    awards = [a for a in t.get("awards", []) if a.get("status") == "active"]
    winner_price = awards[0]["value"]["amount"] if awards else None
    winner_price_missing = _missing(winner_price)
    winner_price = _fillna(winner_price)

    # ── Ціна за одиницю ──────────────────────────────────────────────────────
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
    price_per_unit_missing = _missing(price_per_unit)
    price_per_unit = _fillna(price_per_unit)

    # ── Тривалість тендеру ───────────────────────────────────────────────────
    start = t.get("tenderPeriod", {}).get("startDate")
    end   = t.get("tenderPeriod", {}).get("endDate")

    tender_duration_days = None
    if start:
        start_dt = isoparse(start)
        end_dt   = isoparse(end) if end else FIXED_NOW
        tender_duration_days = (end_dt - start_dt).days

    completion_days = tender_duration_days  # поки що те саме

    tender_duration_days_missing = _missing(tender_duration_days)
    completion_days_missing      = _missing(completion_days)
    tender_duration_days = _fillna(tender_duration_days)
    completion_days      = _fillna(completion_days)

    # ── Вартість за день ─────────────────────────────────────────────────────
    value_per_day = (
        value_amount / tender_duration_days
        if value_amount and tender_duration_days
        else None
    )
    value_per_day_missing = _missing(value_per_day)
    value_per_day = _fillna(value_per_day)

    # ── Ставки ───────────────────────────────────────────────────────────────
    bids        = t.get("bids", [])
    active_bids = [b for b in bids if b.get("status") == "active"]
    bid_prices  = [
        b.get("value", {}).get("amount")
        for b in active_bids
        if b.get("value", {}).get("amount") is not None
    ]

    min_bid = min(bid_prices) if bid_prices else None
    min_bid_missing = _missing(min_bid)
    min_bid = _fillna(min_bid)

    winner_minus_min = (
        winner_price - min(bid_prices)
        if winner_price and bid_prices
        else None
    )
    winner_minus_min_missing = _missing(winner_minus_min)
    winner_minus_min = _fillna(winner_minus_min)

    # ── CPV deviation ────────────────────────────────────────────────────────
    median_cpv = cpv_medians.get(cpv) if (cpv_medians and cpv) else None
    cpv_deviation = (
        abs(value_amount - median_cpv) if not _missing(median_cpv) else None
    )
    cpv_deviation_missing = _missing(cpv_deviation)
    cpv_deviation = _fillna(cpv_deviation)

    # ── Лічильники подій ─────────────────────────────────────────────────────
    complaints       = len(t.get("complaints", []))
    tender_changes   = len(t.get("amendments", []))
    contract_changes = sum(
        len(c.get("changes", [])) for c in t.get("contracts", [])
    )

    # ── Документи (опційно) ──────────────────────────────────────────────────
    docs = t.get("documents") or []
    if DOCUMENT_FETCH_ENABLED:
        doc_t1, doc_blob = summarize_tier1_for_documents(docs, session=http_session)
    else:
        doc_t1, doc_blob = tier1_placeholder_meta_and_blob()

    # ── Log-трансформації ────────────────────────────────────────────────────
    log_value_amount   = _safe_log(value_amount)
    log_winner_price   = _safe_log(winner_price)
    log_price_per_unit = _safe_log(price_per_unit)
    log_value_per_day  = _safe_log(value_per_day)
    log_min_bid        = _safe_log(min_bid)

    # ── Збірка рядка ─────────────────────────────────────────────────────────
    row = {
        # Ідентифікатори
        "tender_id":   tender_id,
        "cpv":         cpv,
        "status":      status,
        "title":       title,
        "description": description,
        # Log-трансформовані фінансові ознаки
        "log_value_amount":   log_value_amount,
        "log_winner_price":   log_winner_price,
        "log_price_per_unit": log_price_per_unit,
        "log_value_per_day":  log_value_per_day,
        # Ставки
        "numberOfBids":     len(bids),
        "numberOfAdmitted": len(active_bids),
        "log_min_bid":      log_min_bid,
        "winner_minus_min": winner_minus_min,
        # Часові
        "tender_duration_days": tender_duration_days,
        "completion_days":      completion_days,
        # Лічильники подій
        "complaints":       complaints,
        "tender_changes":   tender_changes,
        "contract_changes": contract_changes,
        # CPV deviation
        "median_cpv":    median_cpv,
        "cpv_deviation": cpv_deviation,
        # Missing flags
        "description_missing":           description_missing,
        "value_amount_missing":          value_amount_missing,
        "winner_price_missing":          winner_price_missing,
        "price_per_unit_missing":        price_per_unit_missing,
        "value_per_day_missing":         value_per_day_missing,
        "min_bid_missing":               min_bid_missing,
        "winner_minus_min_missing":      winner_minus_min_missing,
        "tender_duration_days_missing":  tender_duration_days_missing,
        "completion_days_missing":       completion_days_missing,
        "cpv_deviation_missing":         cpv_deviation_missing,
    }

    # Документні мета-поля (якщо увімкнено)
    row.update(doc_t1)
    row["_doc_blob"] = doc_blob  # тимчасове поле для NLP, видаляється зовні

    return row


# ──────────────────────────────────────────────────────────────────────────────
# Рівень підрядника
# ──────────────────────────────────────────────────────────────────────────────

def extract_supplier_rows(t: dict) -> list[dict]:
    """
    Повертає список сирих рядків — по одному на кожен bid у тендері.
    Ці рядки потім агрегуються через aggregate_supplier_features().
    """
    rows = []
    tender_id  = t.get("tenderID") or t.get("id")
    bids       = t.get("bids", [])
    awards     = t.get("awards", [])
    contracts  = t.get("contracts", [])

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

    for b in bids:
        tenderers = b.get("tenderers") or []
        if not tenderers:
            continue
        supplier_id = tenderers[0].get("identifier", {}).get("id")
        if not supplier_id:
            continue

        awards_for_supplier = [
            a.get("value", {}).get("amount")
            for a in awards
            if a.get("status") == "active"
            and a.get("suppliers")
            and a["suppliers"][0]["identifier"]["id"] == supplier_id
        ]

        contracts_for_supplier = [
            c for c in contracts
            if c.get("suppliers")
            and c["suppliers"][0]["identifier"]["id"] == supplier_id
        ]
        contract_changes_list = [
            len(c.get("changes", [])) for c in contracts_for_supplier
        ]
        contract_changes_share = (
            sum(1 for ch in contract_changes_list if ch > 0) / len(contract_changes_list)
            if contract_changes_list else 0.0
        )

        contract_value = awards_for_supplier[0] if awards_for_supplier else None
        is_winner      = bool(awards_for_supplier)

        rows.append({
            "supplier_id":            supplier_id,
            "tender_id":              tender_id,
            "bid_price":              b.get("value", {}).get("amount"),
            "contract_value":         contract_value,
            "price_per_unit":         price_per_unit,
            "is_winner":              is_winner,
            "num_competitors":        max(len(bids) - 1, 0),
            "complaints":             len(t.get("complaints", [])),
            "rejected":               b.get("status") == "invalid",
            "contract_changes_share": contract_changes_share,
        })
    return rows


def aggregate_supplier_features(supplier_rows: list[dict]) -> pd.DataFrame:
    """
    Приймає сирі рядки з extract_supplier_rows(), повертає повністю готовий
    DataFrame підрядників із агрегованими ознаками, log-трансформаціями та win_rate.

    Відповідає схемі:
        supplier_id, num_bids, num_wins, avg_competitors, rejected_bids,
        complaints, log_avg_price_per_unit, log_avg_contract,
        contract_changes_share, win_rate, log_max_contract, log_min_contract
    """
    if not supplier_rows:
        return pd.DataFrame()

    df = pd.DataFrame(supplier_rows)

    agg = df.groupby("supplier_id").agg(
        num_bids               = ("tender_id",              "count"),
        num_wins               = ("is_winner",              "sum"),
        avg_competitors        = ("num_competitors",        "mean"),
        rejected_bids          = ("rejected",               "sum"),
        complaints             = ("complaints",             "sum"),
        avg_price_per_unit     = ("price_per_unit",         "mean"),
        avg_contract           = ("contract_value",         "mean"),
        max_contract           = ("contract_value",         "max"),
        min_contract           = ("contract_value",         "min"),
        contract_changes_share = ("contract_changes_share", "mean"),
    )

    # ── win_rate ─────────────────────────────────────────────────────────────
    agg["win_rate"] = (agg["num_wins"] / agg["num_bids"]).fillna(0.0)

    # ── Log-трансформації (fillna(0) перед log щоб не втрачати рядки) ───────
    agg["log_avg_price_per_unit"] = np.log1p(agg["avg_price_per_unit"].fillna(0.0))
    agg["log_avg_contract"]       = np.log1p(agg["avg_contract"].fillna(0.0))
    agg["log_max_contract"]       = np.log1p(agg["max_contract"].fillna(0.0))
    agg["log_min_contract"]       = np.log1p(agg["min_contract"].fillna(0.0))

    # ── Прибираємо сирі колонки які замінені log-версіями ────────────────────
    agg = agg.drop(columns=["avg_price_per_unit", "avg_contract",
                             "max_contract", "min_contract"])

    # ── Порядок колонок відповідно до схеми ──────────────────────────────────
    ordered_cols = [
        "num_bids",
        "num_wins",
        "avg_competitors",
        "rejected_bids",
        "complaints",
        "log_avg_price_per_unit",
        "log_avg_contract",
        "contract_changes_share",
        "win_rate",
        "log_max_contract",
        "log_min_contract",
    ]
    agg = agg[ordered_cols]

    return agg.reset_index()


# ──────────────────────────────────────────────────────────────────────────────
# Рівень зв'язків замовник–підрядник
# ──────────────────────────────────────────────────────────────────────────────

def extract_relationship_rows(t: dict) -> list[dict]:
    """Повертає рядки зв'язків buyer–supplier для кожного активного award."""
    rows  = []
    buyer = t.get("procuringEntity") or {}
    buyer_id       = buyer.get("identifier", {}).get("id")
    buyer_locality = buyer.get("address", {}).get("locality")

    for a in t.get("awards", []):
        if a.get("status") != "active" or not a.get("suppliers"):
            continue
        supplier          = a["suppliers"][0]
        supplier_id       = supplier.get("identifier", {}).get("id")
        supplier_locality = supplier.get("address", {}).get("locality")

        rows.append({
            "buyer_id":          buyer_id,
            "supplier_id":       supplier_id,
            "tender_id":         t.get("tenderID") or t.get("id"),
            "cpv":               t.get("classification", {}).get("id"),
            "price":             a["value"]["amount"],
            "num_bids":          len(t.get("bids", [])),
            "single_bid":        len(t.get("bids", [])) == 1,
            "buyer_locality":    buyer_locality,
            "supplier_locality": supplier_locality,
            "date":              t.get("date"),
        })
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Точка входу (одноразовий запуск)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    client   = MongoClient("mongodb://127.0.0.1:27017/")
    db       = client["prozorro_mcp"]
    tenders  = db["tenders_cpv_09310000_5"]

    _http_session = requests.Session() if DOCUMENT_FETCH_ENABLED else None
    if _http_session:
        _http_session.headers["User-Agent"] = (
            "DyplomProcurementResearch/1.0 (+educational; extraction)"
        )

    # ── Крок 1: зібрати сирі рядки ───────────────────────────────────────────
    all_supplier_rows:    list[dict] = []
    all_relationship_rows: list[dict] = []
    raw_tender_rows:      list[dict] = []
    all_doc_blobs:        list[str]  = []

    for t in tenders.find():
        all_supplier_rows.extend(extract_supplier_rows(t))
        all_relationship_rows.extend(extract_relationship_rows(t))

        # Тендерні ознаки без cpv_medians — додамо після
        row = extract_tender_features(t, cpv_medians=None, http_session=_http_session)
        all_doc_blobs.append(row.pop("_doc_blob"))
        raw_tender_rows.append(row)

    # ── Крок 2: порахувати cpv_medians по всій вибірці ───────────────────────
    tender_df  = pd.DataFrame(raw_tender_rows)

    # Відновлюємо сирий value_amount із log для підрахунку медіани
    tender_df["_value_amount_raw"] = np.expm1(tender_df["log_value_amount"])
    cpv_medians = tender_df.groupby("cpv")["_value_amount_raw"].median().to_dict()

    # Оновлюємо median_cpv та cpv_deviation вже з правильними медіанами
    tender_df["median_cpv"]    = tender_df["cpv"].map(cpv_medians)
    tender_df["cpv_deviation"] = (
        tender_df["_value_amount_raw"] - tender_df["median_cpv"]
    ).abs().fillna(0.0)
    tender_df["cpv_deviation_missing"] = tender_df["median_cpv"].isna()
    tender_df = tender_df.drop(columns=["_value_amount_raw"])

    # ── Крок 3: NLP PCA (опційно) ────────────────────────────────────────────
    if DOCUMENT_FETCH_ENABLED:
        from extraction import add_document_nlp_pca_columns
        add_document_nlp_pca_columns(tender_df, all_doc_blobs)

    # ── Крок 4: Агрегація підрядників ────────────────────────────────────────
    supplier_df = aggregate_supplier_features(all_supplier_rows)

    # ── Крок 5: Агрегація зв'язків ───────────────────────────────────────────
    from dateutil.parser import isoparse as _isoparse

    rel_df = pd.DataFrame(all_relationship_rows)
    rel_ds = rel_df.groupby(["buyer_id", "supplier_id"]).agg(
        num_tenders        = ("tender_id",   "count"),
        avg_price          = ("price",        "mean"),
        avg_competitors    = ("num_bids",     "mean"),
        single_bid_share   = ("single_bid",   "mean"),
        buyer_locality     = ("buyer_locality",    "first"),
        supplier_locality  = ("supplier_locality", "first"),
    )
    buyer_total   = rel_df.groupby("buyer_id")["tender_id"].count()
    supplier_totals = rel_df.groupby("supplier_id")["price"].sum()
    rel_ds["buyer_win_share"] = (
        rel_ds["num_tenders"]
        / rel_ds.index.get_level_values(0).map(buyer_total)
    )
    rel_ds["supplier_income_share"] = (
        rel_ds["avg_price"] * rel_ds["num_tenders"]
    ) / rel_ds.index.get_level_values(1).map(supplier_totals)

    for col in ["buyer_locality", "supplier_locality"]:
        rel_ds[col] = rel_ds[col].fillna("НЕВІДОМО").replace("", "НЕВІДОМО")

    # ── Крок 6: Збереження ───────────────────────────────────────────────────
    tender_df.to_csv("../../data/raw/tender_level_features.csv",
                     index=False, encoding="utf-8")
    supplier_df.to_csv("../../data/raw/supplier_level_features.csv",
                       index=False, encoding="utf-8")
    rel_ds.reset_index().to_csv("../../data/raw/relationship_level_features.csv",
                                index=False, encoding="utf-8")

    print("Готово.")
    print(f"  Тендери:    {len(tender_df)}")
    print(f"  Підрядники: {len(supplier_df)}")
    print(f"  Зв'язки:    {len(rel_ds)}")