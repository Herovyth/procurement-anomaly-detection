"""
batch_extraction.py — батчева обробка тендерів Prozorro із checkpoint-відновленням.

Використовує інкапсульовані функції з extraction.py:
  - extract_tender_features(t, cpv_medians, http_session) → готовий рядок
  - extract_supplier_rows(t)                              → сирі рядки бідів
  - aggregate_supplier_features(rows)                    → готовий DataFrame
  - extract_relationship_rows(t)                         → рядки зв'язків

Структура батчу:
  1. Зібрати сирі рядки тендерів (без cpv_medians)
  2. Порахувати cpv_medians по батчу
  3. Оновити median_cpv / cpv_deviation у вже зібраному DataFrame
  4. NLP PCA (якщо увімкнено)
  5. Скинути в CSV
  6. Supplier / Rel — скидаються кожні SUPPLIER_FLUSH_EVERY батчів

Навіщо в дипломній системі:
  - Масштабовано готувати ознаки для десятків тисяч тендерів без втрати прогресу.
  - Розділити pipeline на 3 рівні (тендер / підрядник / зв'язок) для подальших
    незалежних моделей аномалій.
"""
import os
import sys
import time
import pandas as pd
import warnings

import numpy as np
from pymongo import MongoClient
import requests
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(all="ignore")

try:
    from .document_extraction import (
        DOCUMENT_FETCH_ENABLED,
        DOC_NLP_PCA_COMPONENTS,
        TEXT_MODEL,
    )
except ImportError:
    from document_extraction import (
        DOCUMENT_FETCH_ENABLED,
        DOC_NLP_PCA_COMPONENTS,
        TEXT_MODEL,
    )
try:
    from .extraction import (
        aggregate_supplier_features,
        extract_relationship_rows,
        extract_supplier_rows,
        extract_tender_features,
    )
except ImportError:
    from extraction import (
        aggregate_supplier_features,
        extract_relationship_rows,
        extract_supplier_rows,
        extract_tender_features,
    )

# ── Налаштування ──────────────────────────────────────────────────────────────
BATCH_SIZE            = 1000
SUPPLIER_FLUSH_EVERY  = 10       # скидати supplier/rel кожні N батчів
NLP_BATCH_SIZE        = 512

CHECKPOINT_FILE = "../../data/raw/checkpoint.txt"
OUTPUT_TENDER   = "../../data/raw/tender_level_features.csv"
OUTPUT_SUPPLIER = "../../data/raw/supplier_level_features.csv"
OUTPUT_REL      = "../../data/raw/relationship_level_features.csv"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs("../../data/raw", exist_ok=True)


# ── Checkpoint ────────────────────────────────────────────────────────────────

def load_checkpoint() -> str | None:
    if os.path.exists(CHECKPOINT_FILE):
        val = open(CHECKPOINT_FILE).read().strip()
        return val or None
    return None


def save_checkpoint(last_id: str) -> None:
    with open(CHECKPOINT_FILE, "w") as f:
        f.write(str(last_id))


# ── CSV append ────────────────────────────────────────────────────────────────

def append_to_csv(df: pd.DataFrame, path: str) -> None:
    """Дописує DataFrame до CSV; заголовок тільки якщо файл новий."""
    write_header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=write_header, index=False, encoding="utf-8")


# ── Прогрес-бар ───────────────────────────────────────────────────────────────

def print_progress(
    processed: int,
    total: int,
    batch_num: int,
    docs_found: int,
    docs_downloaded: int,
    elapsed: float,
    eta_sec: float | None,
) -> None:
    pct      = processed / total * 100 if total else 0
    bar_len  = 25
    filled   = int(bar_len * processed / total) if total else 0
    bar      = "█" * filled + "░" * (bar_len - filled)

    doc_pct    = (docs_downloaded / docs_found * 100) if docs_found > 0 else 0.0
    doc_filled = int(20 * docs_downloaded / docs_found) if docs_found > 0 else 0
    doc_bar    = "█" * doc_filled + "░" * (20 - doc_filled)

    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    eta_str     = time.strftime("%H:%M:%S", time.gmtime(eta_sec)) if eta_sec else "--:--:--"

    sys.stdout.write(
        f"\r Тендери [{bar}] {processed:>6}/{total} ({pct:5.1f}%) | "
        f"Батч #{batch_num} | "
        f"Документи [{doc_bar}] {doc_pct:5.1f}% | "
        f"Час: {elapsed_str} | Залишилось: {eta_str}"
    )
    sys.stdout.flush()


# ── NLP PCA по батчу ──────────────────────────────────────────────────────────

def apply_nlp_pca(
    tender_df: pd.DataFrame,
    doc_blobs: list[str],
    nlp_model,
) -> pd.DataFrame:
    """
    Додає колонки doc_nlp_pca_1..N до tender_df (in-place повернення).
    Якщо DOCUMENT_FETCH_ENABLED=False або model=None — заповнює нулями.
    """
    from sklearn.decomposition import PCA

    n = len(tender_df)
    for i in range(1, DOC_NLP_PCA_COMPONENTS + 1):
        tender_df[f"doc_nlp_pca_{i}"] = 0.0

    if n < 2 or not DOCUMENT_FETCH_ENABLED or nlp_model is None:
        return tender_df

    texts  = [b if (b and str(b).strip()) else " " for b in doc_blobs]
    n_comp = min(DOC_NLP_PCA_COMPONENTS, n)

    try:
        all_emb = []
        for i in range(0, len(texts), NLP_BATCH_SIZE):
            chunk = texts[i: i + NLP_BATCH_SIZE]
            all_emb.append(
                nlp_model.encode(chunk, batch_size=32,
                                 show_progress_bar=False,
                                 normalize_embeddings=True)
            )
        emb = np.vstack(all_emb)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pcs = PCA(n_components=n_comp, random_state=42).fit_transform(emb)
    except Exception as e:
        print(f"\n[NLP] Помилка: {e}", flush=True)
        pcs = np.zeros((n, n_comp))

    for i in range(DOC_NLP_PCA_COMPONENTS):
        tender_df[f"doc_nlp_pca_{i + 1}"] = pcs[:, i] if i < pcs.shape[1] else 0.0

    return tender_df


# ── Агрегація зв'язків ────────────────────────────────────────────────────────

def aggregate_relationship_features(rel_rows: list[dict]) -> pd.DataFrame:
    """Агрегує рядки зв'язків buyer–supplier у готовий DataFrame."""
    if not rel_rows:
        return pd.DataFrame()

    rel_df  = pd.DataFrame(rel_rows)
    rel_ds  = rel_df.groupby(["buyer_id", "supplier_id"]).agg(
        num_tenders       = ("tender_id",          "count"),
        avg_price         = ("price",               "mean"),
        avg_competitors   = ("num_bids",            "mean"),
        single_bid_share  = ("single_bid",          "mean"),
        buyer_locality    = ("buyer_locality",      "first"),
        supplier_locality = ("supplier_locality",   "first"),
    )

    buyer_total      = rel_df.groupby("buyer_id")["tender_id"].count()
    supplier_totals  = rel_df.groupby("supplier_id")["price"].sum()

    rel_ds["buyer_win_share"] = (
        rel_ds["num_tenders"]
        / rel_ds.index.get_level_values(0).map(buyer_total)
    )
    rel_ds["supplier_income_share"] = (
        rel_ds["avg_price"] * rel_ds["num_tenders"]
    ) / rel_ds.index.get_level_values(1).map(supplier_totals)

    for col in ["buyer_locality", "supplier_locality"]:
        rel_ds[col] = rel_ds[col].fillna("НЕВІДОМО").replace("", "НЕВІДОМО")

    return rel_ds.reset_index()


# ── Обробка одного батчу тендерів ─────────────────────────────────────────────

def process_tender_batch(
    batch: list[dict],
    nlp_model,
    http_session,
) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    """
    Приймає список сирих тендерів одного батчу.
    Повертає:
      - tender_df    : готовий DataFrame тендерних ознак
      - supplier_rows: сирі рядки для подальшої агрегації
      - rel_rows     : сирі рядки зв'язків

    Кроки всередині:
      1. extract_tender_features без cpv_medians (→ median_cpv/cpv_deviation = None/0)
      2. Рахуємо cpv_medians по батчу
      3. Оновлюємо median_cpv / cpv_deviation у DataFrame
      4. NLP PCA
      5. Видаляємо службові колонки
    """
    tender_rows:   list[dict] = []
    supplier_rows: list[dict] = []
    rel_rows:      list[dict] = []
    doc_blobs:     list[str]  = []

    for t in batch:
        row = extract_tender_features(t, cpv_medians=None, http_session=http_session)
        doc_blobs.append(row.pop("_doc_blob"))
        tender_rows.append(row)
        supplier_rows.extend(extract_supplier_rows(t))
        rel_rows.extend(extract_relationship_rows(t))

    tender_df = pd.DataFrame(tender_rows)

    # ── cpv_medians по батчу ─────────────────────────────────────────────────
    # Відновлюємо сирий value_amount із log для підрахунку медіани
    tender_df["_value_raw"] = np.expm1(tender_df["log_value_amount"])
    cpv_medians = tender_df.groupby("cpv")["_value_raw"].median().to_dict()

    tender_df["median_cpv"]    = tender_df["cpv"].map(cpv_medians)
    tender_df["cpv_deviation"] = (
        tender_df["_value_raw"] - tender_df["median_cpv"]
    ).abs().fillna(0.0)
    tender_df["cpv_deviation_missing"] = tender_df["median_cpv"].isna()
    tender_df = tender_df.drop(columns=["_value_raw"])

    # ── NLP PCA ──────────────────────────────────────────────────────────────
    tender_df = apply_nlp_pca(tender_df, doc_blobs, nlp_model)

    # ── Видаляємо службові doc-колонки ───────────────────────────────────────
    drop_cols = (
        [f"doc_nlp_pca_{i}"         for i in range(1, DOC_NLP_PCA_COMPONENTS + 1)]
        + [f"doc_nlp_pca_{i}_missing" for i in range(1, DOC_NLP_PCA_COMPONENTS + 1)]
        + [
            "documents_count",
            "documents_distinct_types",
            "documents_with_url",
            "documents_format_pdf",
            "documents_downloaded_count",
            "documents_download_bytes",
            "doc_text_features_missing",
        ]
    )
    tender_df = tender_df.drop(
        columns=[c for c in drop_cols if c in tender_df.columns],
        errors="ignore",
    )

    return tender_df, supplier_rows, rel_rows


# ── Головна функція ───────────────────────────────────────────────────────────

def run() -> None:
    client      = MongoClient("mongodb://127.0.0.1:27017/")
    db          = client["prozorro_mcp"]
    tenders_col = db["tenders"]

    total      = tenders_col.estimated_document_count()
    checkpoint = load_checkpoint()

    # ── Завантаження NLP моделі ───────────────────────────────────────────────
    nlp_model = None
    if DOCUMENT_FETCH_ENABLED:
        print("Завантаження SentenceTransformer...", flush=True)
        from sentence_transformers import SentenceTransformer
        nlp_model = SentenceTransformer(TEXT_MODEL)
        print("Модель завантажена.", flush=True)

    # ── Checkpoint ────────────────────────────────────────────────────────────
    query        = {}
    already_done = 0
    if checkpoint:
        from bson import ObjectId
        query        = {"_id": {"$gt": ObjectId(checkpoint)}}
        already_done = tenders_col.count_documents({"_id": {"$lte": ObjectId(checkpoint)}})
        print(f"Продовження з checkpoint. Вже оброблено: {already_done}/{total}")
    else:
        print(f"Початок з нуля. Всього тендерів: {total}")

    # ── HTTP сесія ────────────────────────────────────────────────────────────
    http_session = None
    if DOCUMENT_FETCH_ENABLED:
        http_session = requests.Session()
        http_session.headers["User-Agent"] = (
            "ProcurementResearch/1.0 (+educational; extraction)"
        )

    # ── Стан циклу ────────────────────────────────────────────────────────────
    processed   = already_done
    batch_num   = already_done // BATCH_SIZE
    start_time  = time.time()
    last_id     = None

    accumulated_supplier_rows: list[dict] = []
    accumulated_rel_rows:      list[dict] = []

    total_docs_found      = 0
    total_docs_downloaded = 0

    batch_tenders: list[dict] = []
    cursor = tenders_col.find(query).sort("_id", 1)

    # ── Головний цикл ─────────────────────────────────────────────────────────
    for t in cursor:
        last_id = t["_id"]
        batch_tenders.append(t)

        docs_in_t = len(t.get("documents") or [])
        total_docs_found += docs_in_t
        if DOCUMENT_FETCH_ENABLED:
            total_docs_downloaded += min(docs_in_t, 1)

        if len(batch_tenders) < BATCH_SIZE:
            continue

        # ── Обробка батчу ─────────────────────────────────────────────────────
        batch_num += 1

        tender_df, sup_rows, rel_rows = process_tender_batch(
            batch_tenders, nlp_model, http_session
        )
        accumulated_supplier_rows.extend(sup_rows)
        accumulated_rel_rows.extend(rel_rows)

        append_to_csv(tender_df, OUTPUT_TENDER)

        # ── Скидаємо supplier/rel кожні N батчів ─────────────────────────────
        if batch_num % SUPPLIER_FLUSH_EVERY == 0:
            if accumulated_supplier_rows:
                supplier_df = aggregate_supplier_features(accumulated_supplier_rows)
                append_to_csv(supplier_df, OUTPUT_SUPPLIER)
                accumulated_supplier_rows = []

            if accumulated_rel_rows:
                rel_df = aggregate_relationship_features(accumulated_rel_rows)
                append_to_csv(rel_df, OUTPUT_REL)
                accumulated_rel_rows = []

        batch_tenders.clear()
        processed += BATCH_SIZE
        save_checkpoint(str(last_id))

        elapsed = time.time() - start_time
        rate    = (processed - already_done) / elapsed if elapsed > 0 else 0
        eta     = (total - processed) / rate if rate > 0 else None
        print_progress(processed, total, batch_num,
                       total_docs_found, total_docs_downloaded,
                       elapsed, eta)

    # ── Останній неповний батч ────────────────────────────────────────────────
    if batch_tenders:
        batch_num += 1
        tender_df, sup_rows, rel_rows = process_tender_batch(
            batch_tenders, nlp_model, http_session
        )
        accumulated_supplier_rows.extend(sup_rows)
        accumulated_rel_rows.extend(rel_rows)
        append_to_csv(tender_df, OUTPUT_TENDER)
        processed += len(batch_tenders)
        save_checkpoint(str(last_id))

    # ── Фінальний скид supplier / rel ────────────────────────────────────────
    if accumulated_supplier_rows:
        supplier_df = aggregate_supplier_features(accumulated_supplier_rows)
        append_to_csv(supplier_df, OUTPUT_SUPPLIER)

    if accumulated_rel_rows:
        rel_df = aggregate_relationship_features(accumulated_rel_rows)
        append_to_csv(rel_df, OUTPUT_REL)

    elapsed = time.time() - start_time
    print(f"\n\nОброблено {processed} тендерів "
          f"за {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
    print(f"   → {OUTPUT_TENDER}")
    print(f"   → {OUTPUT_SUPPLIER}")
    print(f"   → {OUTPUT_REL}")


if __name__ == "__main__":
    run()