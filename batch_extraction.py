import sys
import time
import pandas as pd
import numpy as np
from pymongo import MongoClient
import requests

from document_extraction import (
    DOCUMENT_FETCH_ENABLED,
    DOC_NLP_PCA_COMPONENTS,
    TEXT_MODEL,
    summarize_tier1_for_documents,
    tier1_placeholder_meta_and_blob,
)
from extraction import (
    extract_tender_features,
    extract_supplier_rows,
    extract_relationship_rows,
    add_document_nlp_pca_columns,
    FIXED_NOW,
)

import numpy as np
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(all="ignore")

# ─── Налаштування ───────────────────────────────────────────
BATCH_SIZE        = 1000     # тендерів за один батч
CHECKPOINT_FILE   = "data/checkpoint.txt"  # останній оброблений _id
OUTPUT_TENDER     = "data/tender_level_features.csv"
OUTPUT_SUPPLIER   = "data/supplier_level_features.csv"
OUTPUT_REL        = "data/relationship_level_features.csv"
NLP_BATCH_SIZE    = 512       # скільки текстів за раз у SentenceTransformer
# ────────────────────────────────────────────────────────────

import os
os.makedirs("data", exist_ok=True)


def load_checkpoint() -> str | None:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            val = f.read().strip()
            return val if val else None
    return None


def save_checkpoint(last_id: str):
    with open(CHECKPOINT_FILE, "w") as f:
        f.write(str(last_id))


def append_to_csv(df: pd.DataFrame, path: str):
    """Дописує DataFrame до CSV. Заголовок тільки якщо файл новий."""
    write_header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=write_header, index=False, encoding="utf-8")


def print_progress(
    processed: int,
    total: int,
    batch_num: int,
    docs_found: int,
    docs_downloaded: int,
    elapsed: float,
    eta_sec: float | None,
):
    pct = processed / total * 100 if total else 0
    bar_len = 25
    filled = int(bar_len * processed / total) if total else 0
    bar = "█" * filled + "░" * (bar_len - filled)

    doc_pct = (docs_downloaded / docs_found * 100) if docs_found > 0 else 0.0
    doc_bar_len = 20
    doc_filled = int(doc_bar_len * docs_downloaded / docs_found) if docs_found > 0 else 0
    doc_bar = "█" * doc_filled + "░" * (doc_bar_len - doc_filled)

    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_sec)) if eta_sec else "--:--:--"

    sys.stdout.write(
        f"\r Тендери [{bar}] {processed:>6}/{total} ({pct:5.1f}%) | "
        f"Батч #{batch_num} | "
        f"Документи [{doc_bar}] {doc_pct:5.1f}% | "
        f"Час: {elapsed_str} | Залишилось: {eta_str}"
    )
    sys.stdout.flush()


def process_batch_nlp(tender_df: pd.DataFrame, doc_blobs: list[str], model=None) -> pd.DataFrame:
    from sklearn.decomposition import PCA
    import warnings

    n = len(tender_df)
    for i in range(1, DOC_NLP_PCA_COMPONENTS + 1):
        tender_df[f"doc_nlp_pca_{i}"] = 0.0

    # PCA потребує мінімум 2 зразки
    if n < 2 or not DOCUMENT_FETCH_ENABLED or model is None:
        return tender_df

    texts = [b if (b and str(b).strip()) else " " for b in doc_blobs]
    n_comp = min(DOC_NLP_PCA_COMPONENTS, max(1, n))

    try:
        all_emb = []
        for i in range(0, len(texts), NLP_BATCH_SIZE):
            chunk = texts[i: i + NLP_BATCH_SIZE]
            emb_chunk = model.encode(
                chunk,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            all_emb.append(emb_chunk)
        emb = np.vstack(all_emb)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pca = PCA(n_components=n_comp, random_state=42)
            pcs = pca.fit_transform(emb)
    except Exception as e:
        print(f"\n[NLP] Помилка: {e}", flush=True)
        pcs = np.zeros((n, n_comp))

    for i in range(DOC_NLP_PCA_COMPONENTS):
        col = f"doc_nlp_pca_{i + 1}"
        tender_df[col] = pcs[:, i] if i < pcs.shape[1] else 0.0

    return tender_df


def run():
    client = MongoClient("mongodb://127.0.0.1:27017/")
    db = client["prozorro_mcp"]
    tenders_col = db["tenders"]

    total = tenders_col.estimated_document_count()
    checkpoint = load_checkpoint()

    _nlp_model = None
    if DOCUMENT_FETCH_ENABLED:
        print("Завантажуємо SentenceTransformer модель...", flush=True)
        from sentence_transformers import SentenceTransformer
        _nlp_model = SentenceTransformer(TEXT_MODEL)
        print("Модель завантажена", flush=True)

    # Якщо є checkpoint — продовжуємо з місця зупинки
    query = {}
    if checkpoint:
        from bson import ObjectId
        query = {"_id": {"$gt": ObjectId(checkpoint)}}
        already_done = tenders_col.count_documents({"_id": {"$lte": ObjectId(checkpoint)}})
        print(f"Продовжуємо з checkpoint. Вже оброблено: {already_done}/{total}")
    else:
        already_done = 0
        print(f"Починаємо з нуля. Всього тендерів: {total}")

    _http_session = requests.Session() if DOCUMENT_FETCH_ENABLED else None
    if _http_session:
        _http_session.headers["User-Agent"] = (
            "ProcurementResearch/1.0 (+educational; extraction)"
        )

    processed = already_done
    batch_num = already_done // BATCH_SIZE
    start_time = time.time()

    # Supplier агрегація — накопичуємо між батчами
    all_supplier_rows: list[dict] = []
    all_rel_rows: list[dict] = []

    cursor = tenders_col.find(query).sort("_id", 1)  # sort по _id — стабільний порядок
    batch_tenders: list[dict] = []
    last_id = None

    total_docs_found = 0
    total_docs_downloaded = 0

    def flush_batch():
        nonlocal batch_num, all_supplier_rows, all_rel_rows

        batch_num += 1
        tender_rows = []
        doc_blobs = []

        for idx, t in enumerate(batch_tenders):
            # ← Виводимо ЩЕ ДО fetch документа
            elapsed = time.time() - start_time
            done_so_far = processed + idx
            rate = (done_so_far - already_done) / elapsed if elapsed > 0 and (done_so_far - already_done) > 0 else 0
            eta = (total - done_so_far) / rate if rate > 0 else None
            print_progress(done_so_far, total, batch_num,
                           total_docs_found, total_docs_downloaded,
                           elapsed, eta)

            row = extract_tender_features(t, http_session=_http_session)
            doc_blobs.append(row.pop("_doc_blob"))
            tender_rows.append(row)
            all_supplier_rows.extend(extract_supplier_rows(t))
            all_rel_rows.extend(extract_relationship_rows(t))

        tender_df = pd.DataFrame(tender_rows)

        # CPV медіана тільки по поточному батчу
        # (глобальна буде неточна — але прийнятно для великих даних)
        cpv_median = tender_df.groupby("cpv")["value_amount"].median().to_dict()
        tender_df["median_cpv"] = tender_df["cpv"].map(cpv_median)
        tender_df["cpv_deviation"] = (
            tender_df["value_amount"] - tender_df["median_cpv"]
        ).abs()

        # Числові колонки — заповнення пропусків
        tender_numeric_cols = [
            "value_amount", "winner_price", "price_per_unit",
            "value_per_day", "min_bid", "winner_minus_min",
            "tender_duration_days", "completion_days", "cpv_deviation",
        ] + [f"doc_nlp_pca_{i}" for i in range(1, DOC_NLP_PCA_COMPONENTS + 1)]

        tender_df["description_missing"] = (
            tender_df["description"].isna()
            | (tender_df["description"].str.strip() == "")
        )
        tender_df["description"] = (
            tender_df["description"].fillna("НЕВІДОМО").replace("", "НЕВІДОМО")
        )

        # NLP по батчу
        tender_df = process_batch_nlp(tender_df, doc_blobs, model=_nlp_model)

        for col in tender_numeric_cols:
            if col not in tender_df.columns:
                tender_df[col] = 0.0
            tender_df[f"{col}_missing"] = tender_df[col].isna()
            median_val = pd.to_numeric(tender_df[col], errors='coerce').median()
            tender_df[col] = pd.to_numeric(tender_df[col], errors='coerce').fillna(median_val if pd.notna(median_val) else 0.0).infer_objects(copy=False)

        for col in ["value_amount", "winner_price", "price_per_unit",
                    "value_per_day", "min_bid"]:
            if col in tender_df.columns:
                tender_df[f"log_{col}"] = np.log1p(tender_df[col])

        tender_df["doc_text_features_missing"] = (
            tender_df.get("doc_text_features_missing", pd.Series(1.0)).fillna(1.0)
        )

        append_to_csv(tender_df, OUTPUT_TENDER)

        # Supplier — скидаємо кожні 10 батчів щоб не роздувати RAM
        if batch_num % 10 == 0 and all_supplier_rows:
            _flush_suppliers()
        if batch_num % 10 == 0 and all_rel_rows:
            _flush_relationships()

    def _flush_suppliers():
        nonlocal all_supplier_rows
        if not all_supplier_rows:
            return
        df_sup = pd.DataFrame(all_supplier_rows)
        agg = df_sup.groupby("supplier_id").agg(
            num_bids=("tender_id", "count"),
            num_wins=("is_winner", "sum"),
            avg_competitors=("num_competitors", "mean"),
            rejected_bids=("rejected", "sum"),
            complaints=("complaints", "sum"),
            avg_price_per_unit=("price_per_unit", "mean"),
            avg_contract=("contract_value", "mean"),
            max_contract=("contract_value", "max"),
            min_contract=("contract_value", "min"),
            contract_changes_share=("contract_changes_share", "mean"),
        )
        agg["win_rate"] = agg["num_wins"] / agg["num_bids"]
        for col in ["avg_contract", "max_contract", "min_contract", "avg_price_per_unit"]:
            agg[f"log_{col}"] = np.log1p(agg[col].fillna(0))
        append_to_csv(agg.reset_index(), OUTPUT_SUPPLIER)
        all_supplier_rows = []

    def _flush_relationships():
        nonlocal all_rel_rows
        if not all_rel_rows:
            return
        rel_df = pd.DataFrame(all_rel_rows)
        rel_ds = rel_df.groupby(["buyer_id", "supplier_id"]).agg(
            num_tenders=("tender_id", "count"),
            avg_price=("price", "mean"),
            avg_competitors=("num_bids", "mean"),
            single_bid_share=("single_bid", "mean"),
            buyer_locality=("buyer_locality", "first"),
            supplier_locality=("supplier_locality", "first"),
        )
        buyer_total = rel_df.groupby("buyer_id")["tender_id"].count()
        supplier_totals = rel_df.groupby("supplier_id")["price"].sum()
        rel_ds["buyer_win_share"] = rel_ds["num_tenders"] / rel_ds.index.get_level_values(0).map(buyer_total)
        rel_ds["supplier_income_share"] = (
            rel_ds["avg_price"] * rel_ds["num_tenders"]
        ) / rel_ds.index.get_level_values(1).map(supplier_totals)
        for col in ["buyer_locality", "supplier_locality"]:
            rel_ds[col] = rel_ds[col].fillna("НЕВІДОМО").replace("", "НЕВІДОМО")
        append_to_csv(rel_ds.reset_index(), OUTPUT_REL)
        all_rel_rows = []

    for t in cursor:
        last_id = t["_id"]
        batch_tenders.append(t)

        # Рахуємо документи для прогрес-бару
        docs_in_t = len(t.get("documents") or [])
        total_docs_found += docs_in_t
        if DOCUMENT_FETCH_ENABLED:
            # Наближення — точний лічильник всередині flush
            total_docs_downloaded += min(docs_in_t, 1)  # TIER1_MAX_DOCS=1

        if len(batch_tenders) >= BATCH_SIZE:
            flush_batch()
            batch_tenders.clear()
            save_checkpoint(str(last_id))
            processed += BATCH_SIZE

            elapsed = time.time() - start_time
            rate = (processed - already_done) / elapsed if elapsed > 0 else 0
            remaining = total - processed
            eta = remaining / rate if rate > 0 else None
            print_progress(
                processed, total, batch_num,
                total_docs_found, total_docs_downloaded,
                elapsed, eta,
            )

    # Останній неповний батч
    if batch_tenders:
        flush_batch()
        processed += len(batch_tenders)
        save_checkpoint(str(last_id))

    # Фінальний скид supplier/rel
    _flush_suppliers()
    _flush_relationships()

    elapsed = time.time() - start_time
    print(f"\n\nОброблено {processed} тендерів за {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
    print(f"   → {OUTPUT_TENDER}")
    print(f"   → {OUTPUT_SUPPLIER}")
    print(f"   → {OUTPUT_REL}")


if __name__ == "__main__":
    run()