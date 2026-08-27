"""
prozorro_live.py — завантаження тендера з public-api.prozorro.gov.ua за **внутрішнім id CDB**
(32 hex, поле data.id) та оцінка тим самим pipeline, що й detect_tender_anomalies.

Документація API: https://prozorro-api-docs.readthedocs.io/uk/latest/
"""
from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

from prepare.extraction import extract_tender_features
from train.tender_anomaly import _extract_cpv2, detect_tender_anomalies

PROZORRO_TENDER_API = "https://public-api.prozorro.gov.ua/api/2.5/tenders"

# GET /tenders/{id} — лише внутрішній id CDB (32 hex), див. prozorro-api-docs /tenders.
_CDB_UUID32_RE = re.compile(r"^[a-f0-9]{32}$", re.I)


def _normalize_cdb_uuid32(s: str) -> str | None:
    """32 hex-символи (можливі дефіси як у UUID) → id без дефісів для шляху API."""
    t = s.strip().lower().replace("-", "")
    if len(t) == 32 and _CDB_UUID32_RE.match(t):
        return t
    return None


def fetch_tender_from_api(
    tender_id: str,
    *,
    timeout: float = 60.0,
    session: requests.Session | None = None,
) -> dict:
    """
    GET /api/2.5/tenders/{id} — лише внутрішній id CDB (32 hex), не tenderID на кшталт UA-....
    """
    tid = tender_id.strip()
    if not tid:
        raise ValueError("Порожній tender_id")

    cdb_id = _normalize_cdb_uuid32(tid)
    if not cdb_id:
        raise ValueError(
            "Очікується внутрішній id закупівлі CDB — 32 шістнадцяткові символи "
            "(як у відповіді GET /api/2.5/tenders/{id}, поле data.id). "
            "Посилання prozorro.gov.ua з UA-... не використовуються."
        )

    sess = session or requests
    path = quote(cdb_id, safe="")
    url = f"{PROZORRO_TENDER_API}/{path}"
    r = sess.get(url, timeout=timeout)
    if r.status_code == 404:
        raise FileNotFoundError(f"Тендер не знайдено в API (id={cdb_id})")
    r.raise_for_status()
    payload = r.json()
    data = payload.get("data")
    if not isinstance(data, dict):
        raise ValueError("Неочікувана відповідь API (немає поля data)")
    return data


def _cpv_expected_value_medians(features: pd.DataFrame) -> dict:
    log_va = pd.to_numeric(features["log_value_amount"], errors="coerce")
    va = np.expm1(log_va)
    return (
        features.assign(_va=va)
        .groupby("cpv")["_va"]
        .median()
        .to_dict()
    )


def _cohort_for_live(
    all_features: pd.DataFrame,
    live_cpv: object,
    *,
    exclude_tender_id: str | None,
    min_rows: int = 400,
    max_rows: int = 900,
    random_state: int = 42,
) -> pd.DataFrame:
    """Підвибірка історичних тендерів для спільного скорингу з «живим» рядком."""
    pool = all_features.copy()
    if exclude_tender_id:
        ex = str(exclude_tender_id).strip()
        pool = pool[pool["tender_id"].astype(str).str.strip() != ex].reset_index(drop=True)

    live_g = _extract_cpv2(live_cpv)
    gcol = pool["cpv"].apply(_extract_cpv2)
    same = pool.loc[gcol == live_g].copy()
    rest = pool.loc[gcol != live_g].copy()

    if len(same) < min_rows and len(rest) > 0:
        need = min_rows - len(same)
        add = rest.sample(n=min(need, len(rest)), random_state=random_state)
        cohort = pd.concat([same, add], ignore_index=True)
    else:
        cohort = same.reset_index(drop=True)

    cohort = cohort.drop_duplicates(subset=["tender_id"], keep="first")
    if len(cohort) > max_rows:
        cohort = cohort.sample(n=max_rows, random_state=random_state).reset_index(drop=True)
    return cohort


def _feature_columns(raw_features_path: Path) -> list[str]:
    return list(pd.read_csv(raw_features_path, nrows=0).columns)


def score_tender_from_prozorro_api(
    tender_id: str,
    data_dir: Path,
    *,
    session: requests.Session | None = None,
    cohort_min_rows: int = 400,
    cohort_max_rows: int = 900,
) -> tuple[pd.Series, str | None]:
    """
    Завантажує JSON з GET /api/2.5/tenders/{id} (лише внутрішній id CDB, 32 hex),
    будує рядок ознак (extract_tender_features), об'єднує з когортою з tender_level_features.csv
    і викликає detect_tender_anomalies.

    Повертає (рядок результату для цільового тендера, текст_помилки_або_None).
    """
    raw_path = data_dir / "raw" / "tender_level_features.csv"
    if not raw_path.is_file():
        return pd.Series(dtype=object), f"Немає файлу ознак: {raw_path}"

    sess = session or requests.Session()
    try:
        tjson = fetch_tender_from_api(tender_id, session=sess)
    except FileNotFoundError as e:
        return pd.Series(dtype=object), str(e)
    except ValueError as e:
        return pd.Series(dtype=object), str(e)
    except requests.RequestException as e:
        return pd.Series(dtype=object), f"Помилка мережі API Prozorro: {e}"

    all_features = pd.read_csv(raw_path)
    medians = _cpv_expected_value_medians(all_features)

    try:
        row = extract_tender_features(tjson, cpv_medians=medians, http_session=sess)
    except Exception as e:
        return pd.Series(dtype=object), f"Помилка витягування ознак: {e}"

    row.pop("_doc_blob", None)
    cols = _feature_columns(raw_path)
    live_df = pd.DataFrame([{c: row.get(c, np.nan) for c in cols}])
    live_df["_live_query_row"] = True

    live_cpv = live_df["cpv"].iloc[0]
    tid = str(live_df["tender_id"].iloc[0]).strip()

    cohort = _cohort_for_live(
        all_features,
        live_cpv,
        exclude_tender_id=tid,
        min_rows=cohort_min_rows,
        max_rows=cohort_max_rows,
    )
    for c in cols:
        if c not in cohort.columns:
            cohort[c] = np.nan
    cohort = cohort[cols].copy()
    cohort["_live_query_row"] = False

    combined = pd.concat([cohort, live_df], ignore_index=True)

    tmp = tempfile.mkdtemp(prefix="prozorro_live_")
    try:
        scored = detect_tender_anomalies(df=combined, cache_dir=tmp)
    except Exception as e:
        return pd.Series(dtype=object), f"Помилка моделі: {e}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    live_rows = scored.loc[scored["_live_query_row"].astype(bool)]
    if len(live_rows) == 0:
        return pd.Series(dtype=object), "Не вдалося виділити результат для запитуваного тендера"
    return live_rows.iloc[0], None
