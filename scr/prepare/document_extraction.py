"""
selective HTTP fetch + raw text extraction (PDF/DOCX/HTML/plain).
Semantic features (SentenceTransformer + PCA) are computed in extraction.py on the full corpus.
"""
from __future__ import annotations

import io
import os
import re
import time
from typing import Any
from urllib.parse import urlparse

# Same backbone as tender_anomaly.py title/description embeddings.
TEXT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DOC_NLP_PCA_COMPONENTS = 5

# Set DOCUMENT_FETCH=0 to skip Tier 1 downloads (Tier 0 still runs; doc NLP PCs are zeros).
DOCUMENT_FETCH_ENABLED = os.environ.get("DOCUMENT_FETCH", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)

TIER1_MAX_DOCS = 1
TIER1_MAX_BYTES_PER_FILE = 5 * 1024 * 1024
TIER1_MAX_TOTAL_CHARS = 100_000
REQUEST_TIMEOUT = 25
REQUEST_SLEEP_SEC = 0.05
_TYPE_PRIORITY_KEYWORDS = (
    "technicalspecifications",
    "biddingdocuments",
    "evaluationcriteria",
    "eligibilitycriteria",
    "contractproforma",
    "clarifications",
    "billofquantity",
    "bidders",
    "notice",
)


def _priority(doc: dict[str, Any]) -> int:
    dt = (doc.get("documentType") or "").lower()
    for i, kw in enumerate(_TYPE_PRIORITY_KEYWORDS):
        if kw in dt:
            return i
    return len(_TYPE_PRIORITY_KEYWORDS)


def _doc_url(doc: dict[str, Any]) -> str | None:
    u = doc.get("url") or doc.get("url_en") or doc.get("url_uk")
    if not u or not isinstance(u, str):
        return None
    u = u.strip()
    if not u.startswith(("http://", "https://")):
        return None
    return u


def _format_hint(doc: dict[str, Any], url: str) -> str:
    fmt = (doc.get("format") or "").lower()
    if fmt:
        return fmt
    path = urlparse(url).path.lower()
    if path.endswith(".pdf"):
        return "application/pdf"
    if path.endswith(".docx"):
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if path.endswith(".doc"):
        return "application/msword"
    if path.endswith((".htm", ".html")):
        return "text/html"
    return ""


def _strip_html(raw: bytes) -> str:
    try:
        text = raw.decode("utf-8", errors="ignore")
    except Exception:
        text = raw.decode("latin-1", errors="ignore")
    text = re.sub(r"<script[^>]*>[\s\S]*?</script>", " ", text, flags=re.I)
    text = re.sub(r"<style[^>]*>[\s\S]*?</style>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _detect_format_by_magic(data: bytes) -> str:
    """Визначає формат файлу по перших байтах, незалежно від URL."""
    if data[:4] == b'%PDF':
        return "application/pdf"
    if data[:4] == b'PK\x03\x04':
        # ZIP-контейнер — може бути DOCX, XLSX, і т.д.
        if b'word/' in data[:2000]:
            return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        return "application/zip"
    if data[:8] == b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1':
        return "application/msword"  # старий .doc
    if data[:5] in (b'<html', b'<!DOC', b'<?xml'):
        return "text/html"
    return ""


def _extract_text_from_bytes(data: bytes, fmt: str, url: str) -> str:
    # Перевіряємо magic bytes — надійніше ніж URL або fmt
    detected = _detect_format_by_magic(data)
    fmt = detected or fmt  # magic bytes мають пріоритет

    path = urlparse(url).path.lower()

    if "pdf" in fmt or path.endswith(".pdf"):
        try:
            from pypdf import PdfReader
            import logging
            logging.getLogger("pypdf").setLevel(logging.ERROR)
            reader = PdfReader(io.BytesIO(data), strict=False)
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        except Exception:
            return ""

    if "wordprocessingml" in fmt or "msword" in fmt or path.endswith((".docx", ".doc")):
        try:
            import docx
            d = docx.Document(io.BytesIO(data))
            return "\n".join(p.text for p in d.paragraphs if p.text)
        except Exception:
            return ""

    if "html" in fmt or path.endswith((".html", ".htm")):
        return _strip_html(data)

    if "text/" in fmt or path.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")

    # Fallback — спробуй як текст
    if b"\0" not in data[:2000]:
        return data.decode("utf-8", errors="ignore")

    return ""

def tier1_placeholder_meta_and_blob() -> tuple[dict[str, float], str]:
    """No Tier 1 fetch: metadata only; empty blob for NLP."""
    return (
        {
            "doc_text_features_missing": 1.0
        },
        "",
    )


def summarize_tier1_for_documents(
    documents: list[dict[str, Any]] | None,
    session: Any | None = None,
) -> tuple[dict[str, float], str]:
    if not DOCUMENT_FETCH_ENABLED:
        return tier1_placeholder_meta_and_blob()

    docs = [d for d in (documents or []) if isinstance(d, dict)]
    ranked = sorted(docs, key=lambda d: (_priority(d), _doc_url(d) or ""))

    try:
        import requests
    except ImportError:
        return {"doc_text_features_missing": 1.0}, ""

    sess = session or requests.Session()
    sess.headers.setdefault(
        "User-Agent",
        "ProcurementResearch/1.0 (+educational; extraction)",
    )

    combined: list[str] = []
    downloaded = 0

    for doc in ranked:
        if downloaded >= TIER1_MAX_DOCS:
            break
        url = _doc_url(doc)
        if not url:
            continue
        fmt = _format_hint(doc, url)
        try:
            r = sess.get(url, timeout=REQUEST_TIMEOUT, stream=True)
            r.raise_for_status()
            buf = io.BytesIO()
            for chunk in r.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                buf.write(chunk)
                if buf.tell() > TIER1_MAX_BYTES_PER_FILE:
                    break
            data = buf.getvalue()
        except Exception:
            time.sleep(REQUEST_SLEEP_SEC)
            continue

        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                text = _extract_text_from_bytes(data, fmt, url)
        except Exception:
            time.sleep(REQUEST_SLEEP_SEC)
            continue

        if text:
            combined.append(text)
            downloaded += 1
        time.sleep(REQUEST_SLEEP_SEC)

    blob = "\n\n".join(combined)[:TIER1_MAX_TOTAL_CHARS]
    return {"doc_text_features_missing": 0.0 if downloaded > 0 else 1.0}, blob