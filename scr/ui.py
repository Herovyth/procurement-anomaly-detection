import streamlit as st
import pandas as pd
from pyvis.network import Network
import tempfile
import re
from pathlib import Path

st.set_page_config(layout="wide", page_title="Prozorro Anomaly Detector")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Unbounded:wght@400;700;900&display=swap');

:root {
    --bg:      #0a0c10;
    --surface: #111318;
    --border:  #1e2230;
    --accent:  #e84545;
    --blue:    #3b82f6;
    --warn:    #f59e0b;
    --ok:      #22c55e;
    --text:    #e2e8f0;
    --muted:   #64748b;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace;
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stHeader"] { display: none; }

.main-title {
    font-family: 'Unbounded', sans-serif;
    font-size: 1.6rem; font-weight: 900; letter-spacing: -0.02em;
    padding: 1.5rem 0 0.2rem;
}
.main-sub {
    font-size: 0.7rem; color: var(--muted);
    text-transform: uppercase; letter-spacing: .1em; margin-bottom: 1.4rem;
}

[data-testid="stTabs"] button {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important; font-weight: 600 !important;
    text-transform: uppercase !important; letter-spacing: .06em !important;
    color: var(--muted) !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
    padding: 0.55rem 1.1rem !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

.card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 1.2rem 1.4rem; margin-bottom: .9rem;
}
.card-sm {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 6px; padding: .85rem 1.1rem; margin-bottom: .6rem;
}
.card-result {
    background: var(--surface); border: 2px solid var(--border);
    border-radius: 10px; padding: 1.5rem 1.8rem; margin-bottom: 1rem;
    transition: border-color .2s;
}
.card-anomaly  { border-color: var(--accent) !important; }
.card-normal   { border-color: var(--ok)     !important; }

.score-num {
    font-family: 'Unbounded', sans-serif;
    font-size: 3rem; font-weight: 900; line-height: 1;
}
.score-high { color: var(--accent); }
.score-med  { color: var(--warn);   }
.score-low  { color: var(--ok);     }

.gauge-bg {
    background: var(--border); border-radius: 4px;
    height: 6px; overflow: hidden; margin: .5rem 0 .7rem;
}
.gauge-fill { height: 100%; border-radius: 4px; }

.stats { display: flex; gap: .8rem; flex-wrap: wrap; margin: .9rem 0; }
.stat {
    background: rgba(255,255,255,.04); border: 1px solid var(--border);
    border-radius: 6px; padding: .55rem .9rem; flex: 1; min-width: 90px;
}
.stat-val { font-family: 'Unbounded', sans-serif; font-size: 1.15rem; font-weight: 700; }
.stat-key { font-size: .6rem; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; margin-top: .15rem; }

.flag {
    display: inline-block; font-size: .67rem; font-weight: 600;
    padding: .18rem .55rem; border-radius: 3px;
    margin: .18rem .18rem .18rem 0;
    text-transform: uppercase; letter-spacing: .05em;
}
.f-red    { background: rgba(232,69,69,.15);  color: var(--accent); border: 1px solid rgba(232,69,69,.3); }
.f-yellow { background: rgba(245,158,11,.12); color: var(--warn);   border: 1px solid rgba(245,158,11,.28); }
.f-ok     { background: rgba(34,197,94,.12);  color: var(--ok);     border: 1px solid rgba(34,197,94,.28); }
.f-blue   { background: rgba(59,130,246,.12); color: var(--blue);   border: 1px solid rgba(59,130,246,.28); }

.stTextInput input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important; border-radius: 6px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: .85rem !important;
    padding: .65rem .95rem !important;
}
.stTextInput input:focus {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,.18) !important;
}

.sec {
    font-size: .63rem; color: var(--muted);
    text-transform: uppercase; letter-spacing: .11em;
    margin: .9rem 0 .4rem;
}

.stButton > button {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important; border-radius: 5px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .75rem !important; text-align: left !important;
    padding: .45rem .8rem !important;
}
.stButton > button:hover {
    border-color: var(--blue) !important;
    background: rgba(59,130,246,.08) !important;
}

/* Analyze button — primary */
div[data-testid="stButton"].analyze-btn > button {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    color: #fff !important; font-weight: 700 !important;
    font-size: .82rem !important; padding: .65rem 1.4rem !important;
    border-radius: 6px !important; letter-spacing: .06em !important;
    text-transform: uppercase !important;
}
div[data-testid="stButton"].analyze-btn > button:hover {
    background: #c73232 !important; border-color: #c73232 !important;
}

.verdict-anomaly {
    font-family: 'Unbounded', sans-serif; font-size: 1.5rem; font-weight: 900;
    color: var(--accent); letter-spacing: -.01em;
}
.verdict-normal {
    font-family: 'Unbounded', sans-serif; font-size: 1.5rem; font-weight: 900;
    color: var(--ok); letter-spacing: -.01em;
}
</style>
""", unsafe_allow_html=True)


# ── Data ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


@st.cache_data
def load_data():
    tenders = pd.read_csv(DATA_DIR / "prepared/tender_anomaly_results.csv")
    suppliers = pd.read_csv(DATA_DIR / "prepared/supplier_anomaly_results.csv")
    relations = pd.read_csv(DATA_DIR / "prepared/relationship_anomaly_results.csv")
    return tenders, suppliers, relations

tenders, suppliers, relations = load_data()

# ── Helpers ───────────────────────────────────────────────────────────────────
T_MAX = float(tenders["risk_score"].max())   if "risk_score" in tenders.columns   else 1.0
S_MAX = float(suppliers["risk_score"].max()) if "risk_score" in suppliers.columns else 1.0

def score_css(s, mx):
    r = s / mx if mx else 0
    return "score-high" if r >= .65 else ("score-med" if r >= .35 else "score-low")

def flag_css(s, mx):
    r = s / mx if mx else 0
    return "f-red" if r >= .65 else ("f-yellow" if r >= .35 else "f-ok")

def risk_text(s, mx):
    r = s / mx if mx else 0
    return "АНОМАЛЬНИЙ" if r >= .65 else ("ПІДОЗРІЛИЙ" if r >= .35 else "НОРМАЛЬНИЙ")

def gauge_color(s, mx):
    r = s / mx if mx else 0
    return "#e84545" if r >= .65 else ("#f59e0b" if r >= .35 else "#22c55e")

def fmt_pct(v):
    try:    return f"{float(v):.1%}"
    except: return "n/a"

def fmt_money(v):
    try:
        f = float(v)
        if f >= 1_000_000: return f"{f/1_000_000:.2f} млн"
        if f >= 1_000:     return f"{f/1_000:.1f} тис"
        return f"{f:.2f}"
    except: return "n/a"

def extract_tender_id(raw: str) -> str:
    """Extract tenderID from a Prozorro URL or return raw string."""
    raw = raw.strip()
    # https://prozorro.gov.ua/tender/UA-2024-01-15-000123-a
    m = re.search(r'(UA-\d{4}-\d{2}-\d{2}-\d+(?:-\w+)?)', raw)
    return m.group(1) if m else raw

def col_exists(df, *names):
    return [n for n in names if n in df.columns]


def find_tender(tender_id: str) -> pd.DataFrame:
    found = tenders[tenders["tender_id"].astype(str).str.strip() == tender_id]
    if len(found) == 0:
        found = tenders[tenders["tender_id"].astype(str).str.contains(
            re.escape(tender_id), na=False
        )]
    return found


def build_relation_risk_score(df: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=df.index, dtype=float)
    for column, weight in (
        ("single_bid_share", 0.4),
        ("buyer_win_share", 0.3),
        ("supplier_income_share", 0.3),
    ):
        if column in df.columns:
            score = score.add(df[column].fillna(0).astype(float) * weight, fill_value=0)
    return score.round(4)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">⚡ Prozorro Anomaly Detector</div>', unsafe_allow_html=True)
st.markdown("<div class='main-sub'>Виявлення аномальних тендерів · підрядників · зв'язків</div>",
            unsafe_allow_html=True)

overview_rel = relations.copy()
if "risk_score" not in overview_rel.columns:
    overview_rel["risk_score"] = build_relation_risk_score(overview_rel)

high_tender_risk = int((tenders["risk_score"] >= tenders["risk_score"].quantile(0.80)).sum()) if "risk_score" in tenders.columns and len(tenders) else 0
high_supplier_risk = int((suppliers["risk_score"] >= suppliers["risk_score"].quantile(0.80)).sum()) if "risk_score" in suppliers.columns and len(suppliers) else 0
high_relation_risk = int((overview_rel["risk_score"] >= overview_rel["risk_score"].quantile(0.80)).sum()) if "risk_score" in overview_rel.columns and len(overview_rel) else 0

st.markdown(f"""
<div class="stats" style="margin-top:.2rem;margin-bottom:1.4rem;">
    <div class="stat">
        <div class="stat-val">{len(tenders):,}</div>
        <div class="stat-key">Тендерів у вибірці</div>
    </div>
    <div class="stat">
        <div class="stat-val">{high_tender_risk:,}</div>
        <div class="stat-key">Тендерів у top 20% ризику</div>
    </div>
    <div class="stat">
        <div class="stat-val">{high_supplier_risk:,}</div>
        <div class="stat-key">Підрядників у top 20%</div>
    </div>
    <div class="stat">
        <div class="stat-val">{high_relation_risk:,}</div>
        <div class="stat-key">Ризикових зв'язків у top 20%</div>
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔍  Тендер-чекер", "🏢  Підрядники", "🔗  Зв'язки"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Тендер-чекер
# Колонки: tender_id, risk_score, title, cpv_code, value_amount,
#          winner_price, num_bids, tender_duration_days, cpv_deviation
# Вхід: URL з Prozorro → витягуємо UA-XXXX-XX-XX-XXXXXX
# ════════════════════════════════════════════════════════════════════════════

# Преобчислюємо медіани одноразово для порівнянь
_med_dur = float(tenders["tender_duration_days"].median()) if "tender_duration_days" in tenders.columns else 14.0
_med_bids = float(tenders["num_bids"].median())            if "num_bids"             in tenders.columns else 2.0

with tab1:
    # ── Пошукова панель ──────────────────────────────────────────────────
    st.markdown('<div class="sec">Перевірка тендера</div>', unsafe_allow_html=True)

    inp_col, btn_col = st.columns([5, 1], gap="small")
    with inp_col:
        raw_input = st.text_input(
            "", label_visibility="collapsed", key="tender_input",
            placeholder="Вставте посилання на тендер з Prozorro (prozorro.gov.ua/tender/UA-...)"
        )
    with btn_col:
        st.markdown('<div class="analyze-btn">', unsafe_allow_html=True)
        analyze = st.button("Перевірити", key="btn_analyze", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        '<div style="font-size:.65rem;color:var(--muted);margin-top:-.3rem;margin-bottom:1rem;">'
        'Підтримуються посилання: prozorro.gov.ua/tender/UA-... &nbsp;·&nbsp; '
        'bi.prozorro.org/... &nbsp;·&nbsp; dozorro.org/tender/UA-...</div>',
        unsafe_allow_html=True
    )

    if analyze and raw_input.strip():
        st.session_state["last_tender_query"] = raw_input.strip()
        st.session_state["last_tender_id"] = extract_tender_id(raw_input)
    elif not raw_input.strip():
        st.session_state.pop("last_tender_query", None)
        st.session_state.pop("last_tender_id", None)

    active_tid = st.session_state.get("last_tender_id")
    pending_recheck = (
        raw_input.strip()
        and st.session_state.get("last_tender_query")
        and raw_input.strip() != st.session_state.get("last_tender_query")
    )

    # ── Пустий стан ──────────────────────────────────────────────────────
    if not raw_input.strip():
        st.markdown("""
        <div class="card" style="text-align:center;padding:3.5rem 2rem;border-style:dashed;">
            <div style="font-size:2.5rem;margin-bottom:.6rem;">🔍</div>
            <div style="font-family:'Unbounded',sans-serif;font-size:.9rem;">
                Вставте посилання на тендер
            </div>
            <div style="font-size:.7rem;color:var(--muted);margin-top:.5rem;line-height:1.9;">
                Система знайде тендер у датасеті та покаже:<br>
                🔴 Аномальний / ✅ Нормальний &nbsp;·&nbsp;
                ⚡ Risk score &nbsp;·&nbsp;
                📋 Деталі &nbsp;·&nbsp;
                ⚠ Які ознаки спрацювали &nbsp;·&nbsp;
                📊 Порівняння з CPV-аналогами
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Аналіз ───────────────────────────────────────────────────────────
    elif active_tid:
        tid = active_tid
        found = find_tender(tid)

        if pending_recheck:
            st.markdown("""
            <div class="card-sm" style="border-style:dashed;">
                Натисніть <b>Перевірити</b>, щоб оновити результат для нового посилання.
            </div>
            """, unsafe_allow_html=True)

        # ── Не знайдено ──
        if len(found) == 0:
            st.markdown(f"""
            <div class="card" style="border-color:var(--warn);text-align:center;padding:2.5rem;">
                <div style="font-size:1.8rem;margin-bottom:.5rem;">🔎</div>
                <div style="font-family:'Unbounded',sans-serif;font-size:.95rem;color:var(--warn);">
                    Тендер не знайдено в датасеті
                </div>
                <div style="font-size:.72rem;color:var(--muted);margin-top:.6rem;line-height:1.8;">
                    Витягнутий ID: <code style="color:var(--text);background:rgba(255,255,255,.06);
                    padding:.1rem .4rem;border-radius:3px;">{tid}</code><br>
                    Можливі причини: тендер не потрапив у вибірку датасету,
                    або ID у посиланні має інший формат
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Знайдено ──
        else:
            row      = found.iloc[0]
            score    = float(row["risk_score"])
            pct_fill = score / T_MAX * 100
            sc       = score_css(score, T_MAX)
            gc       = gauge_color(score, T_MAX)
            pct_rank = float((tenders["risk_score"] < score).mean() * 100)

            # Поріг аномальності: топ 20% за ризиком
            threshold = float(tenders["risk_score"].quantile(0.80))
            is_anomaly = score >= threshold

            card_cls    = "card-anomaly" if is_anomaly else "card-normal"
            verdict_cls = "verdict-anomaly" if is_anomaly else "verdict-normal"
            verdict_ico = "🚨" if is_anomaly else "✅"
            verdict_txt = "АНОМАЛЬНИЙ ТЕНДЕР" if is_anomaly else "ТЕНДЕР БЕЗ АНОМАЛІЙ"

            # ── Збираємо пояснення (прапорці) ────────────────────────────
            # Кожен прапорець: (css_class, emoji_label, short_explanation)
            flags = []

            # 1. Цінове відхилення від медіани CPV
            dev = float(row["cpv_deviation"]) if pd.notna(row.get("cpv_deviation")) else None
            if dev is not None:
                if dev > 3.0:
                    flags.append(("f-red",
                        f"💰 Ціна у {dev:.1f}× вища за медіану CPV-категорії",
                        f"Очікувана вартість аномально висока відносно аналогічних закупівель "
                        f"у тій самій CPV-категорії (відхилення: {dev:.2f}×)."))
                elif dev > 1.5:
                    flags.append(("f-yellow",
                        f"💰 Ціна у {dev:.1f}× вища за медіану CPV",
                        f"Вартість підвищена відносно типових закупівель категорії "
                        f"(відхилення: {dev:.2f}×)."))
                elif dev < 0.4 and dev > 0:
                    flags.append(("f-yellow",
                        f"💰 Ціна у {1/dev:.1f}× нижча за медіану — можливий демпінг",
                        f"Ціна значно нижча за типову для цієї категорії, "
                        f"що може свідчити про демпінг або некоректне завдання."))
                else:
                    flags.append(("f-ok",
                        f"💰 Ціна в межах норми (відхилення {dev:.2f}×)",
                        "Вартість тендера відповідає типовому діапазону для цієї CPV-категорії."))

            # 2. Конкурентність (num_bids)
            nb = int(row["num_bids"]) if pd.notna(row.get("num_bids")) else None
            if nb is not None:
                if nb == 1:
                    flags.append(("f-red",
                        "👤 Single-bid: тільки 1 учасник",
                        "Лише одна пропозиція — відсутня конкуренція. "
                        "Це один із найсильніших індикаторів ризикової закупівлі."))
                elif nb == 2:
                    flags.append(("f-yellow",
                        f"👥 Низька конкуренція: {nb} учасники",
                        f"Кількість пропозицій ({nb}) нижча за медіанну по датасету "
                        f"({_med_bids:.0f}). Обмежена конкуренція може свідчити про "
                        f"навмисне звуження кола учасників."))
                else:
                    flags.append(("f-ok",
                        f"👥 Нормальна конкуренція: {nb} учасників",
                        f"Кількість пропозицій ({nb}) відповідає нормальному рівню."))

            # 3. Тривалість тендеру
            dur = float(row["tender_duration_days"]) if pd.notna(row.get("tender_duration_days")) else None
            if dur is not None:
                if dur < _med_dur * 0.3:
                    flags.append(("f-red",
                        f"⏱ Підозріло короткий термін: {dur:.0f} днів",
                        f"Тривалість тендеру ({dur:.0f} дн.) у {_med_dur/dur:.1f}× коротша "
                        f"за медіану ({_med_dur:.0f} дн.). Це може навмисно обмежувати "
                        f"коло потенційних учасників."))
                elif dur > _med_dur * 4:
                    flags.append(("f-yellow",
                        f"⏱ Незвично довгий термін: {dur:.0f} днів",
                        f"Тривалість тендеру ({dur:.0f} дн.) у {dur/_med_dur:.1f}× довша "
                        f"за медіану ({_med_dur:.0f} дн.)."))
                else:
                    flags.append(("f-ok",
                        f"⏱ Тривалість в нормі: {dur:.0f} днів",
                        f"Термін проведення відповідає типовому діапазону."))

            # 4. Різниця очікуваної вартості та ціни переможця
            va  = float(row["value_amount"]) if pd.notna(row.get("value_amount")) else None
            wp  = float(row["winner_price"]) if pd.notna(row.get("winner_price")) else None
            if va and wp and va > 0:
                discount = (va - wp) / va
                if discount < 0.005:
                    flags.append(("f-yellow",
                        f"💵 Знижка майже відсутня: {discount:.1%} від очікуваної вартості",
                        f"Переможець запропонував ціну практично рівну очікуваній вартості "
                        f"({fmt_money(wp)} vs {fmt_money(va)}). Типова знижка у конкурентних "
                        f"тендерах вища."))
                elif discount > 0.5:
                    flags.append(("f-yellow",
                        f"💵 Аномально велика знижка: -{discount:.0%}",
                        f"Переможець запропонував ціну на {discount:.0%} нижчу за очікувану — "
                        f"можливий демпінг або помилка в технічному завданні."))
                else:
                    flags.append(("f-ok",
                        f"💵 Знижка в нормі: -{discount:.1%}",
                        f"Різниця між очікуваною вартістю та ціною переможця "
                        f"відповідає типовій конкурентній знижці."))

            # 5. Перцентиль у датасеті
            if pct_rank >= 95:
                flags.append(("f-red",
                    f"📊 Топ 5% найризикованіших в датасеті",
                    f"Risk score {score:.4f} входить у 5% найвищих значень по всьому датасету."))
            elif pct_rank >= 80:
                flags.append(("f-yellow",
                    f"📊 Топ 20% за ризиком (перцентиль {pct_rank:.0f}%)",
                    f"Risk score вище за {pct_rank:.0f}% тендерів датасету."))

            if not flags:
                flags.append(("f-ok",
                    "✓ Жодних значних відхилень не виявлено",
                    "Всі перевірені показники в межах норми."))

            # ── Layout результатів ────────────────────────────────────────
            left_col, right_col = st.columns([1, 2], gap="large")

            # ── Ліво: вердикт + score + gauge ──
            with left_col:
                flag_for_card = "f-red" if is_anomaly else ("f-yellow" if sc == "score-med" else "f-ok")
                st.markdown(f"""
                <div class="card-result {card_cls}">
                    <div class="{verdict_cls}">{verdict_ico} {verdict_txt}</div>
                    <div style="margin-top:1.1rem;">
                        <div class="score-num {sc}">{score:.4f}</div>
                        <div class="gauge-bg" style="margin-top:.6rem;">
                            <div class="gauge-fill" style="width:{pct_fill:.1f}%;background:{gc};"></div>
                        </div>
                    </div>
                    <div class="stats" style="margin-top:.8rem;">
                        <div class="stat">
                            <div class="stat-val {sc}">{score:.4f}</div>
                            <div class="stat-key">Risk Score</div>
                        </div>
                        <div class="stat">
                            <div class="stat-val">{pct_rank:.0f}%</div>
                            <div class="stat-key">Перцентиль</div>
                        </div>
                        <div class="stat">
                            <div class="stat-val">{threshold:.4f}</div>
                            <div class="stat-key">Поріг аномалії (80%)</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Право: деталі тендера ──
            with right_col:
                title_str = str(row["title"])[:120] if pd.notna(row.get("title")) else "—"
                cpv_str   = str(row["cpv_code"])    if pd.notna(row.get("cpv_code"))   else "—"
                va_str    = fmt_money(row["value_amount"])  if pd.notna(row.get("value_amount"))  else "—"
                wp_str    = fmt_money(row["winner_price"])  if pd.notna(row.get("winner_price"))  else "—"
                nb_str    = str(int(row["num_bids"]))       if pd.notna(row.get("num_bids"))       else "—"
                dur_str   = f"{float(row['tender_duration_days']):.0f} дн." if pd.notna(row.get("tender_duration_days")) else "—"
                dev_str   = f"{float(row['cpv_deviation']):.3f}×"           if pd.notna(row.get("cpv_deviation"))       else "—"

                detail_rows = [
                    ("🆔 ID",                     str(row["tender_id"])),
                    ("📋 Назва",                  title_str),
                    ("🏷 CPV код",                cpv_str),
                    ("💰 Очікувана вартість",     va_str),
                    ("💵 Ціна переможця",         wp_str),
                    ("👥 К-сть пропозицій",       nb_str),
                    ("📅 Тривалість",             dur_str),
                    ("📊 Відхилення від медіани CPV", dev_str),
                ]
                rows_html = "".join(
                    f'<tr>'
                    f'<td style="color:var(--muted);font-size:.67rem;padding:.32rem .7rem .32rem 0;'
                    f'white-space:nowrap;vertical-align:top;">{k}</td>'
                    f'<td style="font-size:.75rem;padding:.32rem 0;word-break:break-word;">{v}</td>'
                    f'</tr>'
                    for k, v in detail_rows
                )
                st.markdown(f"""
                <div class="card">
                    <div class="sec" style="margin-top:0;">Деталі тендера</div>
                    <table style="width:100%;border-collapse:collapse;">{rows_html}</table>
                </div>
                """, unsafe_allow_html=True)

            # ── Прапорці з поясненнями ────────────────────────────────────
            st.markdown('<div class="sec">Аналіз показників</div>', unsafe_allow_html=True)

            for (fc, short_lbl, explanation) in flags:
                icon = "▲" if fc == "f-red" else ("◆" if fc == "f-yellow" else "●")
                st.markdown(f"""
                <div class="card-sm" style="{'border-left:3px solid var(--accent)' if fc=='f-red'
                    else ('border-left:3px solid var(--warn)' if fc=='f-yellow'
                    else 'border-left:3px solid var(--ok)')};">
                    <div style="display:flex;align-items:flex-start;gap:.7rem;">
                        <span class="flag {fc}" style="margin:0;white-space:nowrap;">{short_lbl}</span>
                        <span style="font-size:.72rem;color:var(--muted);line-height:1.5;
                                     padding-top:.05rem;">{explanation}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Порівняння з аналогами по CPV ─────────────────────────────
            cpv = row.get("cpv_code")
            if pd.notna(cpv):
                same_cpv = (
                    tenders[
                        (tenders["cpv_code"] == cpv) &
                        (tenders["tender_id"].astype(str) != str(row["tender_id"]))
                    ]
                    .sort_values("risk_score", ascending=False)
                    .head(8)
                )

                if len(same_cpv) > 0:
                    # Статистика по CPV
                    cpv_med_score = tenders[tenders["cpv_code"] == cpv]["risk_score"].median()
                    cpv_count     = len(tenders[tenders["cpv_code"] == cpv])

                    st.markdown(f"""
                    <div class="sec">Порівняння з аналогами у CPV {cpv}
                        &nbsp;<span style="color:var(--text);font-weight:600;">{cpv_count} тендерів</span>
                        &nbsp;·&nbsp; медіана ризику:
                        <span style="color:var(--warn);font-weight:600;">{cpv_med_score:.4f}</span>
                        &nbsp;·&nbsp; ваш тендер:
                        <span style="color:{gc};font-weight:600;">{score:.4f}</span>
                    </div>
                    """, unsafe_allow_html=True)

                    show_cols = ["tender_id", "risk_score", "value_amount",
                                 "winner_price", "num_bids", "cpv_deviation",
                                 "tender_duration_days"]
                    show_cols = [c for c in show_cols if c in same_cpv.columns]
                    st.dataframe(
                        same_cpv[show_cols].reset_index(drop=True),
                        use_container_width=True, hide_index=True, height=280
                    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Підрядники · Картка
# suppliers cols: supplier_id, risk_score, num_wins
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    list_col, card_col = st.columns([1, 2], gap="large")

    with list_col:
        st.markdown('<div class="sec">Пошук</div>', unsafe_allow_html=True)
        search = st.text_input("", placeholder="supplier_id або частина назви…",
                               label_visibility="collapsed", key="s_search")

        st.markdown('<div class="sec">Мінімальна кількість перемог</div>', unsafe_allow_html=True)
        max_wins = int(suppliers["num_wins"].max()) if "num_wins" in suppliers.columns else 200
        min_wins = st.slider("", 1, max(max_wins, 2), min(5, max_wins),
                             label_visibility="collapsed", key="s_minwins")

        fs = suppliers.copy()
        if "num_wins" in suppliers.columns:
            fs = fs[fs["num_wins"] >= min_wins]
        if search:
            fs = fs[fs["supplier_id"].astype(str).str.contains(search, case=False, na=False)]
        fs = fs.sort_values("risk_score", ascending=False)

        st.markdown(f'<div class="sec">{len(fs)} підрядників</div>', unsafe_allow_html=True)

        for i, (idx, row) in enumerate(fs.head(80).iterrows()):
            sid   = str(row["supplier_id"])
            score = float(row["risk_score"])
            wins  = int(row["num_wins"]) if "num_wins" in row.index else 0
            short = f"{sid[:26]}{'…' if len(sid) > 26 else ''}"
            label = f"{short}  ·  ⚡{score:.3f}  ·  🏆{wins}"
            if st.button(label, key=f"sb_{i}_{idx}", use_container_width=True):
                st.session_state["sel_sup"] = idx
                st.rerun()

    with card_col:
        sel = st.session_state.get("sel_sup")
        row = (suppliers.loc[sel]
               if sel is not None and sel in suppliers.index
               else (fs.iloc[0] if len(fs) else None))

        if row is None:
            st.markdown("""
            <div class="card" style="text-align:center;padding:4rem 2rem;border-style:dashed;">
                <div style="font-size:2.5rem;">🔍</div>
                <div style="font-family:'Unbounded',sans-serif;font-size:.9rem;margin-top:.6rem;">
                    Оберіть підрядника
                </div>
                <div style="font-size:.7rem;color:var(--muted);margin-top:.3rem;">зі списку ліворуч</div>
            </div>""", unsafe_allow_html=True)
        else:
            sid   = str(row["supplier_id"])
            score = float(row["risk_score"])
            wins  = int(row["num_wins"]) if "num_wins" in row.index else 0
            pct   = score / S_MAX * 100
            gc    = gauge_color(score, S_MAX)
            sc    = score_css(score, S_MAX)
            fc    = flag_css(score, S_MAX)
            rt    = risk_text(score, S_MAX)
            pct_rank = float((suppliers["risk_score"] < score).mean() * 100)

            st.markdown(f"""
            <div class="card">
                <div style="display:flex;gap:1.4rem;align-items:flex-start;">
                    <div style="min-width:100px;text-align:center;">
                        <div class="score-num {sc}">{score:.3f}</div>
                        <div class="gauge-bg">
                            <div class="gauge-fill" style="width:{pct:.1f}%;background:{gc};"></div>
                        </div>
                        <span class="flag {fc}">{rt}</span>
                    </div>
                    <div style="flex:1;min-width:0;">
                        <div style="font-family:'Unbounded',sans-serif;font-size:.95rem;
                                    font-weight:700;word-break:break-all;line-height:1.35;
                                    margin-bottom:.3rem;">{sid}</div>
                        <div style="font-size:.68rem;color:var(--muted);">Підрядник · Prozorro</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="stats">
                <div class="stat">
                    <div class="stat-val">{wins}</div>
                    <div class="stat-key">Виграних тендерів</div>
                </div>
                <div class="stat">
                    <div class="stat-val {sc}">{score:.4f}</div>
                    <div class="stat-key">Risk Score</div>
                </div>
                <div class="stat">
                    <div class="stat-val">{pct_rank:.0f}%</div>
                    <div class="stat-key">Перцентиль ризику</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Flags
            flags = []
            r = score / S_MAX
            if r >= .65:
                flags.append(("f-red",    "🚨 Статистично аномальний підрядник"))
            elif r >= .35:
                flags.append(("f-yellow", "⚠ Підвищений рівень ризику"))
            else:
                flags.append(("f-ok",     "✓ Нормальний рівень ризику"))

            if pct_rank >= 95:
                flags.append(("f-red",    "🔴 Топ 5% найризикованіших"))
            elif pct_rank >= 90:
                flags.append(("f-yellow", "🟡 Топ 10% найризикованіших"))

            med_wins = float(suppliers["num_wins"].median()) if "num_wins" in suppliers.columns else 10
            if wins > med_wins * 5:
                flags.append(("f-yellow", f"📈 Висока активність: {wins} перемог"))
            elif wins <= 2:
                flags.append(("f-yellow", "🆕 Дуже мало перемог (≤2)"))

            sup_rels = relations[relations["supplier_id"].astype(str) == sid]
            if len(sup_rels) > 0:
                n_buyers = sup_rels["buyer_id"].nunique() if "buyer_id" in sup_rels.columns else 0
                if "single_bid_share" in sup_rels.columns:
                    avg_sbs = sup_rels["single_bid_share"].mean()
                    if avg_sbs > 0.7:
                        flags.append(("f-red",    f"👤 Single-bid частка: {avg_sbs:.0%}"))
                    elif avg_sbs > 0.4:
                        flags.append(("f-yellow", f"👥 Single-bid частка: {avg_sbs:.0%}"))
                if "supplier_income_share" in sup_rels.columns:
                    avg_sis = sup_rels["supplier_income_share"].mean()
                    if avg_sis > 0.8:
                        flags.append(("f-red",    f"🔗 {avg_sis:.0%} доходу від одного замовника"))
                    elif avg_sis > 0.5:
                        flags.append(("f-yellow", f"⚠ Концентрація доходу: {avg_sis:.0%}"))
                if n_buyers == 1:
                    flags.append(("f-red",    "⛓ Всі тендери від одного замовника"))
                elif n_buyers <= 3:
                    flags.append(("f-yellow", f"⚠ Лише {n_buyers} унікальних замовників"))

            flags_html = "".join(f'<span class="flag {f}">{t}</span>' for f, t in flags)
            st.markdown(f"""
            <div class="card-sm">
                <div class="sec" style="margin-top:0;">Індикатори ризику</div>
                {flags_html}
            </div>
            """, unsafe_allow_html=True)

            if len(sup_rels) > 0:
                st.markdown('<div class="sec">Зв\'язки з замовниками</div>', unsafe_allow_html=True)
                disp = col_exists(sup_rels, "buyer_id", "num_tenders",
                                  "single_bid_share", "buyer_win_share", "supplier_income_share")
                st.dataframe(
                    sup_rels[disp].sort_values("num_tenders", ascending=False).reset_index(drop=True)
                    if "num_tenders" in sup_rels.columns else sup_rels[disp],
                    use_container_width=True, hide_index=True, height=200
                )
            else:
                st.markdown(
                    '<div class="sec" style="margin-top:.8rem;">Немає зв\'язків у relations для цього підрядника</div>',
                    unsafe_allow_html=True
                )


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Зв'язки · Граф
# relations cols: buyer_id, supplier_id, num_tenders,
#                 single_bid_share, buyer_win_share, supplier_income_share
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    rel = relations.copy()

    if "risk_score" not in rel.columns:
        rel["risk_score"] = build_relation_risk_score(rel)

    if rel.empty:
        st.markdown("""
        <div class="card" style="text-align:center;padding:4rem;border-style:dashed;">
            <div style="font-size:2rem;">📭</div>
            <div style="font-family:'Unbounded',sans-serif;font-size:.85rem;margin-top:.5rem;">
                У relations немає даних для побудови графа
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        R_MAX = float(rel["risk_score"].max()) or 1.0

        ctrl, graph = st.columns([1, 3], gap="large")

        with ctrl:
            st.markdown('<div class="sec">Фільтри графа</div>', unsafe_allow_html=True)

            r_min_v = float(rel["risk_score"].min())
            r_max_v = float(rel["risk_score"].max())
            step = round((r_max_v - r_min_v) / 100, 4) or 0.001

            min_risk = st.slider(
                "Мін. ризик-скор", r_min_v, r_max_v,
                round(r_min_v + (r_max_v - r_min_v) * 0.25, 4),
                step=step, key="rel_risk", format="%.3f"
            )

            t_max_v = int(rel["num_tenders"].max()) if "num_tenders" in rel.columns else 50
            min_tend = st.slider("Мін. к-сть тендерів", 1, max(t_max_v, 2),
                                 min(3, t_max_v), key="rel_tend")

            frel = rel.copy()
            frel = frel[frel["risk_score"] >= min_risk]
            if "num_tenders" in rel.columns:
                frel = frel[frel["num_tenders"] >= min_tend]

            n_pairs = len(frel)
            n_buy = frel["buyer_id"].nunique() if "buyer_id" in frel.columns else 0
            n_sup = frel["supplier_id"].nunique() if "supplier_id" in frel.columns else 0

            st.markdown(f"""
            <div class="card-sm" style="margin-top:.8rem;">
                <div class="stats" style="gap:.5rem;">
                    <div class="stat"><div class="stat-val">{n_pairs}</div><div class="stat-key">Пар</div></div>
                    <div class="stat"><div class="stat-val">{n_buy}</div><div class="stat-key">Замовн.</div></div>
                    <div class="stat"><div class="stat-val">{n_sup}</div><div class="stat-key">Підряд.</div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="sec">Легенда</div>
            <div style="font-size:.7rem;line-height:2.2;">
                🔵 Замовник &nbsp; 🔴 Підрядник<br>
                <span style="color:#e84545;font-weight:700;">━━</span> Високий ризик<br>
                <span style="color:#f59e0b;font-weight:700;">━━</span> Середній ризик<br>
                <span style="color:#3b82f6;font-weight:700;">━━</span> Низький ризик
            </div>
            <div style="font-size:.62rem;color:var(--muted);margin-top:.7rem;line-height:1.7;">
                Товщина ребра = к-сть тендерів<br>
                Ризик = sbs×0.4 + bws×0.3 + sis×0.3<br>
                Наведіть курсор для деталей
            </div>
            """, unsafe_allow_html=True)

        with graph:
            st.markdown("""
            <div style="font-family:'Unbounded',sans-serif;font-size:.9rem;
                        font-weight:700;margin-bottom:.3rem;">
                Граф ризикових зв'язків
            </div>
            """, unsafe_allow_html=True)

            if len(frel) == 0:
                st.markdown("""
                <div class="card" style="text-align:center;padding:4rem;border-style:dashed;">
                    <div style="font-size:2rem;">📭</div>
                    <div style="font-family:'Unbounded',sans-serif;font-size:.85rem;margin-top:.5rem;">
                        Немає зв'язків за фільтрами
                    </div>
                    <div style="font-size:.7rem;color:var(--muted);margin-top:.3rem;">
                        Спробуйте знизити порогові значення
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                CAP = 200
                draw = frel.sort_values("risk_score", ascending=False).head(CAP)
                if len(frel) > CAP:
                    st.markdown(
                        f'<span class="flag f-yellow">Показано топ-{CAP} з {len(frel)} зв\'язків</span>',
                        unsafe_allow_html=True
                    )

            net = Network(height="640px", width="100%",
                          bgcolor="#111318", font_color="#e2e8f0")
            net.set_options("""
            {
              "physics": {
                "barnesHut": {
                  "gravitationalConstant": -7000,
                  "springLength": 150,
                  "damping": 0.2
                },
                "stabilization": { "iterations": 150 }
              },
              "edges": { "smooth": { "type": "dynamic" } },
              "interaction": { "hover": true, "tooltipDelay": 60 }
            }
            """)

            seen = set()
            for _, row in draw.iterrows():
                b_id  = str(row["buyer_id"])
                s_id  = str(row["supplier_id"])
                bkey  = f"B_{b_id}"
                skey  = f"S_{s_id}"

                score = float(row["risk_score"])
                ntend = int(row["num_tenders"]) if "num_tenders" in row.index else 1
                sbs   = fmt_pct(row.get("single_bid_share",     "n/a"))
                bws   = fmt_pct(row.get("buyer_win_share",       "n/a"))
                sis   = fmt_pct(row.get("supplier_income_share", "n/a"))

                r = score / R_MAX
                if r >= .65:
                    ec, ew = "#e84545", 5
                elif r >= .35:
                    ec, ew = "#f59e0b", 3
                else:
                    ec, ew = "#3b82f6", 1.5

                if bkey not in seen:
                    buyer_total = int(
                        frel[frel["buyer_id"].astype(str) == b_id]["num_tenders"].sum()
                    ) if "num_tenders" in frel.columns else 10
                    net.add_node(bkey,
                        label=b_id[:14],
                        color={"background":"#0f2744","border":"#3b82f6",
                               "highlight":{"background":"#1d4ed8","border":"#93c5fd"}},
                        size=max(14, min(30, buyer_total // 3)),
                        shape="dot",
                        title=f"<b>🏛 Замовник</b><br>{b_id}<br>Всього тендерів: {buyer_total}",
                        font={"color":"#93c5fd","size":11})
                    seen.add(bkey)

                if skey not in seen:
                    sup_r = suppliers.loc[
                        suppliers["supplier_id"].astype(str) == s_id, "risk_score"
                    ].values if "supplier_id" in suppliers.columns else []
                    sup_score_str = f"{sup_r[0]:.3f}" if len(sup_r) else "n/a"
                    net.add_node(skey,
                        label=s_id[:14],
                        color={"background":"#3b0f0f","border":"#e84545",
                               "highlight":{"background":"#991b1b","border":"#fca5a5"}},
                        size=max(10, min(22, ntend // 2)),
                        shape="dot",
                        title=f"<b>🏢 Підрядник</b><br>{s_id}<br>Risk score: {sup_score_str}",
                        font={"color":"#fca5a5","size":11})
                    seen.add(skey)

                net.add_edge(bkey, skey,
                    value=max(ew, ntend / 8),
                    color={"color": ec, "highlight": "#ffffff", "opacity": 0.85},
                    arrows={"to": {"enabled": True, "scaleFactor": 0.55}},
                    title=(
                        f"<b>⚡ Зв'язок</b><br>"
                        f"Ризик-скор: <b>{score:.3f}</b><br>"
                        f"Тендерів: <b>{ntend}</b><br>"
                        f"Single-bid частка: {sbs}<br>"
                        f"Замовник виграє: {bws}<br>"
                        f"Частка доходу підрядника: {sis}"
                    )
                )

                with tempfile.NamedTemporaryFile(delete=False, suffix=".html",
                                                 mode="w", encoding="utf-8") as tmp:
                    net.save_graph(tmp.name)
                    html_str = Path(tmp.name).read_text(encoding="utf-8")

                st.components.v1.html(html_str, height=650, scrolling=False)

            if len(frel) > 0:
                st.markdown('<div class="sec" style="margin-top:1rem;">Таблиця зв\'язків</div>',
                            unsafe_allow_html=True)
                show = col_exists(frel, "buyer_id", "supplier_id", "num_tenders", "risk_score",
                                  "single_bid_share", "buyer_win_share", "supplier_income_share")
                st.dataframe(
                    frel[show].sort_values("risk_score", ascending=False).reset_index(drop=True),
                    use_container_width=True, hide_index=True, height=250
                )
