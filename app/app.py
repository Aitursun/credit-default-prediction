import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.metrics import fbeta_score, precision_score, recall_score, roc_auc_score

from src.config import ID_COL, MODELS_DIR, PROC_DATA_DIR, TARGET_COL
from src.evaluation import business_metric

from scoring import (
    FEATURE_EXPLANATIONS, ZONE_DATA, EDU_OPTIONS, EDU_MAP,
    load_thresholds, compute_threshold_holdout,
    load_model_from_disk, load_train_stats_from_disk,
    predict_proba_safe, scale_if_needed, build_shap_explainer,
    compute_shap_values, get_ebm_local_importance,
    check_business_rules, get_zone, risk_level_label,
    prepare_single_input, prepare_batch_input,
    generate_excel_report, make_template_csv,
)

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("credit_scoring")

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Кредитный скоринг | КГИПИ",
    page_icon=":material/account_balance:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Глобальные стили (палитра #1E2761) ──────────────────────────────────────
st.markdown("""
<style>
/* ── Заголовки ── */
h1, h2, h3, h4 { color: #1E2761 !important; }

/* ── Сайдбар ── */
section[data-testid="stSidebar"] { background-color: #EAF1FC !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label { color: #1E2761 !important; }

/* ── Основные кнопки ── */
button[kind="primary"] {
    background-color: #1E2761 !important;
    color: #FFFFFF !important;
    border: none !important;
}
button[kind="primary"]:hover { background-color: #44506B !important; }

/* ── Вторичные кнопки ── */
button[kind="secondary"] {
    background-color: #FFFFFF !important;
    color: #1E2761 !important;
    border: 1px solid #1E2761 !important;
}
button[kind="secondary"]:hover { background-color: #EAF1FC !important; }

/* ── Метрики ── */
[data-testid="stMetricLabel"]  { color: #44506B !important; font-size: 13px; }
[data-testid="stMetricValue"]  { color: #1E2761 !important; }
[data-testid="stMetricDelta"]  { color: #7C88A8 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab"] { color: #44506B; border-bottom: 2px solid transparent; }
.stTabs [aria-selected="true"] {
    color: #1E2761 !important;
    border-bottom: 2px solid #1E2761 !important;
    font-weight: 600;
}

/* ── Экспандеры ── */
.streamlit-expanderHeader { color: #1E2761 !important; font-weight: 600; }

/* ── Блоки success/warning/error → фирменная палитра ── */
div[data-testid="stAlert"][data-alert-type="success"] {
    background-color: #EAF1FC !important;
    border-left: 4px solid #1E2761 !important;
    color: #1E2761 !important;
}
div[data-testid="stAlert"][data-alert-type="warning"] {
    background-color: #C3D3F4 !important;
    border-left: 4px solid #44506B !important;
    color: #1E2761 !important;
}
div[data-testid="stAlert"][data-alert-type="error"] {
    background-color: #1E2761 !important;
    border-left: 4px solid #FFFFFF !important;
    color: #FFFFFF !important;
}
div[data-testid="stAlert"][data-alert-type="info"] {
    background-color: #CADCFC !important;
    border-left: 4px solid #7C88A8 !important;
    color: #1E2761 !important;
}

/* ── Caption / мелкий текст ── */
small, .stCaption { color: #7C88A8 !important; }

/* ── Прогресс-бар ── */
.stProgress > div > div { background-color: #1E2761 !important; }

/* ── Слайдер ── */
.stSlider [data-baseweb="slider"] [role="slider"] { background-color: #1E2761 !important; }

/* ── Download button ── */
a[data-testid="stDownloadButton"] button {
    background-color: #CADCFC !important;
    color: #1E2761 !important;
    border: 1px solid #7C88A8 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── UI-константы ────────────────────────────────────────────────────────────

# Порядок по OOF ROC-AUC (ноутбук 04):
# CatBoost 0.786 > EBM 0.785 > LGBM 0.778 > LogReg 0.771 > RF 0.757
MODEL_OPTIONS: dict[str, str] = {
    "CatBoost (рекомендуется)": "catboost",
    "EBM (InterpretML)":        "ebm",
    "LightGBM":                 "lgbm",
    "Logistic Regression":      "logreg",
    "Random Forest":            "rf",
}

DISPLAY_MODES      = ["Полный (для всех)", "Только для специалиста", "Только для аналитика"]
REQUIRED_BATCH_COLS = ["AMT_INCOME_TOTAL", "AMT_ANNUITY"]

# ─── Streamlit-кэшированные обёртки над scoring.py ───────────────────────────

@st.cache_resource
def load_model(name: str):
    try:
        return load_model_from_disk(name)
    except FileNotFoundError as e:
        st.error(str(e)); st.stop()


@st.cache_data
def load_train_stats() -> tuple:
    return load_train_stats_from_disk()


@st.cache_data
def compute_threshold(model_name: str, beta: float = 2.0) -> tuple[float, float]:
    thresholds = load_thresholds()
    if model_name in thresholds:
        entry = thresholds[model_name]
        thr, f2 = float(entry["threshold"]), float(entry.get("f2", 0.0))
        logger.info("Порог [%s] из файла: %.4f  F₂=%.4f", model_name.upper(), thr, f2)
        return thr, f2
    logger.warning("thresholds.json не найден — вычисляю на holdout для %s", model_name.upper())
    model = load_model(model_name)
    thr, f2 = compute_threshold_holdout(model_name, model, beta)
    logger.info("Порог [%s] holdout: %.4f  F₂=%.4f", model_name.upper(), thr, f2)
    return float(thr), float(f2)


@st.cache_resource
def get_shap_explainer(_model, _X_background: pd.DataFrame, model_name: str):
    return build_shap_explainer(_model, _X_background, model_name)


# ─── UI-компоненты ───────────────────────────────────────────────────────────

def show_gauge(prob: float, threshold: float) -> None:
    _, _, zone_color, _, _ = get_zone(prob, threshold)
    t  = threshold
    b1 = t * 0.25 * 100
    b2 = t * 0.65 * 100
    b3 = t * 100
    b4 = (t + 0.20 * (1 - t)) * 100
    b5 = (t + 0.50 * (1 - t)) * 100

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 36, "color": "#1E2761"}},
        title={"text": "<span style='font-size:13px;color:#44506B'>Вероятность дефолта</span>"},
        delta={"reference": threshold * 100, "valueformat": ".1f",
               "increasing": {"color": "#dc3545"}, "decreasing": {"color": "#28a745"}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%",
                     "tickvals": [0, round(b1), round(b2), round(b3), round(b4), round(b5), 100],
                     "tickcolor": "#44506B", "tickfont": {"size": 10}},
            "bar": {"color": zone_color, "thickness": 0.3},
            "steps": [
                {"range": [0,     b1],  "color": "#d4edda"},
                {"range": [b1,    b2],  "color": "#e8f5d0"},
                {"range": [b2,    b3],  "color": "#fff3cd"},
                {"range": [b3,    b4],  "color": "#fde8c8"},
                {"range": [b4,    b5],  "color": "#f8d0c8"},
                {"range": [b5,   100],  "color": "#f8d7da"},
            ],
            "threshold": {"line": {"color": "#1E2761", "width": 3}, "thickness": 0.75, "value": threshold * 100},
        },
    ))
    fig.update_layout(height=260, margin=dict(t=40, b=0, l=10, r=10),
                      paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Зоны (порог {threshold:.1%}): "
        f"0–{b1:.0f}% Очень низкий · {b1:.0f}–{b2:.0f}% Низкий · "
        f"{b2:.0f}–{b3:.0f}% Средний · {b3:.0f}–{b4:.0f}% Пограничный · "
        f"{b4:.0f}–{b5:.0f}% Высокий · >{b5:.0f}% Очень высокий."
    )


def show_factors(shap_vals: np.ndarray, feature_cols: list[str],
                 X_row: pd.DataFrame, mode: str) -> None:
    pairs = sorted(zip(feature_cols, shap_vals, X_row.iloc[0].values),
                   key=lambda x: abs(x[1]), reverse=True)
    risk_items = [(f, s, v) for f, s, v in pairs if s > 0][:5]
    safe_items = [(f, s, v) for f, s, v in pairs if s < 0][:5]

    col_risk, col_safe = st.columns(2)

    def _item(col, feat, shap_val, feat_val, icon):
        info = FEATURE_EXPLANATIONS.get(feat)
        if info:
            rus_name, high_desc, low_desc, unit = info
            desc = high_desc if shap_val > 0 else low_desc
        else:
            rus_name, desc, unit = feat, "", ""
        val_str = f"{feat_val:.3g} {unit}".strip() if not np.isnan(feat_val) else "—"
        with col:
            st.markdown(f"**{icon} {rus_name}**")
            if desc:
                st.markdown(f"<span style='color:#555;font-size:13px'>{desc}</span>", unsafe_allow_html=True)
            if mode in ("Полный (для всех)", "Только для аналитика"):
                st.caption(f"{feat} = {val_str} · SHAP: {shap_val:+.3f}")
            st.markdown("---")

    with col_risk:
        st.markdown("### Факторы риска")
        for f, s, v in risk_items:
            _item(col_risk, f, s, v, ":material/arrow_upward:")
        if not risk_items:
            st.info("Значимых факторов риска не выявлено.")

    with col_safe:
        st.markdown("### Защитные факторы")
        for f, s, v in safe_items:
            _item(col_safe, f, s, v, ":material/arrow_downward:")
        if not safe_items:
            st.info("Защитных факторов не выявлено.")


def show_waterfall(shap_vals: np.ndarray, expected_value: float,
                   feature_cols: list[str], X_row: pd.DataFrame,
                   is_ebm: bool = False) -> None:
    shap_exp = shap.Explanation(
        values=shap_vals, base_values=expected_value,
        data=X_row.iloc[0].values, feature_names=feature_cols,
    )
    plot_caption = (
        "EBM Term Contributions — точный вклад каждого признака (из explain_local)"
        if is_ebm else
        "SHAP Waterfall Plot — вклад каждой переменной в итоговую вероятность"
    )
    val_col_name = "Вклад EBM" if is_ebm else "SHAP"

    col1, col2 = st.columns([3, 2])
    with col1:
        st.caption(plot_caption)
        plt.figure()
        shap.plots.waterfall(shap_exp, max_display=14, show=False)
        st.pyplot(plt.gcf(), use_container_width=True)
        plt.close("all")
    with col2:
        st.caption("Расшифровка переменных:")
        top = sorted(zip(feature_cols, shap_vals), key=lambda x: abs(x[1]), reverse=True)[:12]
        st.dataframe(pd.DataFrame([{
            "Переменная": f, "Название": FEATURE_EXPLANATIONS.get(f, (f,))[0],
            "Влияние": "↑ Риск" if v > 0 else "↓ Безопасно", val_col_name: f"{v:+.3f}",
        } for f, v in top]), hide_index=True, use_container_width=True)


def show_quick_stats(income: float, credit: float, annuity: float, term: int = 0) -> None:
    if income <= 0:
        return
    dti = annuity / income
    cti = credit  / income
    remainder = income - annuity
    cols = st.columns(4) if term > 0 else st.columns(3)
    cols[0].metric("Долговая нагрузка", f"{dti:.1%}",
                   delta="норма" if dti < 0.30 else "превышение",
                   delta_color="normal" if dti < 0.30 else "inverse",
                   help="Платёж / месячный доход. Норма: до 30%.")
    cols[0].caption(f"ANNUITY_INCOME_RATIO = {dti:.4f}")
    cols[1].metric("Кредит к доходу", f"{cti:.0f}×",
                   delta="норма" if cti <= 240 else "высокое",
                   delta_color="normal" if cti <= 240 else "inverse",
                   help="Кредит / месячный доход. Норма: до 240× (20 лет).")
    cols[1].caption(f"CREDIT_INCOME_RATIO = {cti:.4f}")
    cols[2].metric("Остаток после платежа", f"{remainder:,.0f} сом/мес",
                   delta="положит." if remainder > 0 else "отрицат.",
                   delta_color="normal" if remainder > 0 else "inverse",
                   help="Сколько остаётся после выплаты кредита.")
    cols[2].caption("AMT_INCOME_TOTAL − AMT_ANNUITY")
    if term > 0:
        total = annuity * term
        overpay = total - credit
        cols[3].metric("Всего выплат", f"{total:,.0f} сом",
                       delta=f"переплата {overpay:,.0f}" if overpay > 0 else "без переплаты",
                       delta_color="inverse" if overpay > 0 else "normal",
                       help=f"Сумма платежей за {term} мес. без учёта ставки.")
        cols[3].caption(f"ANNUITY_CREDIT_RATIO = {annuity/max(credit,1):.4f}")


def show_recommendation(prob: float, threshold: float) -> None:
    st.markdown("### 💬 Рекомендации кредитному специалисту")
    zone_code = get_zone(prob, threshold)[0]
    if zone_code == "very_low":
        st.success("**Очень низкий риск — авто-одобрение.**  \n"
                   "Базель II, PD < 5% — наивысшая кредитная категория.  \n\n"
                   "Можно предложить увеличение лимита или сниженную ставку.")
    elif zone_code == "low":
        st.success("**Низкий риск — одобрение на стандартных условиях.**  \n"
                   "EBA GL/2020/06 — категория «Удовлетворительный».  \n\n"
                   "Стандартный пакет документов, стандартная ставка.")
    elif zone_code == "medium":
        st.success("**Средний риск — одобрение с дополнительной проверкой.**  \n"
                   "Basel II IRB: требует усиленного мониторинга.  \n\n"
                   "Проверить доход, рассмотреть поручителя или повышенную ставку.")
    elif zone_code == "borderline":
        st.warning("**Пограничная зона — направить на ручную проверку.**  \n"
                   "Базель II (judgmental override): случаи вблизи порога требуют офицера.  \n\n"
                   "Проверить доход, историю платежей, рассмотреть снижение суммы на 20–30%.")
    elif zone_code == "high":
        st.error("**Высокий риск — условный отказ.**  \n"
                 "МВФ/Базель II: категория «Субстандартный».  \n\n"
                 "Пересмотр при залоге ≥ 150% суммы или поручителе.")
    else:
        st.error("**Очень высокий риск — авто-отказ.**  \n"
                 "Национальный банк КР: кредитование без обеспечения не рекомендовано.")


# ─── Сайдбар ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Кредитный скоринг")
    st.caption("ВКР · КГИПИ · 2026")
    st.markdown("---")
    st.markdown("**── Модель ──**")
    model_label = st.radio("Модель", list(MODEL_OPTIONS.keys()), label_visibility="collapsed")
    selected_model_name = MODEL_OPTIONS[model_label]
    st.markdown("---")
    st.markdown("**── Режим отображения ──**")
    display_mode = st.radio("Режим", DISPLAY_MODES, label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**── О системе ──**")
    st.info("ℹ️ Система оценивает вероятность невозврата кредита. "
            "Решение носит **рекомендательный** характер.")
    st.download_button("Скачать шаблон", data=make_template_csv(),
                       file_name="template_scoring.csv", mime="text/csv",
                       icon=":material/download:")

# ─── Загрузка ресурсов ───────────────────────────────────────────────────────

with st.spinner("Загрузка модели и данных…"):
    medians, feature_cols, X_background = load_train_stats()
    model     = load_model(selected_model_name)
    threshold, _ = compute_threshold(selected_model_name)

with st.sidebar:
    st.markdown("---")
    st.markdown("**── Порог отсечения ──**")
    st.metric(f"{selected_model_name.upper()} · F₂ (β=2)", f"{threshold:.4f}",
              help="Оптимизирован по F₂-мере.")

show_mode_specialist = display_mode in ("Полный (для всех)", "Только для специалиста")
show_mode_analyst    = display_mode in ("Полный (для всех)", "Только для аналитика")

# ─── Дефолты и сброс (ДО виджетов) ──────────────────────────────────────────

_DEFAULTS: dict = dict(
    age_input=35, gender_input="Женский", married_input=True,
    children_input=0, family_size_input=2, car_input=False, realty_input=False,
    income_input=150_000, credit_input=450_000, term_input=24, annuity_input=18_750,
    emp_input=3, unemp_input=False, edu_input=EDU_OPTIONS[0],
    use_ext1_input=True, ext1_input=float(round(medians["EXT_SOURCE_1"], 2)),
    use_ext2_input=True, ext2_input=float(round(medians["EXT_SOURCE_2"], 2)),
    use_ext3_input=True, ext3_input=float(round(medians["EXT_SOURCE_3"], 2)),
)

_EMPTY: dict = dict(
    age_input=18, gender_input="Мужской", married_input=False,
    children_input=0, family_size_input=1, car_input=False, realty_input=False,
    income_input=0, credit_input=0, term_input=6, annuity_input=0,
    emp_input=0, unemp_input=False, edu_input=EDU_OPTIONS[0],
    use_ext1_input=True, ext1_input=0.50,
    use_ext2_input=True, ext2_input=0.50,
    use_ext3_input=True, ext3_input=0.50,
)

for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

if st.session_state.pop("_do_reset", False):
    st.session_state.update(_EMPTY)
    st.session_state["_show_result"] = False

# ─── Табы ────────────────────────────────────────────────────────────────────

tab1, tab2 = st.tabs(["Проверить заёмщика", "Загрузить список заявок"])

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — Оценка отдельного заёмщика
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Оценка отдельного заёмщика")

    # ── Пресеты ───────────────────────────────────────────────────────────────
    # Material Icons (Google Material Design — открытый источник, Apache 2.0)
    # https://fonts.google.com/icons
    PRESETS: dict[str, tuple[str, dict]] = {
        "Очень низкий": (":material/check_circle:", dict(
            age_input=50, gender_input="Женский", married_input=True,
            children_input=0, family_size_input=2, car_input=True, realty_input=True,
            income_input=320_000, credit_input=250_000, term_input=25, annuity_input=10_000,
            emp_input=18, unemp_input=False, edu_input="Высшее образование",
            use_ext1_input=True, ext1_input=0.88, use_ext2_input=True, ext2_input=0.90,
            use_ext3_input=True, ext3_input=0.85,
        )),
        "Низкий": (":material/task_alt:", dict(
            age_input=40, gender_input="Женский", married_input=True,
            children_input=0, family_size_input=2, car_input=False, realty_input=True,
            income_input=180_000, credit_input=250_000, term_input=20, annuity_input=12_500,
            emp_input=7, unemp_input=False, edu_input="Высшее образование",
            use_ext1_input=True, ext1_input=0.55, use_ext2_input=True, ext2_input=0.58,
            use_ext3_input=True, ext3_input=0.53,
        )),
        "Средний": (":material/info:", dict(
            age_input=34, gender_input="Женский", married_input=True,
            children_input=1, family_size_input=3, car_input=False, realty_input=False,
            income_input=120_000, credit_input=280_000, term_input=16, annuity_input=18_000,
            emp_input=3, unemp_input=False, edu_input="Среднее / среднее специальное",
            use_ext1_input=True, ext1_input=0.42, use_ext2_input=True, ext2_input=0.44,
            use_ext3_input=True, ext3_input=0.40,
        )),
        "Пограничный": (":material/warning:", dict(
            age_input=28, gender_input="Женский", married_input=False,
            children_input=0, family_size_input=1, car_input=False, realty_input=False,
            income_input=90_000, credit_input=270_000, term_input=15, annuity_input=18_000,
            emp_input=2, unemp_input=False, edu_input="Среднее / среднее специальное",
            use_ext1_input=True, ext1_input=0.32, use_ext2_input=True, ext2_input=0.35,
            use_ext3_input=True, ext3_input=0.30,
        )),
        "Высокий": (":material/error:", dict(
            age_input=25, gender_input="Мужской", married_input=False,
            children_input=1, family_size_input=2, car_input=False, realty_input=False,
            income_input=58_000, credit_input=280_000, term_input=12, annuity_input=23_000,
            emp_input=1, unemp_input=False, edu_input="Неполное среднее",
            use_ext1_input=True, ext1_input=0.12, use_ext2_input=True, ext2_input=0.14,
            use_ext3_input=True, ext3_input=0.11,
        )),
        "Очень высокий": (":material/cancel:", dict(
            age_input=21, gender_input="Мужской", married_input=False,
            children_input=2, family_size_input=3, car_input=False, realty_input=False,
            income_input=35_000, credit_input=240_000, term_input=15, annuity_input=16_000,
            emp_input=0, unemp_input=True, edu_input="Неполное среднее",
            use_ext1_input=True, ext1_input=0.05, use_ext2_input=True, ext2_input=0.06,
            use_ext3_input=True, ext3_input=0.05,
        )),
    }

    # Пастельные цвета карточек (фон, текст) — соответствуют зонам риска
    _PRESET_COLORS = [
        ("#d4edda", "#1a7a3c"),  # Очень низкий — пастельный зелёный
        ("#c8f0d0", "#155724"),  # Низкий       — светло-зелёный
        ("#e8f5d0", "#3d6b0e"),  # Средний      — жёлто-зелёный
        ("#fff3cd", "#856404"),  # Пограничный  — пастельный янтарь
        ("#ffe0b2", "#7a3500"),  # Высокий      — пастельный оранжевый
        ("#f8d7da", "#721c24"),  # Очень высокий— пастельный красный
    ]
    # CSS для карточек — каждый класс привязан к своему цвету
    _preset_styles = "".join(
        f".pcard-{i} button[kind='secondary'] "
        f"{{ background-color:{bg} !important; color:{fg} !important; "
        f"border:1px solid {fg}55 !important; font-weight:600; border-radius:8px; }}"
        f".pcard-{i} button[kind='secondary']:hover "
        f"{{ background-color:{fg}25 !important; }}"
        for i, (bg, fg) in enumerate(_PRESET_COLORS)
    )
    st.markdown(f"<style>{_preset_styles}</style>", unsafe_allow_html=True)

    st.caption("Примеры для всех зон риска:")
    preset_cols = st.columns(6)
    for i, (col, (label, (icon, values))) in enumerate(zip(preset_cols, PRESETS.items())):
        with col:
            st.markdown(f'<div class="pcard-{i}">', unsafe_allow_html=True)
            clicked = st.button(label, use_container_width=True, icon=icon,
                                key=f"preset_btn_{i}")
            st.markdown('</div>', unsafe_allow_html=True)
        if clicked:
            st.session_state.update(values)
            st.session_state.pop("tab1_result", None)
            st.session_state["_show_result"] = False
            st.rerun()

    st.markdown("---")

    # ── Колбэки синхронизации ─────────────────────────────────────────────────
    def _sync_family() -> None:
        n = int(st.session_state.get("children_input", 0))
        m = bool(st.session_state.get("married_input", True))
        min_f = n + (2 if m else 1)
        if int(st.session_state.get("family_size_input", min_f)) < min_f:
            st.session_state.family_size_input = min_f

    def _sync_annuity() -> None:
        c = float(st.session_state.get("credit_input", 450_000))
        t = int(st.session_state.get("term_input", 24))
        st.session_state.annuity_input = max(1, int(c / max(t, 1)))

    # ── Личные данные ─────────────────────────────────────────────────────────
    with st.expander("▼ Личные данные", expanded=True):
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        age      = r1c1.number_input("Возраст", 18, 75, step=1, key="age_input")
        gender   = r1c2.radio("Пол", ["Мужской", "Женский"], horizontal=True, key="gender_input")
        children = r1c3.number_input("Кол-во детей", 0, 10, step=1,
                                     key="children_input", on_change=_sync_family)
        _n_ch   = int(st.session_state.get("children_input", 0))
        _married = bool(st.session_state.get("married_input", True))
        min_fam  = _n_ch + (2 if _married else 1)
        family_size = r1c4.number_input(
            "Размер семьи", min_value=min_fam, max_value=15, key="family_size_input",
            help=f"Авто-минимум: {'дети+супруг+вы' if _married else 'дети+вы'} = {min_fam}",
        )
        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        education  = r2c1.selectbox("Образование", EDU_OPTIONS, key="edu_input")
        married    = r2c2.checkbox("Женат / замужем",  key="married_input",  on_change=_sync_family)
        own_car    = r2c3.checkbox("Есть автомобиль",  key="car_input")
        own_realty = r2c4.checkbox("Есть недвижимость", key="realty_input")

    # ── Финансовые данные ─────────────────────────────────────────────────────
    with st.expander("▼ Финансовые данные", expanded=True):
        fc1, fc2, fc3, fc4 = st.columns(4)
        income  = fc1.number_input("Доход (сом/мес)", min_value=0, step=5_000, key="income_input")
        credit  = fc2.number_input("Сумма кредита (сом)", min_value=0, step=10_000,
                                   key="credit_input", on_change=_sync_annuity)
        term    = fc3.number_input("Срок (мес.)", 6, 360, step=6,
                                   key="term_input", on_change=_sync_annuity,
                                   help="Влияет на авто-расчёт платежа")
        annuity = fc4.number_input("Платёж (сом/мес)", min_value=0, step=1_000,
                                   key="annuity_input",
                                   help="Авто-рассчитывается по кредиту и сроку")
        ec1, ec2 = st.columns(2)
        years_employed = ec1.number_input("Трудовой стаж (лет)", 0, 50, step=1, key="emp_input")
        ec2.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        is_unemployed  = ec2.checkbox("Безработный", key="unemp_input")

    show_quick_stats(float(income), float(credit), float(annuity), int(term))

    # ── Кредитная история ─────────────────────────────────────────────────────
    with st.expander("Кредитная история (необязательно)", expanded=False):
        st.caption("0 = плохая история, 1 = отличная. Снимите галочку, если данных нет.")
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            use_ext1 = st.checkbox("Данные бюро 1", key="use_ext1_input")
            ext1 = st.slider("Скоринг бюро 1", 0.0, 1.0, step=0.01,
                             key="ext1_input", disabled=not use_ext1)
        with sc2:
            use_ext2 = st.checkbox("Данные бюро 2", key="use_ext2_input")
            ext2 = st.slider("Скоринг бюро 2", 0.0, 1.0, step=0.01,
                             key="ext2_input", disabled=not use_ext2)
        with sc3:
            use_ext3 = st.checkbox("Данные бюро 3", key="use_ext3_input")
            ext3 = st.slider("Скоринг бюро 3", 0.0, 1.0, step=0.01,
                             key="ext3_input", disabled=not use_ext3)

    # ── Валидация и кнопки ───────────────────────────────────────────────────
    _missing = [f for f, v in [("доход", income), ("сумма кредита", credit),
                                ("ежемесячный платёж", annuity)] if not v]
    if _missing:
        st.warning(f"Заполните обязательные поля: **{', '.join(_missing)}**")

    btn_col, clear_col = st.columns([4, 1])
    submitted = btn_col.button("Оценить заёмщика", type="primary", icon=":material/search:",
                               use_container_width=True, disabled=bool(_missing))
    if clear_col.button("Сброс", icon=":material/refresh:", use_container_width=True):
        st.session_state["_do_reset"] = True
        st.session_state.pop("tab1_result", None)
        st.rerun()

    # ── Вычисление ────────────────────────────────────────────────────────────
    if submitted:
        form_data = dict(
            age=int(age), gender=gender, married=bool(married),
            children=int(children), family_size=int(family_size),
            own_car=bool(own_car), own_realty=bool(own_realty),
            education=education,
            income=float(income), credit=float(credit), annuity=float(annuity),
            years_employed=float(years_employed), is_unemployed=bool(is_unemployed),
            ext_source_1=float(ext1) if use_ext1 else None,
            ext_source_2=float(ext2) if use_ext2 else None,
            ext_source_3=float(ext3) if use_ext3 else None,
        )

        X_input = prepare_single_input(form_data, medians, feature_cols)
        prob    = float(predict_proba_safe(model, X_input)[0])

        rule_violations = check_business_rules(income, credit, annuity)
        for v in rule_violations:
            logger.warning("БИЗНЕС-ПРАВИЛО НАРУШЕНО: %s", v)

        logger.info(
            "Оценка | %s | prob=%.4f | thr=%.4f | %s | DTI=%.1f%% | доход=%.0f | кредит=%.0f",
            selected_model_name.upper(), prob, threshold,
            "ОТКАЗАТЬ" if (prob >= threshold or rule_violations) else "ОДОБРИТЬ",
            annuity / max(income, 1) * 100, income, credit,
        )

        explainer = get_shap_explainer(model, X_background, selected_model_name)
        shap_vals, expected_value = compute_shap_values(model, selected_model_name, X_input, explainer)
        if shap_vals is None and selected_model_name == "ebm":
            ebm_scores, ebm_names = get_ebm_local_importance(model, X_input)
            if ebm_scores is not None:
                idx_map = {n: i for i, n in enumerate(feature_cols)}
                shap_vals = np.zeros(len(feature_cols))
                for n, s in zip(ebm_names, ebm_scores):
                    if n in idx_map:
                        shap_vals[idx_map[n]] = s
                expected_value = 0.0

        st.session_state["tab1_result"] = dict(
            prob=prob, rule_violations=rule_violations,
            shap_vals=shap_vals, expected_value=expected_value,
            X_input=X_input, income=income, credit=credit, annuity=annuity,
            model_name=selected_model_name,
        )
        st.session_state["_show_result"] = True
        st.rerun()

    # ── Рендер результата ─────────────────────────────────────────────────────
    if st.session_state.get("_show_result", False) and "tab1_result" in st.session_state:
        r = st.session_state["tab1_result"]
        prob            = r["prob"]
        rule_violations = r["rule_violations"]
        shap_vals       = r["shap_vals"]
        expected_value  = r["expected_value"]
        X_input         = r["X_input"]
        _income, _credit, _annuity = r["income"], r["credit"], r["annuity"]

        zone_code, zone_label, zone_color, zone_decision, zone_icon = get_zone(prob, threshold)
        if rule_violations:
            zone_code, zone_label, zone_color = "very_high", "Очень высокий", "#dc3545"
            zone_decision, zone_icon = "АВТО-ОТКАЗ", "✕"

        st.markdown("---")
        banner = f"## {zone_icon}  {zone_decision}"
        if zone_code in ("very_low", "low", "medium"):
            st.success(banner)
        elif zone_code == "borderline":
            st.warning(banner)
        else:
            st.error(banner)

        st.caption(f"Зона риска: **{zone_label}** · Вероятность: **{prob:.1%}** · Порог F₂: **{threshold:.4f}**")

        if rule_violations:
            st.warning("**Автоматический отказ по бизнес-правилам**\n\n"
                       + "\n".join(f"- {v}" for v in rule_violations))

        if show_mode_specialist and show_mode_analyst:
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Зона риска",          zone_label)
            m2.metric("Долговая нагрузка",   f"{_annuity/max(_income,1):.1%}")
            m3.metric("Кредит к доходу",     f"{_credit/max(_income,1):.1f}×")
            m4.metric("Вероятность дефолта", f"{prob:.1%}")
            m5.metric("Порог (F₂, β=2)",     f"{threshold:.3f}")
            m6.metric("Отступ от порога",    f"{(prob-threshold)*100:+.1f} п.п.")
        elif show_mode_specialist:
            m1, m2, m3 = st.columns(3)
            m1.metric("Зона риска",        zone_label)
            m2.metric("Долговая нагрузка", f"{_annuity/max(_income,1):.1%}")
            m3.metric("Кредит к доходу",   f"{_credit/max(_income,1):.1f}×")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Вероятность дефолта", f"{prob:.1%}")
            m2.metric("Порог (F₂, β=2)",     f"{threshold:.3f}")
            m3.metric("Отступ от порога",    f"{(prob-threshold)*100:+.1f} п.п.")

        show_gauge(prob, threshold)
        st.markdown("---")
        st.markdown("### Почему такое решение?")
        if shap_vals is not None:
            show_factors(shap_vals, feature_cols, X_input, display_mode)
        else:
            st.info("Объяснение факторов недоступно для данной модели.")

        if show_mode_analyst and shap_vals is not None and expected_value is not None:
            is_ebm = r["model_name"] == "ebm"
            expander_label = (
                "Детальный анализ (EBM — встроенная интерпретация)"
                if is_ebm else
                "Детальный SHAP-анализ"
            )
            with st.expander(expander_label, expanded=(display_mode == "Только для аналитика")):
                if is_ebm:
                    st.info(
                        "EBM (Explainable Boosting Machine) интерпретируема по архитектуре: "
                        "каждый признак вносит независимый вклад `f_i(x_i)`. "
                        "Ниже показаны точные term-вклады из `explain_local()` — без SHAP."
                    )
                show_waterfall(shap_vals, expected_value, feature_cols, X_input, is_ebm=is_ebm)
                st.caption(
                    f"Модель: {r['model_name'].upper()} | "
                    f"{'Вклады EBM (term contributions)' if is_ebm else 'SHAP values'} | "
                    f"Порог: {threshold:.4f} | P(дефолт): {prob:.4f}"
                )

        st.markdown("---")
        show_recommendation(prob, threshold)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — Пакетная оценка
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Пакетная оценка заявок")
    st.info(
        "**Пакетная оценка заявок**\n\n"
        "Загрузите CSV или Excel. Система оценит риск по каждой заявке.\n\n"
        f"Обязательные колонки: **{', '.join(REQUIRED_BATCH_COLS)}**. "
        "Остальные будут заполнены средними значениями по базе."
    )

    up_col, dl_col = st.columns([3, 1])
    with up_col:
        uploaded = st.file_uploader("Выберите файл", type=["csv", "xlsx", "xls"])
    with dl_col:
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button("Скачать шаблон", data=make_template_csv(),
                           file_name="template_scoring.csv", mime="text/csv")

    if uploaded is not None:
        try:
            raw_df = pd.read_excel(uploaded) if uploaded.name.endswith((".xlsx", ".xls")) else pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Не удалось прочитать файл.\n\n{e}"); st.stop()
        if raw_df.empty:
            st.error("Файл пустой."); st.stop()

        st.success(f" **{uploaded.name}** · {len(raw_df):,} заявок · {len(raw_df.columns)} колонок")
        st.dataframe(raw_df.head(5), use_container_width=True)

        miss = [c for c in REQUIRED_BATCH_COLS if c not in raw_df.columns]
        if miss:
            st.warning(f"Отсутствуют: **{', '.join(miss)}** — будут заполнены средними.")

        if st.button("Запустить оценку", type="primary", icon=":material/play_arrow:", use_container_width=True):
            progress = st.progress(0, text="Подготовка данных…")
            try:
                X_batch = prepare_batch_input(raw_df, medians, feature_cols)
                progress.progress(40, text="Предсказание…")
                probs = predict_proba_safe(model, X_batch)
                progress.progress(80, text="Результаты…")

                result_df = raw_df.copy()
                result_df["default_prob"]     = probs
                result_df["default_prob_pct"] = (probs * 100).round(1)
                result_df["risk_level"]       = [risk_level_label(p, threshold) for p in probs]
                result_df["decision"]         = ["Отказать" if p >= threshold else "Одобрить" for p in probs]
                if ID_COL not in result_df.columns:
                    result_df.insert(0, ID_COL, range(1, len(result_df) + 1))

                # Бизнес-правила на батч
                if "AMT_INCOME_TOTAL" in raw_df.columns and "AMT_ANNUITY" in raw_df.columns:
                    _inc  = raw_df["AMT_INCOME_TOTAL"].clip(lower=1)
                    _ann  = raw_df["AMT_ANNUITY"]
                    _cred = (raw_df["AMT_CREDIT"] if "AMT_CREDIT" in raw_df.columns
                             else raw_df.get("AMT_GOODS_PRICE",
                                             pd.Series(medians.get("AMT_GOODS_PRICE", 0), index=raw_df.index)))
                    _hard = ((_ann / _inc) >= 0.80) | ((_cred / _inc) > 240)
                    if _hard.any():
                        n_h = int(_hard.sum())
                        result_df.loc[_hard.values, "decision"]   = "Отказать"
                        result_df.loc[_hard.values, "risk_level"] = "Очень высокий"
                        logger.warning("Батч: %d строк — бизнес-правила", n_h)
                        st.warning(f"**{n_h} заявок** отклонено по бизнес-правилам (DTI ≥ 80% или кредит > 240× дохода).")

                progress.progress(100, text="Готово!")

            except Exception as e:
                logger.error("Ошибка батч-обработки: %s", e, exc_info=True)
                st.error(f"Ошибка: {e}"); st.stop()

            # ── Сводка ────────────────────────────────────────────────────────
            st.markdown("---")
            n_total    = len(result_df)
            n_approved = (result_df["decision"] == "Одобрить").sum()
            n_rejected = (result_df["decision"] == "Отказать").sum()
            mean_prob  = probs.mean()

            logger.info("Батч | %s | строк=%d | одобрено=%d | отказано=%d | avg_risk=%.1f%%",
                        uploaded.name, n_total, n_approved, n_rejected, mean_prob * 100)

            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.metric("Всего заявок",  f"{n_total:,}")
            sm2.metric("Одобрено",    f"{n_approved:,} ({n_approved/n_total:.1%})")
            sm3.metric("Отказано",    f"{n_rejected:,} ({n_rejected/n_total:.1%})")
            sm4.metric("Средний риск",   f"{mean_prob:.1%}")

            if TARGET_COL in raw_df.columns and show_mode_analyst:
                y_true_b = raw_df[TARGET_COL].values
                y_pred_b = (probs >= threshold).astype(int)
                am1, am2, am3, am4 = st.columns(4)
                am1.metric("ROC-AUC",   f"{roc_auc_score(y_true_b, probs):.4f}")
                am2.metric("F₂-score",  f"{fbeta_score(y_true_b, y_pred_b, beta=2, zero_division=0):.4f}")
                am3.metric("Recall",    f"{recall_score(y_true_b, y_pred_b, zero_division=0):.1%}")
                am4.metric("Precision", f"{precision_score(y_true_b, y_pred_b, zero_division=0):.1%}")

            # Гистограмма
            fig_h = px.histogram(result_df, x="default_prob_pct", color="decision",
                                 color_discrete_map={"Одобрить": "#28a745", "Отказать": "#dc3545"},
                                 nbins=25, title="Распределение по уровню риска",
                                 labels={"default_prob_pct": "Риск (%)", "count": "Заявок"})
            fig_h.add_vline(x=threshold * 100, line_dash="dash", line_color="black",
                            annotation_text=f"Порог {threshold:.1%}", annotation_position="top right")
            st.plotly_chart(fig_h, use_container_width=True)

            # Таблица результатов
            st.markdown("### Результаты оценки")
            if show_mode_specialist and not show_mode_analyst:
                rus = {ID_COL: "ID", "AMT_INCOME_TOTAL": "Доход", "AMT_GOODS_PRICE": "Кредит",
                       "AMT_ANNUITY": "Платёж", "default_prob_pct": "Риск (%)",
                       "risk_level": "Уровень", "decision": "Решение"}
                show_df = result_df[[c for c in rus if c in result_df.columns]].rename(columns=rus)
            else:
                cols_s = [ID_COL, "AMT_INCOME_TOTAL", "AMT_GOODS_PRICE", "AMT_ANNUITY",
                          "default_prob", "default_prob_pct", "risk_level", "decision"]
                show_df = result_df[[c for c in cols_s if c in result_df.columns]]

            def _color_rows(row):
                val = row["Решение"] if "Решение" in row.index else row.get("decision", "")
                return [f"background-color: {'#fff0f0' if val == 'Отказать' else '#f0fff0'}"] * len(row)

            st.dataframe(show_df.style.apply(_color_rows, axis=1), use_container_width=True, height=400)

            # Аналитика
            if show_mode_analyst:
                with st.expander("Аналитика по портфелю", expanded=False):
                    an1, an2 = st.columns(2)
                    with an1:
                        st.subheader("Топ-10 рискованных")
                        top10 = [c for c in [ID_COL, "default_prob_pct", "risk_level", "decision"]
                                 if c in result_df.columns]
                        st.dataframe(result_df.nlargest(10, "default_prob")[top10],
                                     hide_index=True, use_container_width=True)
                    with an2:
                        rc = result_df["risk_level"].value_counts()
                        cmap = {label: color for _, label, color, _, _ in ZONE_DATA}
                        st.plotly_chart(px.pie(values=rc.values, names=rc.index,
                                               color=rc.index, color_discrete_map=cmap,
                                               title="Структура портфеля"), use_container_width=True)

                    if TARGET_COL in raw_df.columns:
                        st.markdown("---")
                        bm = business_metric(y_true_b, y_pred_b)
                        st.info(f"**Бизнес-метрика:**  \n"
                                f"Пропущено дефолтов (FN): **{bm['fn']}**  \n"
                                f"Ложных отказов (FP): **{bm['fp']}**  \n"
                                f"Издержки (FN×1 + FP×5): **{bm['total_cost']:.0f}**")

            # Скачать
            st.markdown("---")
            dl1, dl2 = st.columns(2)
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            dl1.download_button("Скачать Excel", data=generate_excel_report(result_df, threshold, selected_model_name),
                                file_name=f"scoring_{ts}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                type="primary")
            dl2.download_button("Скачать CSV", data=result_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                                file_name=f"scoring_{ts}.csv", mime="text/csv")
