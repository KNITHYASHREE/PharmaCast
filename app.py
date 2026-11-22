from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Import custom forecasting functions before loading the model
import forecast_utils as fu


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ART_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ART_DIR / "final_model.joblib"

st.set_page_config(
    page_title="PharmaCast | Pharmaceutical Demand Forecasting",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

PAGES = ["Demand Forecasting", "Inventory Planner"]


# ============================================================
# 2. DESIGN  (online-pharmacy storefront style:
#    white header, teal nav bar, dark-teal search hero,
#    pastel quick cards, white icon-tile grid)
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{
    --bar:#1a6558; --hero:#17403c; --teal:#1a6558; --teal-d:#124c42; --orange:#f26b1d;
    --ink:#1f2a2e; --mut:#68767a; --line:#e6ebea; --bg:#ffffff;
    --c-green:#dff5ec; --c-lav:#ebe7fb; --c-yel:#fcf0d2; --c-pink:#fbe8ee;
    --ok:#0f7b4d; --ok-bg:#e3f5ec; --warn:#9a6700; --warn-bg:#fff4d6; --crit:#b42318; --crit-bg:#fde8e6;
}
html, body, .stApp { font-family:"Inter","Segoe UI",system-ui,sans-serif; color:var(--ink); background:#fff; }
#MainMenu, footer { visibility:hidden; }
header[data-testid="stHeader"] { background:transparent; }
.block-container { max-width:1320px; padding:1rem clamp(1rem,2.5vw,2.4rem) 3rem; }
h1,h2,h3,h4 { color:var(--ink); font-family:"Inter",sans-serif; letter-spacing:-.01em; }
[data-testid="stCaptionContainer"] { color:var(--mut); }

/* Sidebar */
section[data-testid="stSidebar"] { background:#f7faf9; border-right:1px solid var(--line); }
section[data-testid="stSidebar"] h3 { font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; color:var(--mut); font-weight:700; }
.sb-brand { display:flex; align-items:center; gap:.65rem; padding:.3rem 0; }
.sb-brand .nm { font-weight:800; font-size:1.2rem; color:var(--teal); line-height:1.1; }
.sb-brand .tg { font-size:.64rem; letter-spacing:.1em; color:var(--mut); font-weight:600; }

/* Shell: header + nav + hero joined like a storefront top */
.st-key-shell { gap:0 !important; border:1px solid var(--line); border-radius:16px; overflow:hidden; margin-bottom:1.6rem; }
.st-key-shell [data-testid="stMarkdownContainer"] p { margin:0; }
.apbar { display:flex; justify-content:space-between; align-items:center; gap:1rem; flex-wrap:wrap; padding:.85rem 1.4rem; background:#fff; }
.apbar .l { display:flex; align-items:center; gap:1.2rem; flex-wrap:wrap; }
.logo { display:inline-flex; align-items:center; gap:.5rem; font-weight:800; font-size:1.3rem; color:var(--teal); letter-spacing:-.02em; }
.logo .mk { width:36px; height:36px; border-radius:10px; background:var(--teal); color:#fff; display:inline-flex; align-items:center; justify-content:center; }
.addr { display:flex; align-items:center; gap:.5rem; color:var(--ink); }
.addr small { display:block; font-size:.72rem; color:var(--mut); line-height:1.1; }
.addr b { font-size:.92rem; font-weight:600; }
.pill { border:1.5px solid var(--teal); color:var(--teal); border-radius:10px; padding:.5rem 1.2rem; font-weight:650; font-size:.9rem; display:inline-flex; align-items:center; gap:.5rem; }

/* Compact navigation pill (placed below the summary cards) */
.st-key-nav {
    background:transparent !important;
    padding:0 !important;
    margin:1rem 0 1.4rem !important;
}
.st-key-nav [data-testid="stRadio"] > label { display:none; }
.st-key-nav [role="radiogroup"] {
    display:flex !important;
    align-items:center !important;
    justify-content:center !important;
    flex-wrap:wrap !important;
    gap:1rem !important;
    width:max-content !important;
    max-width:100% !important;
    margin:0 auto !important;
    padding:.65rem 1.1rem !important;
    background:#eaf2ef !important;
    border:1px solid #d7e6df !important;
    border-radius:15px !important;
}
.st-key-nav label[data-baseweb="radio"] {
    display:flex !important;
    align-items:center !important;
    padding:.45rem .65rem !important;
    margin:0 !important;
    border:0 !important;
    white-space:nowrap !important;
}
.st-key-nav label[data-baseweb="radio"] > div:first-child {
    display:flex !important;
}
.st-key-nav label[data-baseweb="radio"] p {
    color:#24564a !important;
    font-weight:650 !important;
    font-size:.95rem !important;
}
.st-key-nav label[data-baseweb="radio"]:has(input:checked) p {
    color:#124c42 !important;
}

/* Dark teal hero with search */
.st-key-hero { background:var(--hero); padding:2.2rem 1.5rem 2.6rem; position:relative; isolation:isolate; overflow:hidden; }
.st-key-hero .htitle { color:#fff; text-align:center; font-weight:800; font-size:clamp(1.7rem,3.2vw,2.5rem); letter-spacing:-.025em; line-height:1.2; margin-bottom:1.3rem; }
.st-key-hero .hsub { color:#b9d6d1; text-align:center; margin:-.8rem 0 1.2rem; font-size:.95rem; }
.deco { position:absolute; bottom:0; width:300px; z-index:-1; pointer-events:none; opacity:.95; }
.deco.l { left:2%; } .deco.r { right:2%; }
.st-key-hero [data-testid="stSelectbox"] { max-width:790px; margin:0 auto; }
.st-key-hero div[data-baseweb="select"] > div { min-height:62px; border-radius:10px; background:#fff; font-size:1.05rem; border:0; box-shadow:0 6px 18px rgba(0,0,0,.18); }
@media (max-width:900px){ .deco{display:none} }

/* Pastel quick cards */
.qcard { display:flex; align-items:center; gap:.9rem; border-radius:14px; padding:1.05rem 1.2rem; min-height:100px; }
.qcard .ic { color:var(--teal); flex:none; }
.qcard .t { font-weight:700; font-size:1.05rem; line-height:1.25; }
.qcard .s { font-size:.68rem; letter-spacing:.12em; font-weight:650; color:#55646a; margin-top:.25rem; text-transform:uppercase; }
.qcard .go { margin-left:auto; font-size:1.4rem; color:var(--ink); }
.c-green{background:var(--c-green)} .c-lav{background:var(--c-lav)} .c-yel{background:var(--c-yel)} .c-pink{background:var(--c-pink)}

/* Section titles */
.sec { margin:1.5rem 0 .8rem; }
.sec .t { font-size:1.55rem; font-weight:800; letter-spacing:-.02em; }
.sec .s { font-size:.9rem; color:var(--mut); margin-top:.15rem; }

/* Medicine icon-tile grid (real buttons) */
.st-key-medgrid .stButton > button { justify-content:flex-start; gap:.9rem; background:#fff; color:var(--ink); border:1px solid var(--line); border-radius:12px; min-height:84px; padding:.55rem .7rem; text-align:left; font-weight:600; box-shadow:none; }
.st-key-medgrid .stButton > button::before { content:""; flex:none; width:60px; height:60px; border-radius:10px; background:#f2f4f4 url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23555' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'><path d='M10.5 20.5a4.95 4.95 0 0 1-7-7l9-9a4.95 4.95 0 0 1 7 7z'/><path d='M8.5 8.5l7 7'/></svg>") center/34px no-repeat; }
.st-key-medgrid .stButton > button:hover { border-color:var(--teal); background:#f7fbfa; color:var(--teal); }
.st-key-medgrid .stButton > button p { overflow:hidden; text-overflow:ellipsis; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; white-space:normal; }

/* Page header strip, chips, notice */
.meta { display:flex; gap:.5rem; flex-wrap:wrap; margin-bottom:.9rem; }
.chip { font-size:.78rem; background:#f4f7f6; border:1px solid var(--line); border-radius:8px; padding:.3rem .7rem; color:var(--mut); }
.chip b { color:var(--ink); font-weight:700; }
.notice { display:flex; gap:.7rem; background:var(--warn-bg); border:1px solid #f1d98c; border-left:4px solid #d9a100; border-radius:10px; padding:.8rem 1rem; font-size:.86rem; line-height:1.6; color:#5c4300; margin-bottom:1rem; }

/* Metrics, buttons, widgets */
div[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-top:4px solid var(--teal); border-radius:12px; padding:1rem 1.15rem; min-height:106px; }
div[data-testid="stMetricLabel"] p { font-size:.74rem !important; text-transform:uppercase; letter-spacing:.07em; font-weight:700 !important; color:var(--mut); white-space:normal !important; }
div[data-testid="stMetricValue"] { color:var(--ink); font-weight:800; white-space:normal !important; overflow-wrap:anywhere; }
.stButton > button { background:var(--teal); color:#fff; border:1px solid var(--teal); border-radius:10px; min-height:2.75rem; padding:.5rem 1.2rem; font-weight:650; white-space:normal; }
.stButton > button:hover { background:var(--teal-d); border-color:var(--teal-d); color:#fff; }
.stDownloadButton > button { background:#fff; color:var(--teal); border:1.5px solid var(--teal); border-radius:10px; min-height:2.6rem; font-weight:650; white-space:normal; }
.stDownloadButton > button:hover { background:#e9f5f2; color:var(--teal-d); border-color:var(--teal-d); }
section[data-testid="stSidebar"] .stButton > button { background:#fff; color:var(--teal); border:1.5px solid var(--teal); }
section[data-testid="stSidebar"] .stButton > button:hover { background:#e9f5f2; }
div[data-testid="stDataFrame"], .stPlotlyChart { border:1px solid var(--line); border-radius:12px; padding:.4rem; background:#fff; }
hr { border-color:var(--line); }
.foot { text-align:center; color:var(--mut); font-size:.78rem; margin-top:2rem; line-height:1.8; }
</style>
""", unsafe_allow_html=True)


ICONS = {
    "plus": '<path d="M12 5v14M5 12h14" stroke-width="3"/>',
    "pin": '<path d="M12 21s7-6.2 7-11.5A7 7 0 0 0 5 9.5C5 14.8 12 21 12 21z"/><circle cx="12" cy="9.5" r="2.5"/>',
    "chart": '<polyline points="3 17 9 11 13 15 21 7"/><polyline points="15 7 21 7 21 13"/>',
    "cal": '<rect x="3" y="5" width="18" height="16" rx="2"/><path d="M16 3v4M8 3v4M3 11h18"/>',
    "shield": '<path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6l8-3z"/><path d="M9 12l2 2 4-4"/>',
    "sliders": '<line x1="4" y1="7" x2="20" y2="7"/><circle cx="9" cy="7" r="2"/><line x1="4" y1="17" x2="20" y2="17"/><circle cx="15" cy="17" r="2"/>',
    "box": '<path d="M21 8l-9-5-9 5 9 5 9-5z"/><path d="M3 8v8l9 5 9-5V8"/><path d="M12 13v8"/>',
}


def icon(name, size=22):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg>')


DECO = """
<svg class="deco l" viewBox="0 0 300 200"><g fill="#2d6d64"><circle cx="80" cy="70" r="26"/>
<path d="M30 200c0-48 22-80 50-80s50 32 50 80z"/><circle cx="172" cy="84" r="24"/>
<path d="M124 200c0-44 20-72 48-72s48 28 48 72z"/></g><rect x="196" y="122" width="44" height="58" rx="6" fill="#3b8176"/>
<circle cx="216" cy="40" r="16" fill="#f26b1d"/><path d="M216 32v16M208 40h16" stroke="#fff" stroke-width="4"/></svg>
<svg class="deco r" viewBox="0 0 300 200"><g fill="#2d6d64"><circle cx="110" cy="76" r="26"/>
<path d="M60 200c0-48 22-80 50-80s50 32 50 80z"/><circle cx="204" cy="86" r="24"/>
<path d="M156 200c0-44 20-72 48-72s48 28 48 72z"/></g>
<g transform="rotate(35 238 40)"><rect x="212" y="28" width="52" height="24" rx="12" fill="#fff"/>
<path d="M238 28h14a12 12 0 0 1 0 24h-14z" fill="#f26b1d"/></g><circle cx="56" cy="50" r="16" fill="#7ad6c0" opacity=".8"/></svg>
"""


def header_html(date_text):
    return f"""<div class="apbar">
    <div class="l"><span class="logo"><span class="mk">{icon('plus', 20)}</span>PharmaCast</span>
    <span class="addr">{icon('pin', 22)}<span><small>Data through</small><b>{date_text} ▾</b></span></span></div>
    <span class="pill">{icon('shield', 18)}Research prototype</span></div>"""


def heading(text, sub=None):
    extra = f'<div class="s">{sub}</div>' if sub else ""
    st.markdown(f'<div class="sec"><div class="t">{text}</div>{extra}</div>', unsafe_allow_html=True)


def quick_cards(items):
    for col, (cls, ic, title, sub) in zip(st.columns(len(items), gap="medium"), items):
        col.markdown(
            f'<div class="qcard {cls}"><span class="ic">{icon(ic, 34)}</span><div><div class="t">{title}</div>'
            f'<div class="s">{sub}</div></div><span class="go">›</span></div>',
            unsafe_allow_html=True
        )


def style_fig(fig, height):
    fig.update_layout(
        template="plotly_white", height=height, paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font=dict(family="Inter, sans-serif", color="#1f2a2e", size=12), margin=dict(l=20, r=20, t=70, b=20)
    )
    fig.update_xaxes(gridcolor="#eef2f1", linecolor="#cfd8d6")
    fig.update_yaxes(gridcolor="#eef2f1", linecolor="#cfd8d6")
    return fig


# ============================================================
# 3. DATA LOADING FUNCTIONS
# ============================================================

@st.cache_resource
def load_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_csv(filename, **kwargs):
    file_path = ART_DIR / filename
    if file_path.exists():
        return pd.read_csv(file_path, **kwargs)
    return None


@st.cache_data(show_spinner="Generating pharmaceutical demand forecast...")
def run_forecast(hist, horizon, _model, feature_columns, origin):
    return fu.recursive_forecast(_model, hist, feature_columns, horizon, origin)


def catalog(hist):
    """Medicines ordered by total historical units, with display names."""
    if "Medicine_Name" in hist.columns:
        names = hist.drop_duplicates("Medicine_ID").set_index("Medicine_ID")["Medicine_Name"]
    else:
        names = pd.Series(dtype="object")
    totals = hist.groupby("Medicine_ID")["Units_Sold"].sum().sort_values(ascending=False)
    return pd.DataFrame({"id": totals.index, "name": [names.get(i, i) for i in totals.index]})


# ============================================================
# 4. LOAD & VALIDATE HISTORICAL DATA
# ============================================================

history = load_csv("history.csv", parse_dates=["Week_Start"])

if history is None:
    st.error("history.csv is missing from the artifacts folder.")
    st.stop()

required_columns = {"Week_Start", "Medicine_ID", "Units_Sold"}
missing_columns = required_columns - set(history.columns)

if missing_columns:
    st.error(f"Missing required history columns: {missing_columns}")
    st.stop()

history["Week_Start"] = pd.to_datetime(history["Week_Start"], errors="coerce")
history["Medicine_ID"] = history["Medicine_ID"].astype(str)
history["Units_Sold"] = pd.to_numeric(history["Units_Sold"], errors="coerce")
history = history.dropna(subset=["Week_Start", "Medicine_ID", "Units_Sold"])

if history.empty:
    st.error("No valid historical demand records found.")
    st.stop()


# ============================================================
# 5. NAVIGATION STATE + CALLBACKS
# ============================================================

if "app_started" not in st.session_state:
    st.session_state.app_started = False


def go_dashboard(mid=None):
    if mid is not None:
        st.session_state["sel_med"] = mid
        st.session_state["selected_medicine"] = mid
    st.session_state["page"] = PAGES[0]
    st.session_state.app_started = True


def pick_from_hero():
    mid = st.session_state.get("hero_search")
    if mid:
        go_dashboard(mid)


def back_home():
    st.session_state.app_started = False


def med_grid(cat, n=10, prefix="med"):
    top = cat.head(n)
    with st.container(key="medgrid"):
        for start in range(0, len(top), 5):
            cols = st.columns(5, gap="medium")
            for col, (_, r) in zip(cols, top.iloc[start:start + 5].iterrows()):
                col.button(str(r["name"]), key=f"{prefix}_{r['id']}", on_click=go_dashboard,
                           args=(r["id"],), use_container_width=True)


SIDE_BRAND = f"""<div class="sb-brand"><span class="logo"><span class="mk">{icon('plus', 18)}</span></span>
<div><div class="nm">PharmaCast</div><div class="tg">{{tag}}</div></div></div>"""


# ============================================================
# 6. LANDING PAGE
# ============================================================

if not st.session_state.app_started:
    cat = catalog(history)
    labels = dict(zip(cat["id"], cat["name"]))

    st.sidebar.markdown(SIDE_BRAND.replace("{tag}", "DEMAND INTELLIGENCE"), unsafe_allow_html=True)
    st.sidebar.divider()
    with st.sidebar.expander("Application workflow", expanded=False):
        st.markdown(
            "**1. Prepare data** – bundled history or updated workbook.\n\n"
            "**2. Configure** – horizon and demand scenario.\n\n"
            "**3. Forecast** – weekly medicine demand.\n\n"
            "**4. Plan inventory** – stock levels and reorder advice."
        )
    st.sidebar.caption("Research prototype · Decision-support tool")

    with st.container(key="shell"):
        st.markdown(header_html(history["Week_Start"].max().date()), unsafe_allow_html=True)
        with st.container(key="hero"):
            st.markdown(DECO + '<div class="htitle">Forecast Medicine Demand and Essentials</div>', unsafe_allow_html=True)
            st.selectbox(
                "Search medicines", cat["id"].tolist(), index=None,
                placeholder="🔍  Search a medicine to see its demand forecast",
                format_func=lambda mid: f"{labels.get(mid, mid)} ({mid})",
                key="hero_search", on_change=pick_from_hero, label_visibility="collapsed"
            )

    quick_cards([
        ("c-green", "chart", f"{len(cat)} medicines<br>tracked", "Demand forecasting"),
        ("c-lav", "cal", "Weekly forecasts<br>up to 12 weeks", "Choose horizon"),
        ("c-yel", "box", "Reorder<br>recommendations", "Inventory planning"),
        ("c-pink", "sliders", "Scenario<br>analysis", "Demand multiplier"),
    ])

    heading("Browse by Medicine", "Top medicines by historical units sold. Select one to open its forecast.")
    med_grid(cat, 10, "home")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    _, launch_col, _ = st.columns([1.3, 1, 1.3])
    launch_col.button("Open forecasting dashboard  →", type="primary", use_container_width=True, on_click=go_dashboard)

    st.markdown(
        '<div class="foot">PharmaCast · Dengue-associated pharmaceutical demand forecasting<br>'
        'Research prototype. Decision-support tool; verify outputs before purchasing decisions.</div>',
        unsafe_allow_html=True
    )
    st.stop()


# ============================================================
# 7. LOAD MODEL
# ============================================================

try:
    bundle = load_bundle()
    model = bundle["model"]
    FEATURES = bundle["feature_columns"]
    ORIGIN = pd.Timestamp(bundle["origin"])
except Exception as e:
    st.error("Unable to load the forecasting model.")
    st.exception(e)
    st.stop()


# ============================================================
# 8. LOAD SUPPORTING ARTIFACTS
# ============================================================

resid_q = load_csv("residual_quantiles.csv", index_col=0)
sigma_df = load_csv("sigma.csv", index_col=0)
stock_default = load_csv("stock_inputs.csv")


# ============================================================
# 9. SIDEBAR CONTROLS
# ============================================================

st.sidebar.markdown(SIDE_BRAND.replace("{tag}", "FORECASTING SYSTEM"), unsafe_allow_html=True)
st.sidebar.divider()
st.sidebar.button("← Back to home", use_container_width=True, on_click=back_home)

with st.sidebar.expander("How to use PharmaCast", expanded=False):
    st.markdown(
        "**1. Prepare data** – bundled history or updated workbook.\n\n"
        "**2. Configure** – horizon and demand scenario.\n\n"
        "**3. Explore** – search a medicine in the hero bar.\n\n"
        "**4. Plan inventory** – open the Inventory Planner."
    )

st.sidebar.subheader("Forecast settings")

source = st.sidebar.radio("Historical Data", ["Bundled history", "Upload updated workbook"])

if source == "Upload updated workbook":
    uploaded_file = st.sidebar.file_uploader(
        "Upload Excel workbook", type=["xlsx"],
        help="Workbook must contain the Master_Weekly sheet."
    )

    if uploaded_file is not None:
        try:
            uploaded_history = fu.load_history_from_workbook(uploaded_file)
            missing_upload = required_columns - set(uploaded_history.columns)

            if missing_upload:
                st.sidebar.error(f"Missing columns: {missing_upload}")
            else:
                uploaded_history["Week_Start"] = pd.to_datetime(uploaded_history["Week_Start"], errors="coerce")
                uploaded_history["Medicine_ID"] = uploaded_history["Medicine_ID"].astype(str)
                uploaded_history["Units_Sold"] = pd.to_numeric(uploaded_history["Units_Sold"], errors="coerce")
                uploaded_history = uploaded_history.dropna(subset=["Week_Start", "Medicine_ID", "Units_Sold"])

                if uploaded_history.empty:
                    st.sidebar.error("No valid records in uploaded workbook.")
                else:
                    history = uploaded_history
                    st.sidebar.success("Updated workbook loaded successfully.")
        except Exception as e:
            st.sidebar.error(f"Workbook loading failed: {e}")

horizon = st.sidebar.slider("Forecast horizon (weeks)", min_value=1, max_value=12, value=8)

multiplier = st.sidebar.slider(
    "Demand scenario multiplier", min_value=0.5, max_value=3.0, value=1.0, step=0.05,
    help="1.0 = baseline demand. 1.5 = 50% higher demand."
)

st.sidebar.divider()
st.sidebar.subheader("Inventory settings")

service_level = st.sidebar.select_slider(
    "Target service level", options=[0.80, 0.85, 0.90, 0.95, 0.975, 0.99], value=0.90
)

review_period = st.sidebar.slider("Review period (weeks)", min_value=1, max_value=4, value=1)

st.sidebar.divider()
st.sidebar.caption("PharmaCast | Dengue-associated pharmaceutical demand forecasting")


# ============================================================
# 10. GENERATE FORECAST
# ============================================================

try:
    with st.spinner("Preparing demand forecasts..."):
        forecast_data = run_forecast(history, horizon, model, FEATURES, ORIGIN)

        if forecast_data is None or forecast_data.empty:
            st.error("Forecast generation returned no data.")
            st.stop()

        if resid_q is not None:
            forecast_data = fu.add_intervals(forecast_data, resid_q)
except Exception as e:
    st.error("Forecast generation failed.")
    st.exception(e)
    st.stop()

forecast_data["Medicine_ID"] = forecast_data["Medicine_ID"].astype(str)

if "Medicine_Name" in history.columns:
    medicine_names = history.drop_duplicates("Medicine_ID").set_index("Medicine_ID")["Medicine_Name"]
else:
    medicine_names = pd.Series(dtype="object")

forecast_data["Medicine_Name"] = (
    forecast_data["Medicine_ID"].map(medicine_names).fillna(forecast_data["Medicine_ID"])
)

medicine_ids = sorted(forecast_data["Medicine_ID"].unique())
cat = catalog(history)
cat = cat[cat["id"].isin(medicine_ids)]


# ============================================================
# 11. HEADER AND SEARCH HERO
# ============================================================

# Read the current page before drawing the hero. The navigation widget is
# intentionally rendered lower down, in the space previously used by the warning.
page = st.session_state.get("page", PAGES[0])

with st.container(key="shell"):
    st.markdown(header_html(history["Week_Start"].max().date()), unsafe_allow_html=True)

    with st.container(key="hero"):
        if page == PAGES[0]:
            st.markdown(DECO + '<div class="htitle">Search Medicines and Forecast Demand</div>', unsafe_allow_html=True)

            # Keep the chosen medicine when switching pages
            saved = st.session_state.get("sel_med")
            if "selected_medicine" not in st.session_state and saved in medicine_ids:
                st.session_state["selected_medicine"] = saved
            if st.session_state.get("selected_medicine") not in medicine_ids:
                st.session_state.pop("selected_medicine", None)

            if medicine_ids:
                selected_medicine = st.selectbox(
                    "Search medicines", medicine_ids, key="selected_medicine",
                    format_func=lambda mid: f"{medicine_names.get(mid, mid)} ({mid})",
                    label_visibility="collapsed"
                )
                st.session_state["sel_med"] = selected_medicine
        else:
            st.markdown(
                DECO + '<div class="htitle">Inventory Planning and Reorder Advice</div>'
                '<div class="hsub">Review stock position, reorder points and recommended purchase quantities.</div>',
                unsafe_allow_html=True
            )

quick_cards([
    ("c-green", "chart", f"{len(medicine_ids)} medicines<br>forecast", "Demand forecasting"),
    ("c-lav", "cal", f"{horizon}-week<br>horizon", f"Data through {history['Week_Start'].max().date()}"),
    ("c-yel", "shield", f"Service level<br>{service_level * 100:.1f}%", f"Review period {review_period} wk"),
    ("c-pink", "sliders", f"Scenario<br>{multiplier:.2f}x", "Demand multiplier"),
])

# Main page navigation now sits here instead of the large inventory warning.
with st.container(key="nav"):
    page = st.radio(
        "Page", PAGES, horizontal=True, key="page",
        label_visibility="collapsed"
    )


# ============================================================
# PAGE 1: DEMAND FORECASTING
# ============================================================

if page == PAGES[0]:

    if not medicine_ids:
        st.info("No medicines available.")
    else:
        heading("Browse by Medicine", "Quickly switch to another medicine.")
        med_grid(cat, 10, "dash")

        actual_data = (
            history[history["Medicine_ID"] == selected_medicine].sort_values("Week_Start").tail(30)
        )

        selected_forecast = (
            forecast_data[forecast_data["Medicine_ID"] == selected_medicine]
            .sort_values("Forecast_Week").copy()
        )

        for column in ["Forecast", "Lower", "Upper"]:
            if column in selected_forecast.columns:
                selected_forecast[column] = selected_forecast[column] * multiplier

        total_predicted = selected_forecast["Forecast"].sum()
        average_predicted = selected_forecast["Forecast"].mean()
        last_actual = actual_data["Units_Sold"].iloc[-1] if not actual_data.empty else 0

        # Demand signal: forecast average vs the last 8 actual weeks
        recent_avg = actual_data["Units_Sold"].tail(8).mean() if not actual_data.empty else 0
        trend_pct = ((average_predicted - recent_avg) / recent_avg * 100) if recent_avg else None

        heading(f"{medicine_names.get(selected_medicine, selected_medicine)}", "Demand forecast summary")

        m1, m2, m3 = st.columns(3)
        m1.metric("Total forecast demand", f"{total_predicted:,.1f} units")
        m2.metric(
            "Average weekly demand", f"{average_predicted:,.1f} units",
            delta=f"{trend_pct:+.1f}% vs last 8 weeks" if trend_pct is not None else None
        )
        m3.metric("Latest actual demand", f"{last_actual:,.1f} units")

        st.divider()
        heading("Historical vs predicted demand")

        chart = go.Figure()
        chart.add_trace(go.Scatter(
            x=actual_data["Week_Start"], y=actual_data["Units_Sold"],
            name="Historical demand", mode="lines+markers",
            line=dict(color="#1a6558", width=3), marker=dict(size=6)
        ))

        if (
            {"Lower", "Upper"}.issubset(selected_forecast.columns)
            and selected_forecast["Lower"].notna().any()
            and selected_forecast["Upper"].notna().any()
        ):
            chart.add_trace(go.Scatter(
                x=list(selected_forecast["Forecast_Week"]) + list(selected_forecast["Forecast_Week"])[::-1],
                y=list(selected_forecast["Upper"]) + list(selected_forecast["Lower"])[::-1],
                fill="toself", fillcolor="rgba(242,107,29,0.14)",
                line=dict(width=0), name="Approx. prediction interval", hoverinfo="skip"
            ))

        chart.add_trace(go.Scatter(
            x=selected_forecast["Forecast_Week"], y=selected_forecast["Forecast"],
            name="Predicted demand", mode="lines+markers",
            line=dict(color="#f26b1d", width=3, dash="dash"), marker=dict(size=8)
        ))

        chart.update_layout(
            title={"text": f"{medicine_names.get(selected_medicine, selected_medicine)} — demand forecast", "x": 0.02},
            xaxis_title="Week", yaxis_title="Units sold", hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        style_fig(chart, 470)
        chart.update_layout(margin=dict(l=20, r=20, t=80, b=20))
        st.plotly_chart(chart, use_container_width=True)

        st.caption(
            "Prediction intervals are approximate, based on historical out-of-fold "
            "forecast errors. They may not capture a new or unusually large dengue outbreak."
        )

        st.divider()
        heading("Weekly forecast details")

        table_columns = [
            c for c in ["Medicine_Name", "Medicine_ID", "Forecast_Week", "Forecast", "Lower", "Upper"]
            if c in selected_forecast.columns
        ]
        forecast_table = selected_forecast[table_columns].copy()

        for column in ["Forecast", "Lower", "Upper"]:
            if column in forecast_table.columns:
                forecast_table[column] = forecast_table[column].round(2)

        st.dataframe(forecast_table, use_container_width=True, hide_index=True)

        st.download_button(
            label="Download selected medicine forecast (CSV)",
            data=forecast_table.to_csv(index=False).encode("utf-8"),
            file_name=f"forecast_{selected_medicine}.csv", mime="text/csv"
        )

        st.divider()
        heading("All medicines forecast summary")

        all_forecasts = forecast_data.copy()
        all_forecasts["Forecast"] = all_forecasts["Forecast"] * multiplier

        summary = (
            all_forecasts.groupby(["Medicine_ID", "Medicine_Name"], as_index=False)["Forecast"]
            .sum()
            .rename(columns={"Forecast": "Total_Forecast_Units"})
            .sort_values("Total_Forecast_Units", ascending=False)
        )
        summary["Total_Forecast_Units"] = summary["Total_Forecast_Units"].round(2)

        top_chart = px.bar(
            summary.head(8), x="Total_Forecast_Units", y="Medicine_Name", orientation="h",
            title="Top medicines by forecast demand", color_discrete_sequence=["#1a6558"]
        )
        top_chart.update_layout(yaxis=dict(autorange="reversed", title=""), xaxis_title="Total forecast units")
        st.plotly_chart(style_fig(top_chart, 380), use_container_width=True)

        st.dataframe(summary, use_container_width=True, hide_index=True)

        st.download_button(
            label="Download all medicines forecast (CSV)",
            data=summary.to_csv(index=False).encode("utf-8"),
            file_name="all_medicines_forecast.csv", mime="text/csv"
        )


# ============================================================
# PAGE 2: INVENTORY PLANNER
# ============================================================

else:

    if stock_default is None:
        st.error(f"stock_inputs.csv not found in {ART_DIR}")
    elif "Medicine_ID" not in stock_default.columns:
        st.error("Medicine_ID column is missing from stock_inputs.csv.")
    else:
        heading("Inventory information", "Edit the available stock and purchasing details below.")

        editable_columns = ["On_Hand", "On_Order", "Lead_Time_Weeks", "MOQ", "Unit_Cost"]

        stock = st.data_editor(
            stock_default, use_container_width=True, num_rows="fixed",
            disabled=[c for c in stock_default.columns if c not in editable_columns],
            key="inventory_editor"
        )
        stock["Medicine_ID"] = stock["Medicine_ID"].astype(str)

        if sigma_df is not None and "sigma" in sigma_df.columns:
            sigma = sigma_df["sigma"].copy()
        else:
            sigma = pd.Series(dtype=float)
        sigma.index = sigma.index.astype(str)

        try:
            plan = fu.inventory_plan(
                forecast_data, stock, sigma, service_level, review_period, multiplier
            )

            st.divider()
            heading("Inventory overview")

            mc = st.columns(4)
            critical_count = int(plan["Status"].astype(str).str.startswith("Critical").sum())
            reorder_count = int((plan["Status"] == "Reorder now").sum())
            total_order = int(plan["Recommended_Order"].sum())

            mc[0].metric("Critical medicines", critical_count)
            mc[1].metric("Reorder now", reorder_count)
            mc[2].metric("Recommended units", f"{total_order:,}")

            if "Est_Order_Cost" in plan.columns and plan["Est_Order_Cost"].notna().any():
                mc[3].metric("Estimated order cost", f"₹{plan['Est_Order_Cost'].sum():,.2f}")
            else:
                mc[3].metric("Medicines monitored", len(plan))

            st.divider()
            heading("Reorder recommendations")

            status_styles = {
                "OK": "background-color:#e3f5ec; color:#0f7b4d; font-weight:600",
                "Reorder now": "background-color:#fff4d6; color:#9a6700; font-weight:600",
            }
            crit_style = "background-color:#fde8e6; color:#b42318; font-weight:600"

            styled_plan = plan.style.apply(
                lambda row: [
                    status_styles.get(row["Status"], crit_style) if column == "Status" else ""
                    for column in plan.columns
                ],
                axis=1
            )
            st.dataframe(styled_plan, use_container_width=True, hide_index=True)

            required_chart_columns = {"Medicine_Name", "Inventory_Position", "Reorder_Point", "Lead_Time_Demand"}

            if required_chart_columns.issubset(plan.columns):
                st.divider()
                heading("Stock level analysis")

                chart_data = plan.melt(
                    id_vars="Medicine_Name",
                    value_vars=["Inventory_Position", "Reorder_Point", "Lead_Time_Demand"],
                    var_name="Measure", value_name="Units"
                )

                inventory_chart = px.bar(
                    chart_data, x="Medicine_Name", y="Units", color="Measure", barmode="group",
                    title="Inventory position vs reorder point vs lead-time demand",
                    color_discrete_sequence=["#1a6558", "#f26b1d", "#d9a100"]
                )
                inventory_chart.update_layout(
                    xaxis_title="Medicine", yaxis_title="Units", legend_title="Inventory measure"
                )
                st.plotly_chart(style_fig(inventory_chart, 450), use_container_width=True)

            st.divider()
            st.download_button(
                label="Download inventory reorder plan (CSV)",
                data=plan.to_csv(index=False).encode("utf-8"),
                file_name="inventory_reorder_plan.csv", mime="text/csv"
            )

        except Exception as e:
            st.error("Unable to generate inventory recommendations.")
            st.exception(e)


# ============================================================
# 12. FOOTER
# ============================================================

st.divider()
st.markdown(
    """<div class="foot">PharmaCast | Dengue-Associated Pharmaceutical Demand Forecasting<br>
    Research Prototype | Machine Learning Based Demand Prediction</div>""",
    unsafe_allow_html=True
)

# CUSTOM CSS
st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #F7FAF9;
    color: #19332F;
}

/* Headings */
h1, h2, h3 {
    color: #193F37 !important;
    line-height: 1.4 !important;
}

/* Buttons */
.stButton > button {
    background-color: #176B5B !important;
    color: white !important;
    border-radius: 9px !important;
    border: none !important;
}

/* Navigation container */
div[data-testid="stRadio"] {
    width: 100% !important;
    margin: 12px 0 22px 0 !important;
}

/* Navigation bar */
div[data-testid="stRadio"] > div[role="radiogroup"] {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: wrap !important;
    align-items: center !important;
    gap: 12px !important;
    width: 100% !important;
    box-sizing: border-box !important;

    background: #EAF2EF !important;
    padding: 10px 16px !important;
    border: 1px solid #DCEAE4 !important;
    border-radius: 12px !important;
}

/* Navigation options */
div[data-testid="stRadio"] label {
    background: transparent !important;
    padding: 9px 14px !important;
    margin: 0 !important;
    border-radius: 8px !important;
    white-space: nowrap !important;
}

/* Navigation text */
div[data-testid="stRadio"] label p,
div[data-testid="stRadio"] label span {
    color: #24564A !important;
    font-weight: 600 !important;
    font-size: 15px !important;
}

/* Input fields */
div[data-baseweb="select"] > div {
    border-radius: 9px !important;
    border-color: #D5E4DE !important;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: white !important;
    border: 1px solid #E1EBE6 !important;
    border-radius: 14px !important;
    padding: 18px !important;
}

/* Responsive headings */
.stMainBlockContainer h1,
.stMainBlockContainer h2,
.stMainBlockContainer h3 {
    max-width: 100% !important;
    overflow-wrap: break-word !important;
}

</style>
""", unsafe_allow_html=True)
