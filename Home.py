"""
Repo_6_Air_Quality — Home.py
Author : Mohamed · M3
Dataset: UCI Air Quality · Italy · 2004–2005
Run    : streamlit run Home.py
"""
# =============================================================================
## path = streamlit run "E:\FINAL PROJECTS\P6_Air_Quality_(UCI )_Experments\Home.py" 
# ─────────────────────────────────────────────────────────────────────────────


import pathlib
import streamlit as st
import pandas as pd
import numpy as np

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="Air Quality ML Engine · M3",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── PATHS ─────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).parent
LOGO = ROOT / "M3_logo.png"
DATA = ROOT / "data" / "AirQualityUCI.csv"

# ── CLR PALETTE ───────────────────────────────────────────────
CLR = {
    "primary"  : "#1565c0",
    "success"  : "#2e7d32",
    "warning"  : "#e65100",
    "danger"   : "#c62828",
    "teal"     : "#00695c",
    "accent"   : "#00695c",
    "secondary": "#455a64",
    "light"    : "#e3f2fd",
    "dark"     : "#1a237e",
    "purple"   : "#6a1b9a",
    "amber"    : "#f57f17",
    "pink"     : "#ad1457",
    "indigo"   : "#283593",
    "cyan"     : "#00838f",
    "lime"     : "#558b2f",
    "brown"    : "#4e342e",
    "grey"     : "#546e7a",
    "white"    : "#ffffff",
    "black"    : "#212121",
}

# ── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
/* sidebar */
[data-testid="stSidebar"] { background: #0f1923; }
[data-testid="stSidebar"] * { color: #e0e8f0 !important; }

/* metric cards */
div[data-testid="metric-container"] {
    background: #e3f2fd;
    border-left: 4px solid #1565c0;
    border-radius: 6px;
    padding: 12px 16px;
}

/* page background */
.main { background: #f4f7fb; }

/* hero banner */
.hero {
    background: linear-gradient(135deg, #0f1923 0%, #1565c0 60%, #00695c 100%);
    border-radius: 12px;
    padding: 40px 48px;
    color: white;
    margin-bottom: 24px;
}
.hero h1 { font-size: 2.4rem; font-weight: 800; margin: 0 0 8px; letter-spacing:-1px; }
.hero p  { font-size: 1.05rem; opacity: 0.85; margin: 0; }

/* info cards */
.info-card {
    background: white;
    border-radius: 10px;
    padding: 20px 24px;
    border-left: 5px solid #1565c0;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    margin-bottom: 16px;
}
.info-card h4 { margin: 0 0 8px; color: #0f1923; font-size:1rem; }
.info-card p  { margin: 0; color: #455a64; font-size:0.88rem; line-height:1.6; }

/* tab badge */
.tab-badge {
    display:inline-block;
    background:#1565c0;
    color:white;
    border-radius:4px;
    padding:2px 10px;
    font-size:0.78rem;
    font-weight:600;
    margin-right:6px;
}
.tab-new {
    display:inline-block;
    background:#00695c;
    color:white;
    border-radius:4px;
    padding:2px 8px;
    font-size:0.72rem;
    font-weight:600;
    margin-left:4px;
}
</style>
""", unsafe_allow_html=True)

# ── LOGO IN SIDEBAR ───────────────────────────────────────────
with st.sidebar:
    if LOGO.exists():
        st.image(str(LOGO), width=70)
    st.markdown("---")
    st.markdown("### 🌍 Air Quality · M3")
    st.markdown("UCI Dataset · Italy · 2004–2005")
    st.markdown("---")
    st.markdown("**Navigate:**")
    st.markdown("📊 EDA Dashboard → 13 Tabs")
    st.markdown("🤖 ML Models    → 5 Tabs")
    st.markdown("---")
    st.markdown(
        "<small style='color:#9aaabb'>Mechanical Engineer → Data Analyst<br/>M3 · 2026</small>",
        unsafe_allow_html=True
    )

# ── HERO ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🌍 Air Quality Analysis · ML Engine</h1>
  <p>UCI Air Quality Dataset · Italian Road-Level Sensor Station · March 2004 – April 2005<br/>
  End-to-end EDA · Time Series · Statistical Testing · A/B Test · Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ── LOAD DATA (cached) ────────────────────────────────────────
@st.cache_data
def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA, sep=",", decimal=".")
    # Drop empty trailing columns
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    # Replace -200 sensor null marker
    df.replace(-200, np.nan, inplace=True)
    df.replace(-200.0, np.nan, inplace=True)
    return df

# ── KPI METRICS ───────────────────────────────────────────────
try:
    df_raw = load_raw()
    rows, cols = df_raw.shape
    date_start = "Mar 2004"
    date_end   = "Apr 2005"
    pct_missing = round(df_raw.isnull().mean().mean() * 100, 1)
    sensors     = len([c for c in df_raw.columns if "PT08" in c])

    st.markdown("### 📊 Dataset at a Glance")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📋 Records",      f"{rows:,}")
    c2.metric("📐 Features",     f"{cols}")
    c3.metric("📅 Period",       f"{date_start} → {date_end}")
    c4.metric("❓ Missing Data", f"{pct_missing}%")
    c5.metric("🔬 Sensors",      f"{sensors}")

except Exception:
    st.info("📁 Place `AirQualityUCI.csv` in the `data/` folder then refresh.")

st.markdown("---")

# ── TWO COLUMN LAYOUT ─────────────────────────────────────────
left, right = st.columns([1.1, 1], gap="large")

with left:
    st.markdown("### 🗂 About This Project")
    st.markdown("""
    <div class="info-card">
      <h4>🎯 Business / Research Problem</h4>
      <p>Can we predict urban air pollution levels from low-cost metal-oxide sensors —
      and identify the key drivers of high CO concentrations in a real Italian city?</p>
    </div>
    <div class="info-card" style="border-left-color:#00695c;">
      <h4>🔬 Dataset — UCI Air Quality</h4>
      <p>Hourly sensor readings from a multisensor device deployed in a polluted area
      of an Italian city. Contains ground-truth (GT) measurements from a certified
      reference analyzer alongside 5 metal-oxide sensor responses.</p>
    </div>
    <div class="info-card" style="border-left-color:#e65100;">
      <h4>⚠️ Key Data Challenge</h4>
      <p><b>-200</b> is the sensor null marker (not real data) — replaced with NaN.<br/>
      <b>NMHC(GT)</b> has 90% missing → dropped from analysis.<br/>
      <b>CO(GT), NOx, NO2</b> have ~19% missing → median imputation.</p>
    </div>
    """, unsafe_allow_html=True)

with right:
    st.markdown("### 🗺 Analysis Roadmap")

    tabs_info = [
        ("1",  "Data Overview",              False),
        ("2",  "Univariate Analysis",        False),
        ("3",  "Bivariate Analysis",         False),
        ("4",  "Correlation Analysis",       False),
        ("5",  "Feature Engineering",        False),
        ("6",  "Missing Values & Imputation",False),
        ("7",  "Pollutant Time Trends",      True),
        ("8",  "Weather vs Pollution",       True),
        ("9",  "A/B Test (Rush Hour)",       True),
        ("10", "Time Series Decomposition",  False),
        ("11", "ADF & Statistical Tests",    False),
        ("12", "Multicollinearity / VIF",    False),
        ("13", "Insights & Recommendations", False),
    ]

    for num, name, is_new in tabs_info:
        new_badge = '<span class="tab-new">★ NEW</span>' if is_new else ""
        st.markdown(
            f'<span class="tab-badge">{num}</span> {name} {new_badge}',
            unsafe_allow_html=True
        )

st.markdown("---")

# ── ML TARGETS ────────────────────────────────────────────────
st.markdown("### 🤖 Machine Learning Targets")
m1, m2 = st.columns(2)

with m1:
    st.markdown("""
    <div class="info-card">
      <h4>📈 Regression Target</h4>
      <p><b>CO(GT)</b> — Carbon Monoxide ground truth (mg/m³)<br/>
      Predict hourly CO concentration from sensor readings + weather + time features.</p>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown("""
    <div class="info-card" style="border-left-color:#00695c;">
      <h4>🏷 Classification Target</h4>
      <p><b>high_pollution</b> — Binary (CO > median → 1, else → 0)<br/>
      Predict whether the current hour is a high-pollution event. ~50/50 balanced.</p>
    </div>
    """, unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small style='color:#9aaabb'>"
    "M3 · Air Quality ML Engine · UCI Dataset · 2026 · "
    "Built with Python · Streamlit · Scikit-learn · Plotly"
    "</small></center>",
    unsafe_allow_html=True
)
