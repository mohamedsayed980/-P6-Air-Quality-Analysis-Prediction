# 🌫️ P6 — Air Quality Analysis & Prediction
**M3 · ML Engine Portfolio · Project 6 of 12**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Dataset](https://img.shields.io/badge/Source-UCI_ML_Repository-4285F4)](https://archive.ics.uci.edu/dataset/360/air+quality)

---

## 📌 Project Overview

End-to-end data analysis and machine learning on the **UCI Air Quality dataset** — hourly readings from a multisensor device at a road intersection in an Italian city, March 2004 – April 2005.

**Core Questions:**
- What drives urban CO pollution — traffic, weather, or time of day?
- Do rush hours produce significantly higher CO? (A/B Test)
- Can low-cost metal-oxide sensors reliably proxy reference analyzers?
- Can we predict CO concentration and classify high-pollution events?

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | UCI Machine Learning Repository |
| Records | 9,357 hourly observations |
| Features | 21 (after cleaning & engineering) |
| Period | March 2004 – April 2005 |
| Location | Italian city road sensor station |
| Null marker | −200 → replaced with NaN in preprocessing |
| Dropped | NMHC(GT) — 90.3% missing |

---

## 🗂 Project Structure

```
📁 Repo_6_Air_Quality/
├── Home.py
├── M3_logo.png
├── requirements.txt
├── README.md
├── data/
│   └── AirQualityUCI_clean.csv   ← pre-cleaned via Jupyter
└── pages/
│   ├── EDA_dashboard.py          ← 13-tab analysis
│   └── ML_Models.py              ← 5-tab ML engine
│
└── Reports/P6_EDA_Tab 1.pdf


```

---

## 📈 EDA Dashboard — 13 Tabs

| # | Tab | Content |
|---|-----|---------|
| 1 | Data Overview | Shape, types, stats, data dictionary |
| 2 | Univariate | Histograms, box plots, skewness |
| 3 | Bivariate | Sensor vs GT scatter, hourly CO bar |
| 4 | Correlation | Heatmap, top predictors of CO(GT) |
| 5 | Feature Engineering | Time flags, rush hour, season, target balance |
| 6 | Missing Values | −200 impact, NMHC drop, imputation |
| 7 ⭐ | Pollutant Trends | Hourly/daily/monthly + 7-day rolling |
| 8 ⭐ | Weather vs Pollution | T, RH, AH vs CO by season |
| 9 ⭐ | A/B Test | Rush hour vs off-peak — Welch + Cohen's d |
| 10 | Time Series Decomp | Trend + Seasonal + Residual |
| 11 | ADF & Stationarity | ADF test + Shapiro-Wilk + QQ plot |
| 12 | Multicollinearity | VIF analysis |
| 13 | Insights & Report | Findings + recommendations + download |

---

## 🤖 ML Models — 5 Tabs

| Tab | Content |
|-----|---------|
| 1 | Training — 6 Regression + 6 Classification models |
| 2 | Regression Results — R², MAE, RMSE |
| 3 | Classification Results — F1, Accuracy, ROC-AUC |
| 4 | Feature Importance |
| 5 | Interactive Prediction |

**Targets:** Regression → `CO(GT)` · Classification → `high_pollution` (CO > median)
**Expected:** R² ≈ 0.90 · F1 ≈ 0.93 (Random Forest / Gradient Boosting)

---

## 🔬 Key Findings

**🚦 Traffic is the Primary Driver**
Rush hours (7–9 AM, 5–7 PM) produce 30–40% more CO than off-peak.
A/B Test: p < 0.001, Cohen's d = 0.639 — highly significant.

**❄️ Winter Effect**
Cold months (Oct–Jan) trap pollutants near ground via thermal inversion.
Temperature negatively correlated with CO (r ≈ −0.45).

**🔬 Sensor Reliability**
Metal-oxide sensors correlate r > 0.85 with reference analyzers.
Suitable for dense city-wide low-cost monitoring networks.

**📊 Stationarity**
All GT pollutants stationary (ADF p ≈ 0.000) — no long-term drift.

---

## 💡 Recommendations

| Priority | Action |
|----------|--------|
| 🚦 High | Congestion pricing / odd-even policy during rush hours |
| ❄️ High | Real-time CO alerts triggered by cold weather + rush hour |
| 🔬 Medium | Expand low-cost PT08 sensor network city-wide |
| 📊 Medium | Deploy ML model for next-hour CO forecasting |

---

## 🚀 How to Run

```bash
git clone https://github.com/YourUsername/Repo_6_Air_Quality.git
cd Repo_6_Air_Quality
pip install -r requirements.txt
streamlit run Home.py
```

> The app loads `AirQualityUCI_clean.csv` automatically from the `data/` folder.
> Run `P6_clean_data.py` in Jupyter first if you need to regenerate the clean file.

---

## 🛠 Tech Stack

`Python 3.11` · `Streamlit` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `Plotly` · `Scikit-learn` · `SciPy` · `Statsmodels`

---

**Mohamed · M3 · ML Engine Portfolio — 12 End-to-End Data Science Projects**
