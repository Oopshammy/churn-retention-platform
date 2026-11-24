# ⚡ ChurnShield AI: Telecom Churn Prediction & Retention Intelligence System

> **An Enterprise SaaS Machine Learning & Prescriptive Retention Platform authentically tailored for Indian Telecom Operators. Powered by XGBoost, SHAP Game-Theoretic Explainability, Scikit-Learn, Plotly, and Streamlit in Indian Rupees (₹).**

[![Python](https://img.shields.io/badge/Python-3.14-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.67-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4-EB4034.svg)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-0.54-10B981.svg)](https://shap.readthedocs.io)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E.svg?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Plotly](https://img.shields.io/badge/Plotly-7.1-3F4F75.svg?logo=plotly&logoColor=white)](https://plotly.com)
[![Currency](https://img.shields.io/badge/Currency-INR%20(%E2%82%B9)-10B981.svg)](https://en.wikipedia.org/wiki/Indian_rupee)

---

## 📌 Executive Overview

In the Indian telecom market, subscriber retention hinges on recharge cycle dynamics—such as **plan validity (28-day vs. 84-day vs. annual lock-in)**, **daily data allowances (1 GB/day vs. 2 GB/day vs. bulk pool)**, **voice/SMS pack structures**, and **effective monthly ARPU**.

**ChurnShield AI** integrates authentic Indian telecom recharge attributes directly into its machine learning pipeline, SHAP explainability engine, and interactive "What-If" retention simulator:
1. **Authentic Indian Telecom Recharge Features:** Integrates 7 explicit recharge attributes (`ConnectionType`, `PlanPrice_INR`, `PlanValidityDays`, `DataAllowance_GB`, `DataQuotaMode`, `SMSQuota`, `CallsQuota`, `TenureMonths`) alongside full **Account Demographics** (`Gender`, `SeniorCitizen`, `Partner`, `Dependents`).
2. **Cross-Validated Hyperparameter Tuning:** Tuned via `StratifiedKFold(n_splits=5)` and `RandomizedSearchCV(scoring='f1')` across tree depth, learning rate, subsample, and estimators, achieving **0.910 ROC-AUC** and **0.724 cross-validated F1**.
3. **Decision Threshold Optimization (0.25 to 0.70):** Identifies the optimal F1 decision boundary (`0.64 Cutoff`) on holdout validation data, lifting Precision by **+4.7%** and substantially curbing wasteful retention offers given to non-churners.
4. **Interactive Threshold Sliders (Tab 2 & Tab 3):** Enables retention and data teams to dynamically shift the decision boundary, immediately recalculating Precision, Recall, and Flagged Accounts in real time.
5. **Local SHAP Explainability:** Deconstructs subscriber risk into exact feature attributions—revealing how short 28-day validity, low daily data quotas, or friction-heavy manual recharge channels elevate churn risk.
6. **Interactive "What-If" Retention Simulator:** Enables retention managers to simulate validity extensions (`28D -> 84D`), daily data boosters (`+1 GB/day`), priority tech support onboarding, or UPI AutoPay cashback, calculating **risk reduction (%)**, **gross revenue saved (₹)**, and **net ROI (₹)** in real time.
7. **Enterprise Batch Scoring Matrix:** Scores customer cohorts and ranks high-risk accounts by Annual Recurring Revenue (ARR) at risk in Indian Rupees (₹).

---

## 🏗️ Repository Architecture

```
churn-retention-platform/
│
├── .streamlit/
│   └── config.toml             # Sleek dark-slate SaaS theme configuration
│
├── data/
│   ├── telco_churn.csv         # Full Indian telecom churn dataset (7,043 records)
│   └── telco_test_sample.csv   # Built-in evaluator cohort (500 telecom subscribers with TEL-IDs and recharge features)
│
├── models/
│   ├── churn_pipeline.joblib   # Serialized ColumnTransformer + XGBoost pipeline (0.07 MB)
│   ├── shap_artifacts.joblib   # Fitted TreeExplainer & sample background matrix (0.17 MB)
│   └── metrics.json            # Model benchmark metrics, ROC curves, confusion matrices
│
├── app.py                      # Presentation-grade Streamlit application (4 tabs in INR ₹)
├── train.py                    # End-to-end ML training, winsorization & SHAP export script
├── requirements.txt            # Python dependencies
└── README.md                   # Technical documentation & project guide
```

> **Git LFS Compliant:** All serialized model artifacts are under **0.2 MB** (well below GitHub's 100 MB hard limit and 10 MB LFS threshold), enabling seamless GitHub cloning and cloud deployment.

---

## 🧠 Indian Telecom Recharge Attributes & Engineering

| Attribute | Field Name | Options / Format | Impact on Churn & Customer Lifecycle |
| :--- | :--- | :--- | :--- |
| **1. Connection Type** | `ConnectionType` | `Mobile Data (4G/5G SIM)` vs. `Router / Fiber Broadband (Home/Family)` | Routers exhibit higher stickiness and bulk pools; mobile SIMs experience daily quota sensitivity. |
| **2. Plan Price (₹)** | `PlanPrice_INR` | `₹149` to `₹3,999` (e.g. ₹299, ₹666, ₹899, ₹2,999) | Direct financial outlay; normalized into `MonthlyEffectiveARPU = PlanPrice / (Validity / 30)`. |
| **3. Plan Validity** | `PlanValidityDays` | `28 Days`, `56 Days`, `84 Days`, `90 Days`, `365 Days` | Short 28-day cycles create frequent drop-off friction; 84D & 365D plans substantially lower churn. |
| **4. Data Allowance** | `DataAllowance_GB` | `1.0`, `1.5`, `2.0`, `2.5`, `3.0 GB` (Daily) or `15–1000 GB` (Bulk Pool) | Daily allowances <1.5 GB/day trigger throttling friction; 2+ GB/day and bulk pools foster loyalty. |
| **5. Quota Mode** | `DataQuotaMode` | `Per Day (e.g., 2 GB/day)` vs. `Entire Plan Validity (Bulk / Unlimited Pool)` | Evaluated with `EffectiveMonthlyData_GB` to capture data density. |
| **6. SMS Quota** | `SMSQuota` | `100 SMS / Day`, `300 SMS Total`, `1000 SMS Total`, `No SMS Included` | Important for OTP/banking notifications and two-factor authentication. |
| **7. Voice Calls** | `CallsQuota` | `Unlimited True 4G/5G Calls`, `1,000 Mins/Mo`, `No Voice Calls (Router)` | Unlimited calls establish standard baseline; router data-only packs exclude voice. |
| **8. Service Tenure** | `TenureMonths` | `0 to 72 Months` | Live lifecycle stages: `New User: 0–6m`, `Growing: 6–24m`, `Established: 24–48m`, `Loyal: 48m+`. |
| **9. Account Demographics** | `gender`, `SeniorCitizen`, `Partner`, `Dependents` | Intact profile controls | Family connections and senior citizen assistance programs. |

---

## 📊 Model Benchmark Comparison

| Model Architecture | ROC-AUC | Recall (Churn) | F1-Score | Precision | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Classifier (Selected for ChurnShield AI) 🏆** | **0.9110** | **0.8369** | **0.7162** | **0.6260** | **0.8346** |
| Random Forest Classifier | 0.8964 | 0.8209 | 0.7190 | 0.6396 | 0.8417 |
| Logistic Regression | 0.8523 | 0.7781 | 0.6306 | 0.5301 | 0.7729 |

---

## 🖥️ Streamlit Platform Features

### Tab 1: 🎯 Single Customer XAI & What-If Simulator
- **Interactive Recharge Controls:** Configure Connection Type, Validity, Price (₹), Daily/Bulk Data, SMS, Calls, Tenure, and Demographics.
- **1-Click Presets:**
  - ⚠️ **High-Risk:** Mobile SIM, ₹299 (28 Days), 1.0 GB/day, Manual UPI, 3 months tenure (**99.7% Churn Risk**).
  - ✅ **Loyal:** Router, ₹999 (365 Days), 1000 GB bulk, UPI AutoPay, 42 months tenure (**20.9% Churn Risk**).
- **KPI Strip (₹):** Live Churn Probability %, Risk Tier Badge, and Annual Revenue at Risk in `₹`.
- **Plotly Gauge Meter:** Multi-band speedometer indicating real-time risk zone.
- **SHAP Diverging Bar Chart:** Identifies the top factors pushing risk **UP (Red)** vs. **LOYAL (Green)**.
- **Interactive "What-If" Retention Simulator:**
  - Toggle 1: Upgrade to 84-Day Quarterly Plan (15% tariff discount)
  - Toggle 2: Boost Daily Data (+1 GB/day High-Speed booster)
  - Toggle 3: Activate Complimentary VIP Tech Support + Security Shield
  - Toggle 4: Switch to UPI AutoPay (₹50/mo cashback)
  - Displays instant **Risk Reduction (%)**, **Offer Cost (₹)**, **Gross Revenue Saved (₹)**, and **Net ROI (₹)**!

### Tab 2: 📂 Batch Scoring & ROI Prioritizer
- **Interactive Decision Threshold Slider (0.25–0.70):** Adjust cutoff in real-time, dynamically updating Precision, Recall, F1, and Flagged High-Risk counts.
- **1-Click Evaluator Cohort:** Scores 500 telecom subscribers (`TEL-XXXXXX`) with authentic recharge features.
- **Revenue KPIs in ₹:** Analyzed Subscribers (500), High-Risk Accounts, Monthly ARR Exposure in `₹`, and Projected Revenue Saved in `₹` (assuming 30% retention lift).
- **Interactive Portfolio Scatter Chart:** `Tenure` vs. `Monthly Tariff (₹)` with hover tooltips in `₹`.
- **Exportable Action List:** Filterable table with priority scores and downloadable `.csv`.

### Tab 3: 📊 Model Evaluation & Global Insights
- **Tuned Hyperparameters Card:** Stratified 5-Fold CV metrics and optimal parameter specifications for XGBoost and Random Forest.
- **Precision-Recall Threshold Optimization Curve:** Interactive Plotly curve displaying Precision, Recall, and F1 across 0.25–0.70 candidate thresholds with vertical markers at optimal F1 cutoff (0.64) and operating slider value.
- **Side-by-Side Metric Comparison:** Default Cutoff (0.50) vs. Optimal Cutoff (0.64).
- **Interactive Operating Slider:** Live KPI cards for Operating Cutoff, Precision, Recall, F1, and Flagged Test Accounts.
- **Multi-Model ROC Curves & Confusion Matrix:** Interactive ROC comparisons and confusion matrix heatmap.
- **Global SHAP Feature Importance:** Bar chart displaying top 15 drivers across the subscriber base.

### Tab 4: 🧠 ML Architecture & Methodology
- Architectural deep-dive designed for technical evaluators and AI/ML internship interviewers.

---

## ⚡ Quickstart & Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/churn-retention-platform.git
cd churn-retention-platform

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train models and generate serialized artifacts
python train.py

# 5. Launch the Streamlit platform
streamlit run app.py
```

Open `http://localhost:8501` in your browser to experience the platform.
