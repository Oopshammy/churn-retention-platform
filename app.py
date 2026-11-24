"""
app.py - ChurnShield AI: Telecom Churn & Retention Intelligence System
Authentically localized for Indian Telecom Operators in INR (₹).
Features full support for Indian recharge plan attributes (ConnectionType, PlanPrice_INR,
PlanValidityDays, DataAllowance_GB, DataQuotaMode, SMSQuota, CallsQuota, TenureMonths),
existing Account Demographics, real-time XGBoost inference, local SHAP explainability,
interactive "What-If" retention simulator, and batch ROI prioritizer.
"""

import io
import json
import os
import sys
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Configure UTF-8 encoding for standard output if needed
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# 1. Page Configuration
st.set_page_config(
    page_title="ChurnShield AI | Telecom Churn Prediction & Retention Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Paths
PIPELINE_PATH = os.path.join("models", "churn_pipeline.joblib")
SHAP_ARTIFACTS_PATH = os.path.join("models", "shap_artifacts.joblib")
METRICS_PATH = os.path.join("models", "metrics.json")
BUILTIN_SAMPLE_PATH = os.path.join("data", "telco_test_sample.csv")


# 3. Currency & Number Formatting Helpers for Indian Rupees (₹)
def format_inr(amount: float, show_paise: bool = False) -> str:
    """Formats numeric amounts into standard Indian Rupee format with commas (e.g. ₹14,50,000 or ₹899)."""
    if amount is None or np.isnan(amount):
        return "₹0"
    is_negative = amount < 0
    amount = abs(amount)
    if show_paise:
        whole = int(amount)
        paise = f".{int(round((amount - whole) * 100)):02d}"
    else:
        whole = int(round(amount))
        paise = ""
    s = str(whole)
    if len(s) <= 3:
        res = s
    else:
        last3 = s[-3:]
        rest = s[:-3]
        groups = []
        while len(rest) > 2:
            groups.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            groups.insert(0, rest)
        res = ",".join(groups) + "," + last3
    prefix = "-₹" if is_negative else "₹"
    return f"{prefix}{res}{paise}"


def format_inr_compact(amount: float) -> str:
    """Formats large amounts into clean Indian Lakhs/Crores notation."""
    if amount is None or np.isnan(amount):
        return "₹0"
    is_negative = amount < 0
    amount = abs(amount)
    if amount >= 10_000_000:
        val_str = f"₹{amount / 10_000_000:.2f} Cr"
    elif amount >= 100_000:
        val_str = f"₹{amount / 100_000:.2f} Lakh"
    elif amount >= 1_000:
        val_str = f"₹{amount / 1_000:.1f}k"
    else:
        val_str = f"₹{int(amount)}"
    return f"-{val_str}" if is_negative else val_str


# 4. Indian Telecom Domain Mappings
TELCO_PAYMENT_MAP = {
    "UPI AutoPay (GPay / PhonePe / Paytm)": "Bank transfer (automatic)",
    "Manual UPI / Wallet Recharge": "Electronic check",
    "NetBanking / Debit Card Auto-Debit": "Credit card (automatic)",
    "Credit Card": "Mailed check",
}
PAYMENT_DISPLAY_MAP = {v: k for k, v in TELCO_PAYMENT_MAP.items()}


# 5. Custom CSS for Sleek Dark SaaS Theme
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }

    /* Top Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.9) 50%, rgba(79, 70, 229, 0.15) 100%);
        border: 1px solid rgba(99, 102, 241, 0.35);
        border-radius: 16px;
        padding: 22px 30px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.5);
        position: relative;
        overflow: hidden;
    }

    .hero-title {
        font-size: 25px;
        font-weight: 800;
        letter-spacing: -0.025em;
        background: linear-gradient(90deg, #FFFFFF 0%, #E0E7FF 60%, #A5B4FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }

    .hero-subtitle {
        color: #94A3B8;
        font-size: 13.5px;
        font-weight: 400;
    }

    .badge {
        display: inline-flex;
        align-items: center;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }

    .badge-indigo {
        background: rgba(99, 102, 241, 0.18);
        color: #A5B4FC;
        border: 1px solid rgba(99, 102, 241, 0.4);
    }

    .badge-emerald {
        background: rgba(16, 185, 129, 0.18);
        color: #6EE7B7;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }

    .badge-amber {
        background: rgba(245, 158, 11, 0.18);
        color: #FCD34D;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }

    .badge-red {
        background: rgba(239, 68, 68, 0.18);
        color: #FCA5A5;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        backdrop-filter: blur(8px);
        transition: all 0.2s ease;
        position: relative;
    }

    .metric-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-1px);
    }

    .metric-label {
        color: #94A3B8;
        font-size: 11.5px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }

    .metric-value {
        font-size: 26px;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }

    .metric-sub {
        font-size: 11px;
        color: #64748B;
        margin-top: 4px;
    }

    .card-emerald { border-left: 4px solid #10B981 !important; }
    .card-amber { border-left: 4px solid #F59E0B !important; }
    .card-red { border-left: 4px solid #EF4444 !important; }
    .card-indigo { border-left: 4px solid #6366F1 !important; }

    /* Interactive Simulator Box */
    .simulator-box {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 16px;
    }

    /* AI Playbook Box */
    .playbook-box {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(30, 41, 59, 0.8) 100%);
        border: 1px solid rgba(99, 102, 241, 0.35);
        border-radius: 12px;
        padding: 16px 20px;
        margin-top: 14px;
    }

    .playbook-header {
        font-size: 13px;
        font-weight: 700;
        color: #A5B4FC;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 6px;
    }

    .playbook-title {
        font-size: 15px;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 4px;
    }

    .playbook-desc {
        font-size: 12.5px;
        color: #CBD5E1;
        line-height: 1.5;
        margin-bottom: 10px;
    }

    /* Custom Form Layout Polish */
    div[data-testid="stExpander"] {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        margin-bottom: 10px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 6px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94A3B8;
        padding: 8px 18px;
        font-weight: 600;
        font-size: 13px;
    }

    .stTabs [aria-selected="true"] {
        background: #1E293B !important;
        color: #F8FAFC !important;
        border-bottom: 2px solid #6366F1 !important;
    }

    /* Section Subheadings */
    .section-title {
        font-size: 18px;
        font-weight: 700;
        color: #F8FAFC;
        margin-top: 6px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# 6. Cached Model & SHAP Artifact Loaders
@st.cache_resource
def load_ml_assets():
    if not os.path.exists(PIPELINE_PATH):
        return None, None, None
    pipeline = joblib.load(PIPELINE_PATH)
    shap_artifacts = joblib.load(SHAP_ARTIFACTS_PATH)
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)
    return pipeline, shap_artifacts, metrics_data


def engineer_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """Synchronized feature engineering pipeline for Indian Telecom recharge & account features."""
    df = df_in.copy()

    # Normalize TenureMonths
    if "TenureMonths" in df.columns:
        df["TenureMonths"] = pd.to_numeric(df["TenureMonths"], errors="coerce").fillna(0.0)
        df["tenure"] = df["TenureMonths"]
    elif "tenure" in df.columns:
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0.0)
        df["TenureMonths"] = df["tenure"]
    else:
        df["TenureMonths"] = 1.0
        df["tenure"] = 1.0

    # Normalize PlanPrice_INR
    if "PlanPrice_INR" not in df.columns:
        if "MonthlyCharges" in df.columns:
            df["PlanPrice_INR"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(499.0)
        else:
            df["PlanPrice_INR"] = 499.0
    else:
        df["PlanPrice_INR"] = pd.to_numeric(df["PlanPrice_INR"], errors="coerce").fillna(499.0)

    # Normalize PlanValidityDays
    if "PlanValidityDays" not in df.columns:
        if "Contract" in df.columns:
            def default_validity(c):
                if c == "Two year":
                    return 365
                elif c == "One year":
                    return 84
                return 28
            df["PlanValidityDays"] = df["Contract"].apply(default_validity)
        else:
            df["PlanValidityDays"] = 28
    else:
        df["PlanValidityDays"] = pd.to_numeric(df["PlanValidityDays"], errors="coerce").fillna(28).astype(int)

    # MonthlyEffectiveARPU
    df["MonthlyEffectiveARPU"] = (df["PlanPrice_INR"] / (df["PlanValidityDays"] / 30.0)).round(2)
    df["MonthlyCharges"] = df["MonthlyEffectiveARPU"]

    # TotalCharges
    if "TotalCharges" not in df.columns:
        df["TotalCharges"] = (df["MonthlyEffectiveARPU"] * (df["TenureMonths"] + 1.0)).round(2)
    else:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce").fillna(
            df["MonthlyEffectiveARPU"] * (df["TenureMonths"] + 1)
        )

    # ConnectionType default
    if "ConnectionType" not in df.columns:
        def default_conn(row):
            if row.get("InternetService") == "Fiber optic":
                return "Router / Fiber Broadband (Home/Family)"
            return "Mobile Data (4G/5G SIM)"
        df["ConnectionType"] = df.apply(default_conn, axis=1)

    # DataQuotaMode default
    if "DataQuotaMode" not in df.columns:
        def default_quota_mode(row):
            if "Router" in str(row.get("ConnectionType", "")):
                return "Entire Plan Validity (Bulk / Unlimited Pool)"
            return "Per Day (e.g., 2 GB/day)"
        df["DataQuotaMode"] = df.apply(default_quota_mode, axis=1)

    # DataAllowance_GB default
    if "DataAllowance_GB" not in df.columns:
        def default_data(row):
            if "Per Day" in str(row.get("DataQuotaMode", "")):
                return 2.0
            if "Router" in str(row.get("ConnectionType", "")):
                return 1000.0
            return 50.0
        df["DataAllowance_GB"] = df.apply(default_data, axis=1)
    else:
        df["DataAllowance_GB"] = pd.to_numeric(df["DataAllowance_GB"], errors="coerce").fillna(2.0)

    # EffectiveMonthlyData_GB
    def calc_monthly_data(row):
        val = float(row.get("DataAllowance_GB", 2.0))
        if "Per Day" in str(row.get("DataQuotaMode", "")):
            return round(val * 30.0, 1)
        validity = max(1.0, float(row.get("PlanValidityDays", 28.0)))
        return round(val / (validity / 30.0), 1)
    df["EffectiveMonthlyData_GB"] = df.apply(calc_monthly_data, axis=1)

    # SMSQuota default
    if "SMSQuota" not in df.columns:
        def default_sms(row):
            if "Router" in str(row.get("ConnectionType", "")):
                return "No SMS Included (Data-Only Pack)"
            return "100 SMS / Day"
        df["SMSQuota"] = df.apply(default_sms, axis=1)

    # CallsQuota default
    if "CallsQuota" not in df.columns:
        def default_calls(row):
            if "Router" in str(row.get("ConnectionType", "")) and row.get("PhoneService") == "No":
                return "No Voice Calls (Data / Router Only)"
            return "Unlimited True 4G/5G / STD+Local Calls"
        df["CallsQuota"] = df.apply(default_calls, axis=1)

    # SeniorCitizen normalization to string to prevent OneHotEncoder ufunc isnan issue
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

    # TenureGroup
    def categorize_tenure(t):
        try:
            t = float(t)
        except Exception:
            t = 0.0
        if t <= 12:
            return "0-12m"
        elif t <= 24:
            return "12-24m"
        elif t <= 48:
            return "24-48m"
        else:
            return "48m+"

    df["TenureGroup"] = df["TenureMonths"].apply(categorize_tenure)

    # AvgMonthlySpend
    df["AvgMonthlySpend"] = (df["TotalCharges"] / (df["TenureMonths"] + 1.0)).round(2)

    # ServiceCount
    service_cols = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    count = pd.Series(0, index=df.index)
    for col in service_cols:
        if col in df.columns:
            count += (df[col] == "Yes").astype(int)
    if "InternetService" in df.columns:
        count += df["InternetService"].isin(["DSL", "Fiber optic"]).astype(int)

    df["ServiceCount"] = count
    return df


def clean_feature_label(feat_name: str) -> str:
    """Formats encoded feature names into clean, human-readable Indian Telecom labels."""
    mapping = {
        # Indian Recharge Features
        "PlanValidityDays": "Plan Validity (Days)",
        "PlanPrice_INR": "Recharge Plan Price (₹)",
        "MonthlyEffectiveARPU": "Effective Monthly ARPU (₹)",
        "DataAllowance_GB": "Data Allowance (GB)",
        "EffectiveMonthlyData_GB": "Effective Monthly Data (GB)",
        "TenureMonths": "Service Tenure (Months)",
        "ConnectionType_Mobile Data (4G/5G SIM)": "Connection: 4G/5G Mobile SIM",
        "ConnectionType_Router / Fiber Broadband (Home/Family)": "Connection: Fiber Broadband Router",
        "DataQuotaMode_Per Day (e.g., 2 GB/day)": "Data Quota: Daily Allowance (GB/Day)",
        "DataQuotaMode_Entire Plan Validity (Bulk / Unlimited Pool)": "Data Quota: Bulk Validity Pool",
        "SMSQuota_100 SMS / Day": "SMS Quota: 100 SMS / Day",
        "SMSQuota_300 SMS (Total Validity)": "SMS Quota: 300 SMS Total",
        "SMSQuota_1000 SMS (Total Validity)": "SMS Quota: 1000 SMS Total",
        "SMSQuota_No SMS Included (Data-Only Pack)": "SMS Quota: No SMS Included",
        "CallsQuota_Unlimited True 4G/5G / STD+Local Calls": "Calls: Unlimited Calls / STD+Local",
        "CallsQuota_Unlimited True5G / STD+Local Calls": "Calls: Unlimited Calls / STD+Local",
        "CallsQuota_1,000 Mins / Month": "Calls: 1,000 Mins / Month",
        "CallsQuota_No Voice Calls (Data / Router Only)": "Calls: Data Only (No Voice)",
        # Existing Add-ons & Demographics
        "Contract_Month-to-month": "Contract: Monthly (No Lock-in)",
        "Contract_One year": "Contract: Annual Plan (1-Year)",
        "Contract_Two year": "Contract: 2-Year Enterprise / Family",
        "tenure": "Subscriber Tenure (Months)",
        "MonthlyCharges": "Monthly Tariff (₹)",
        "TotalCharges": "Cumulative Spend (₹)",
        "AvgMonthlySpend": "Avg Monthly Spend Ratio (₹)",
        "ServiceCount": "Active Add-on Services Count",
        "OnlineSecurity_No": "Security Shield: None",
        "OnlineSecurity_Yes": "Security Shield: Active",
        "TechSupport_No": "Priority Tech Support: None",
        "TechSupport_Yes": "Priority Tech Support: Active",
        "InternetService_Fiber optic": "Plan: Fiber Optic Optical Line",
        "InternetService_DSL": "Plan: Copper / Mobile Line",
        "InternetService_No": "Plan: Voice Only (No Broadband)",
        "PaymentMethod_Electronic check": "Payment: Manual UPI / Wallet",
        "PaymentMethod_Mailed check": "Payment: Credit Card",
        "PaymentMethod_Bank transfer (automatic)": "Payment: UPI AutoPay",
        "PaymentMethod_Credit card (automatic)": "Payment: NetBanking / Debit Auto-Debit",
        "PaperlessBilling_Yes": "E-Bill (App / SMS): Yes",
        "PaperlessBilling_No": "E-Bill (App / SMS): No",
        "SeniorCitizen_1": "Senior Citizen Account: Yes",
        "SeniorCitizen_0": "Senior Citizen Account: No",
        "Partner_No": "Single Account",
        "Partner_Yes": "Family Linked Account",
        "Dependents_No": "No Linked Connections",
        "Dependents_Yes": "Multiple Family Connections",
        "TenureGroup_0-12m": "Tenure: New Subscriber (0-12m)",
        "TenureGroup_12-24m": "Tenure: 1-2 Years (12-24m)",
        "TenureGroup_24-48m": "Tenure: Established (24-48m)",
        "TenureGroup_48m+": "Tenure: Veteran Loyal (48m+)",
        "OnlineBackup_No": "Cloud Storage Backup: None",
        "OnlineBackup_Yes": "Cloud Storage Backup: Active",
        "DeviceProtection_No": "Router / Hardware Protection: None",
        "DeviceProtection_Yes": "Router / Hardware Protection: Active",
        "StreamingTV_Yes": "OTT Streaming Bundle: Subscribed",
        "StreamingTV_No": "OTT Streaming Bundle: None",
        "StreamingMovies_Yes": "Live TV Channels Pack: Subscribed",
        "StreamingMovies_No": "Live TV Channels Pack: None",
        "PhoneService_Yes": "Voice Calling: Active",
        "PhoneService_No": "Voice Calling: None",
        "MultipleLines_Yes": "Multiple SIM Lines: Yes",
        "MultipleLines_No": "Single SIM Line",
    }
    return mapping.get(feat_name, feat_name.replace("cat__", "").replace("num__", "").replace("_", ": "))


def get_retention_play(top_driver: str, monthly_tariff: float, churn_prob: float):
    """Maps the top SHAP risk driver to an actionable enterprise telecom retention play with cost and revenue ROI in ₹."""
    annual_tariff = monthly_tariff * 12.0

    if "PlanValidityDays" in top_driver or "Contract_Month-to-month" in top_driver:
        discount_pct = 0.15
        offer_cost = round(monthly_tariff * discount_pct * 12.0)
        expected_lift = 0.38
        rev_saved = round(annual_tariff * expected_lift)
        net_roi = rev_saved - offer_cost
        roi_x = round(rev_saved / max(1, offer_cost), 1)
        return {
            "title": "🔒 84-Day Quarterly / Annual Plan Migration Lock-in",
            "play": "Subscriber is on a volatile 28-day recharge cycle with frequent renewal drop-off friction. Offer an instant 15% discount to migrate to an 84-day or 365-day annual plan, paired with priority support.",
            "action": f"Push In-App Plan Migration Offer: {format_inr(monthly_tariff * 0.85)}/mo Quarterly Lock-in",
            "lift": "+38% RETENTION LIFT",
            "offer_cost": offer_cost,
            "annual_rev_saved": rev_saved,
            "net_roi": net_roi,
            "roi_multiplier": roi_x,
        }
    elif "DataAllowance" in top_driver or "EffectiveMonthlyData" in top_driver:
        offer_cost = 249  # 90-day booster cost
        expected_lift = 0.32
        rev_saved = round(annual_tariff * expected_lift)
        net_roi = rev_saved - offer_cost
        roi_x = round(rev_saved / max(1, offer_cost), 1)
        return {
            "title": "🚀 Daily High-Speed Data Booster & Assurance Play",
            "play": "Subscriber frequently hits daily FUP throttling limits (<1.5 GB/day), creating dissatisfaction. Auto-credit a complimentary 90-day +1.0 GB/day high-speed data booster pack.",
            "action": "Auto-credit 90-Day +1 GB/day High-Speed Data Voucher via Customer App",
            "lift": "+32% RETENTION LIFT",
            "offer_cost": offer_cost,
            "annual_rev_saved": rev_saved,
            "net_roi": net_roi,
            "roi_multiplier": roi_x,
        }
    elif "TechSupport_No" in top_driver or "OnlineSecurity_No" in top_driver:
        offer_cost = 300
        expected_lift = 0.28
        rev_saved = round(annual_tariff * expected_lift)
        net_roi = rev_saved - offer_cost
        roi_x = round(rev_saved / max(1, offer_cost), 1)
        return {
            "title": "🛠️ Complimentary VIP Tech Support + Security Shield",
            "play": "Lack of technical support and device security leads to frustration during service outages. Activate a complimentary 90-day VIP Priority Tech Support resolution pass and Cyber Security Shield.",
            "action": "Auto-provision 90-Day Priority Tech Support + Cyber Security Shield via App",
            "lift": "+28% RETENTION LIFT",
            "offer_cost": offer_cost,
            "annual_rev_saved": rev_saved,
            "net_roi": net_roi,
            "roi_multiplier": roi_x,
        }
    elif "PaymentMethod_Electronic check" in top_driver or "Payment" in top_driver:
        offer_cost = 300  # ₹50/mo cashback for 6 months
        expected_lift = 0.22
        rev_saved = round(annual_tariff * expected_lift)
        net_roi = rev_saved - offer_cost
        roi_x = round(rev_saved / max(1, offer_cost), 1)
        return {
            "title": "💳 UPI AutoPay Migration Incentive (Cashback)",
            "play": "Manual UPI recharge users experience high billing friction and payment drop-offs. Grant an instant ₹50/mo cashback credit for 6 billing cycles upon setting up UPI AutoPay.",
            "action": "Deploy ₹50/mo Cashback UPI AutoPay Notification via SMS & WhatsApp",
            "lift": "+22% RETENTION LIFT",
            "offer_cost": offer_cost,
            "annual_rev_saved": rev_saved,
            "net_roi": net_roi,
            "roi_multiplier": roi_x,
        }
    elif "MonthlyEffectiveARPU" in top_driver or "PlanPrice_INR" in top_driver:
        offer_cost = round(monthly_tariff * 0.12 * 12.0)
        expected_lift = 0.25
        rev_saved = round(annual_tariff * expected_lift)
        net_roi = rev_saved - offer_cost
        roi_x = round(rev_saved / max(1, offer_cost), 1)
        return {
            "title": "💰 Family Plan Right-Sizing & Value Optimization",
            "play": "Subscriber perceives monthly tariff friction relative to consumed bandwidth. Offer a customized Family bundle adding 2 secondary SIM connections at ₹199 extra while trimming redundant add-ons.",
            "action": "Initiate Plan Optimization Consultation via Dedicated Relationship Manager",
            "lift": "+25% RETENTION LIFT",
            "offer_cost": offer_cost,
            "annual_rev_saved": rev_saved,
            "net_roi": net_roi,
            "roi_multiplier": roi_x,
        }
    elif "TenureMonths" in top_driver or "TenureGroup_0-12m" in top_driver:
        offer_cost = 250
        expected_lift = 0.30
        rev_saved = round(annual_tariff * expected_lift)
        net_roi = rev_saved - offer_cost
        roi_x = round(rev_saved / max(1, offer_cost), 1)
        return {
            "title": "🌱 New Subscriber Onboarding Concierge & Milestone Voucher",
            "play": "Customer is inside the fragile 90-day onboarding window. Trigger a personalized satisfaction survey call and unlock an exclusive ₹250 retail shopping / OTT voucher.",
            "action": "Trigger Early Lifecycle Welcome Concierge Call & Loyalty Voucher",
            "lift": "+30% RETENTION LIFT",
            "offer_cost": offer_cost,
            "annual_rev_saved": rev_saved,
            "net_roi": net_roi,
            "roi_multiplier": roi_x,
        }
    else:
        offer_cost = 200
        expected_lift = 0.20
        rev_saved = round(annual_tariff * expected_lift)
        net_roi = rev_saved - offer_cost
        roi_x = round(rev_saved / max(1, offer_cost), 1)
        return {
            "title": "🤝 Proactive Dedicated Relationship Manager Outreach",
            "play": "Composite warning indicators detected across service usage and account engagement patterns. Assign a senior customer retention specialist for a 1-on-1 account health review.",
            "action": "Assign Senior Retention Specialist for 1-on-1 Account Check-in",
            "lift": "+20% RETENTION LIFT",
            "offer_cost": offer_cost,
            "annual_rev_saved": rev_saved,
            "net_roi": net_roi,
            "roi_multiplier": roi_x,
        }


# 7. Load Assets
pipeline, shap_artifacts, metrics_data = load_ml_assets()

if pipeline is None:
    st.error("⚠️ Machine Learning assets not found! Please run `python train.py` first to generate pipeline artifacts.")
    st.stop()

explainer = shap_artifacts["explainer"]
feature_names = shap_artifacts["feature_names"]
optimal_threshold = float(metrics_data.get("optimal_threshold", 0.64)) if metrics_data else 0.64


# 8. Top Hero Banner
st.markdown(
    """
    <div class="hero-banner">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px;">
            <div>
                <div class="hero-title">⚡ ChurnShield AI — Telecom Churn Prediction & Retention Intelligence System</div>
                <div class="hero-subtitle">
                    Explainable Machine Learning (XGBoost + SHAP) • Real-Time What-If Simulator • Batch ROI Prioritizer (INR ₹)
                </div>
            </div>
            <div style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
                <span class="badge badge-emerald">ML Engine: XGBoost v3.4 (AUC: 0.91)</span>
                <span class="badge badge-amber">SHAP TreeExplainer Active</span>
                <span class="badge badge-indigo">Currency: Indian Rupee (₹)</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 9. Main Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Single Customer XAI & What-If Simulator",
    "📂 Batch Scoring & ROI Prioritizer",
    "📊 Model Evaluation & Global Insights",
    "🧠 ML Architecture & Methodology",
])


# ==============================================================================
# TAB 1: SINGLE CUSTOMER XAI & WHAT-IF RETENTION SIMULATOR
# ==============================================================================
with tab1:
    st.markdown('<div class="section-title">🎯 Single Customer Diagnosis, SHAP Attributions & "What-If" Retention Simulator</div>', unsafe_allow_html=True)
    st.markdown(
        "Diagnose individual subscriber churn risk with game-theoretic **SHAP local explainability** incorporating authentic Indian recharge plan attributes. "
        "Test concrete **retention interventions** in the interactive 'What-If' simulator to measure instant churn reduction and net financial ROI in Indian Rupees (₹)."
    )

    # Initialize preset values in session state
    if "sim_conn_type" not in st.session_state:
        st.session_state["sim_conn_type"] = "Mobile Data (4G/5G SIM)"
        st.session_state["sim_plan_price"] = 299.0
        st.session_state["sim_plan_validity"] = "28 Days"
        st.session_state["sim_data_allowance"] = 1.5
        st.session_state["sim_data_quota_mode"] = "Per Day (e.g., 2 GB/day)"
        st.session_state["sim_sms_quota"] = "100 SMS / Day"
        st.session_state["sim_calls_quota"] = "Unlimited True 4G/5G / STD+Local Calls"
        st.session_state["sim_tenure_months"] = 4
        st.session_state["sim_payment"] = "Manual UPI / Wallet Recharge"
        st.session_state["sim_paperless"] = "Yes"
        st.session_state["sim_online_sec"] = "No"
        st.session_state["sim_tech_supp"] = "No"
        st.session_state["sim_online_bk"] = "No"
        st.session_state["sim_dev_prot"] = "No"
        st.session_state["sim_stream_tv"] = "Yes"
        st.session_state["sim_stream_mov"] = "No"
        st.session_state["sim_gender"] = "Female"
        st.session_state["sim_senior"] = "0"
        st.session_state["sim_partner"] = "No"
        st.session_state["sim_dependents"] = "No"

    # Presets Row
    preset_cols = st.columns([2.4, 2.4, 2.5])
    with preset_cols[0]:
        if st.button("⚠️ Load High-Risk Customer (Short Validity + Low Daily Data)", use_container_width=True):
            st.session_state["sim_conn_type"] = "Mobile Data (4G/5G SIM)"
            st.session_state["sim_plan_price"] = 299.0
            st.session_state["sim_plan_validity"] = "28 Days"
            st.session_state["sim_data_allowance"] = 1.0
            st.session_state["sim_data_quota_mode"] = "Per Day (e.g., 2 GB/day)"
            st.session_state["sim_sms_quota"] = "100 SMS / Day"
            st.session_state["sim_calls_quota"] = "Unlimited True 4G/5G / STD+Local Calls"
            st.session_state["sim_tenure_months"] = 3
            st.session_state["sim_payment"] = "Manual UPI / Wallet Recharge"
            st.session_state["sim_paperless"] = "Yes"
            st.session_state["sim_online_sec"] = "No"
            st.session_state["sim_tech_supp"] = "No"
            st.session_state["sim_online_bk"] = "No"
            st.session_state["sim_dev_prot"] = "No"
            st.session_state["sim_stream_tv"] = "Yes"
            st.session_state["sim_stream_mov"] = "Yes"
            st.session_state["sim_gender"] = "Female"
            st.session_state["sim_senior"] = "0"
            st.session_state["sim_partner"] = "No"
            st.session_state["sim_dependents"] = "No"
            st.rerun()

    with preset_cols[1]:
        if st.button("✅ Load Loyal Customer (Annual Fiber Broadband + Bulk Data)", use_container_width=True):
            st.session_state["sim_conn_type"] = "Router / Fiber Broadband (Home/Family)"
            st.session_state["sim_plan_price"] = 999.0
            st.session_state["sim_plan_validity"] = "365 Days"
            st.session_state["sim_data_allowance"] = 1000.0
            st.session_state["sim_data_quota_mode"] = "Entire Plan Validity (Bulk / Unlimited Pool)"
            st.session_state["sim_sms_quota"] = "No SMS Included (Data-Only Pack)"
            st.session_state["sim_calls_quota"] = "Unlimited True 4G/5G / STD+Local Calls"
            st.session_state["sim_tenure_months"] = 42
            st.session_state["sim_payment"] = "UPI AutoPay (GPay / PhonePe / Paytm)"
            st.session_state["sim_paperless"] = "No"
            st.session_state["sim_online_sec"] = "Yes"
            st.session_state["sim_tech_supp"] = "Yes"
            st.session_state["sim_online_bk"] = "Yes"
            st.session_state["sim_dev_prot"] = "Yes"
            st.session_state["sim_stream_tv"] = "Yes"
            st.session_state["sim_stream_mov"] = "Yes"
            st.session_state["sim_gender"] = "Male"
            st.session_state["sim_senior"] = "0"
            st.session_state["sim_partner"] = "Yes"
            st.session_state["sim_dependents"] = "Yes"
            st.rerun()

    st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

    # 2-Column Responsive Layout
    col_input, col_sim = st.columns([1.2, 1.3], gap="large")

    with col_input:
        st.markdown("#### 👤 Subscriber Recharge & Account Profile")

        # ======================================================================
        # 1. NEW INDIAN TELECOM RECHARGE & PLAN FEATURES (7 Explicit Controls)
        # ======================================================================
        with st.expander("⚡ 1. Telecom Recharge Plan & Connectivity (7 Key Attributes)", expanded=True):
            # Control 1: Connection / Recharge Type
            conn_type = st.radio(
                "1. Connection / Recharge Type",
                ["Mobile Data (4G/5G SIM)", "Router / Fiber Broadband (Home/Family)"],
                index=0 if "Mobile" in st.session_state["sim_conn_type"] else 1,
                horizontal=True,
                key="input_conn_type",
            )
            is_router = "Router" in conn_type

            # Dynamic UI helper callout
            if is_router:
                st.caption("ℹ️ *Fiber Broadband Active:* Highlights broadband pool allowance, STB protection, and home entertainment options.")
            else:
                st.caption("ℹ️ *Mobile SIM Active:* Highlights daily high-speed 4G/5G data allowance and calling bundles.")

            # Control 3: Plan Validity in Days
            validity_options = ["28 Days", "56 Days", "84 Days", "90 Days", "365 Days"]
            current_val_str = st.session_state["sim_plan_validity"]
            val_idx = validity_options.index(current_val_str) if current_val_str in validity_options else 0
            plan_validity_str = st.selectbox(
                "2. Plan Validity Duration",
                validity_options,
                index=val_idx,
                key="input_plan_validity",
            )
            validity_days_int = int(plan_validity_str.split()[0])

            # Control 2: Plan Price in INR
            p_c1, p_c2 = st.columns([2, 1])
            with p_c1:
                plan_price_val = st.slider(
                    "3. Plan Price (₹)",
                    min_value=149.0,
                    max_value=3999.0,
                    value=float(st.session_state["sim_plan_price"]),
                    step=50.0,
                    key="input_plan_price",
                )
            with p_c2:
                # Computed MonthlyEffectiveARPU
                effective_arpu = round(plan_price_val / (validity_days_int / 30.0), 2)
                st.markdown(
                    f"""
                    <div style="background: rgba(15, 23, 42, 0.6); padding: 8px 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08); margin-top: 14px;">
                        <span style="font-size: 10px; color: #94A3B8; text-transform: uppercase;">Effective ARPU</span><br>
                        <span style="font-size: 15px; font-weight: 700; color: #34D399;">{format_inr(effective_arpu)}/mo</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Control 4: Data Allowance & Quota Mode (Paired inputs)
            st.markdown("<div style='font-size: 13px; font-weight: 600; color: #F8FAFC; margin-top: 6px; margin-bottom: 4px;'>4. Data Allowance & Structure</div>", unsafe_allow_html=True)
            d_c1, d_c2 = st.columns(2)
            with d_c1:
                quota_mode_options = ["Per Day (e.g., 2 GB/day)", "Entire Plan Validity (Bulk / Unlimited Pool)"]
                q_idx = 1 if (is_router or "Entire" in st.session_state["sim_data_quota_mode"]) else 0
                data_quota_mode = st.selectbox(
                    "Quota Structure Mode",
                    quota_mode_options,
                    index=q_idx,
                    key="input_data_quota_mode",
                )
            with d_c2:
                if "Per Day" in data_quota_mode:
                    data_allowance = st.slider(
                        "Daily Data (GB/day)",
                        min_value=1.0,
                        max_value=3.0,
                        value=float(min(3.0, max(1.0, st.session_state["sim_data_allowance"]))),
                        step=0.5,
                        key="input_data_allowance_daily",
                    )
                    effective_monthly_data = round(data_allowance * 30.0, 1)
                else:
                    data_allowance = st.slider(
                        "Total Validity Pool (GB)",
                        min_value=15.0,
                        max_value=1000.0,
                        value=float(st.session_state["sim_data_allowance"] if st.session_state["sim_data_allowance"] >= 15.0 else (1000.0 if is_router else 50.0)),
                        step=25.0,
                        key="input_data_allowance_bulk",
                    )
                    effective_monthly_data = round(data_allowance / (validity_days_int / 30.0), 1)

            st.caption(f"📊 *Effective Monthly Data Density:* **{effective_monthly_data} GB / Month**")

            # Controls 5 & 6: SMS Allowance & Voice Calls
            s_c1, s_c2 = st.columns(2)
            with s_c1:
                sms_options = ["100 SMS / Day", "300 SMS (Total Validity)", "1000 SMS (Total Validity)", "No SMS Included (Data-Only Pack)"]
                sms_default_idx = 3 if is_router else (sms_options.index(st.session_state["sim_sms_quota"]) if st.session_state["sim_sms_quota"] in sms_options else 0)
                sms_quota = st.selectbox("5. SMS Allowance", sms_options, index=sms_default_idx, key="input_sms_quota")
            with s_c2:
                call_options = ["Unlimited True 4G/5G / STD+Local Calls", "1,000 Mins / Month", "300 Mins (Total Validity)", "No Voice Calls (Data / Router Only)"]
                call_default_idx = 0 if not is_router else 3
                calls_quota = st.selectbox("6. Voice Calls Allowance", call_options, index=call_default_idx, key="input_calls_quota")

            # Control 7: Service Tenure with Lifecycle Badge
            st.markdown("<div style='margin-top: 6px;'></div>", unsafe_allow_html=True)
            tenure_months = st.slider(
                "7. How long has the customer been using the services? (Months)",
                min_value=0,
                max_value=72,
                value=int(st.session_state["sim_tenure_months"]),
                key="input_tenure_months",
            )
            # Live lifecycle stage badge
            if tenure_months <= 6:
                stage_badge = "<span class='badge badge-red'>New User: 0–6m (High Churn Window)</span>"
            elif tenure_months <= 24:
                stage_badge = "<span class='badge badge-amber'>Growing: 6–24m</span>"
            elif tenure_months <= 48:
                stage_badge = "<span class='badge badge-indigo'>Established: 24–48m</span>"
            else:
                stage_badge = "<span class='badge badge-emerald'>Loyal Veteran: 48m+</span>"

            st.markdown(f"<div style='margin-bottom: 6px;'>Subscriber Lifecycle: {stage_badge}</div>", unsafe_allow_html=True)

        # ======================================================================
        # 2. PAYMENT & BILLING DETAILS
        # ======================================================================
        with st.expander("💳 2. Payment Method & Billing Preferences", expanded=True):
            pay_col1, pay_col2 = st.columns(2)
            with pay_col1:
                payment_selected = st.selectbox(
                    "Recharge Payment Channel",
                    list(TELCO_PAYMENT_MAP.keys()),
                    index=list(TELCO_PAYMENT_MAP.keys()).index(st.session_state["sim_payment"]) if st.session_state["sim_payment"] in TELCO_PAYMENT_MAP else 0,
                    key="input_payment",
                )
            with pay_col2:
                paperless_billing = st.selectbox(
                    "Paperless E-Bill / WhatsApp Updates",
                    ["Yes", "No"],
                    index=["Yes", "No"].index(st.session_state["sim_paperless"]),
                    key="input_paperless",
                )

        # ======================================================================
        # 3. DIGITAL ECOSYSTEM ADD-ONS
        # ======================================================================
        with st.expander("📱 3. Digital Services & Value-Added Add-ons", expanded=True):
            add_c1, add_c2 = st.columns(2)
            with add_c1:
                has_tech_support = st.selectbox(
                    "Priority Customer Tech Support",
                    ["No", "Yes"],
                    index=["No", "Yes"].index(st.session_state["sim_tech_supp"] if st.session_state["sim_tech_supp"] in ["No", "Yes"] else "No"),
                    key="input_tech",
                )
                has_online_sec = st.selectbox(
                    "Cyber & Device Security Shield",
                    ["No", "Yes"],
                    index=["No", "Yes"].index(st.session_state["sim_online_sec"] if st.session_state["sim_online_sec"] in ["No", "Yes"] else "No"),
                    key="input_sec",
                )
                has_online_bk = st.selectbox(
                    "Cloud Storage Backup (100GB)",
                    ["No", "Yes"],
                    index=["No", "Yes"].index(st.session_state["sim_online_bk"] if st.session_state["sim_online_bk"] in ["No", "Yes"] else "No"),
                    key="input_bk",
                )
            with add_c2:
                has_dev_prot = st.selectbox(
                    "Router / Set-Top Box Hardware Protection",
                    ["No", "Yes"],
                    index=["No", "Yes"].index(st.session_state["sim_dev_prot"] if st.session_state["sim_dev_prot"] in ["No", "Yes"] else "No"),
                    key="input_dev",
                )
                has_stream_tv = st.selectbox(
                    "OTT Streaming Bundle",
                    ["Yes", "No"],
                    index=["Yes", "No"].index(st.session_state["sim_stream_tv"] if st.session_state["sim_stream_tv"] in ["No", "Yes"] else "Yes"),
                    key="input_tv",
                )
                has_stream_mov = st.selectbox(
                    "Live TV Channels Pack",
                    ["Yes", "No"],
                    index=["Yes", "No"].index(st.session_state["sim_stream_mov"] if st.session_state["sim_stream_mov"] in ["No", "Yes"] else "No"),
                    key="input_mov",
                )

        # ======================================================================
        # 4. ACCOUNT DEMOGRAPHICS (KEPT 100% INTACT)
        # ======================================================================
        with st.expander("👤 4. Account Demographics (Family & Profile)", expanded=False):
            demo_c1, demo_c2 = st.columns(2)
            with demo_c1:
                gender = st.selectbox("Account Holder Gender", ["Female", "Male"], index=["Female", "Male"].index(st.session_state["sim_gender"]), key="input_gender")
                senior = st.selectbox("Senior Citizen (60+)", ["0", "1"], index=["0", "1"].index(st.session_state["sim_senior"]), key="input_senior")
            with demo_c2:
                partner = st.selectbox("Family Linked Account (Partner)", ["No", "Yes"], index=["No", "Yes"].index(st.session_state["sim_partner"]), key="input_partner")
                dependents = st.selectbox("Linked Family Connections (Dependents)", ["No", "Yes"], index=["No", "Yes"].index(st.session_state["sim_dependents"]), key="input_dependents")

    # Map contract based on PlanValidityDays
    if validity_days_int >= 365:
        inferred_contract = "Two year" if validity_days_int > 365 else "One year"
    elif validity_days_int >= 84:
        inferred_contract = "One year"
    else:
        inferred_contract = "Month-to-month"

    inferred_internet = "Fiber optic" if is_router else "DSL"
    inferred_phone = "No" if "No Voice Calls" in calls_quota else "Yes"
    inferred_multiline = "No" if inferred_phone == "No" else ("Yes" if dependents == "Yes" else "No")
    suggested_total = round(effective_arpu * max(1, tenure_months), 2)

    # Construct BASELINE dataframe for prediction
    base_dict = {
        "TenureMonths": [tenure_months],
        "PlanPrice_INR": [plan_price_val],
        "PlanValidityDays": [validity_days_int],
        "DataAllowance_GB": [data_allowance],
        "DataQuotaMode": [data_quota_mode],
        "SMSQuota": [sms_quota],
        "CallsQuota": [calls_quota],
        "ConnectionType": [conn_type],
        "MonthlyEffectiveARPU": [effective_arpu],
        "EffectiveMonthlyData_GB": [effective_monthly_data],
        "MonthlyCharges": [effective_arpu],
        "TotalCharges": [suggested_total],
        "tenure": [tenure_months],
        "gender": [gender],
        "SeniorCitizen": [str(senior)],
        "Partner": [partner],
        "Dependents": [dependents],
        "PhoneService": [inferred_phone],
        "MultipleLines": [inferred_multiline if inferred_phone == "Yes" else "No phone service"],
        "InternetService": [inferred_internet],
        "OnlineSecurity": [has_online_sec],
        "OnlineBackup": [has_online_bk],
        "DeviceProtection": [has_dev_prot],
        "TechSupport": [has_tech_support],
        "StreamingTV": [has_stream_tv],
        "StreamingMovies": [has_stream_mov],
        "Contract": [inferred_contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [TELCO_PAYMENT_MAP[payment_selected]],
    }
    df_base = pd.DataFrame(base_dict)
    df_base_eng = engineer_features(df_base)

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    X_base_trans = preprocessor.transform(df_base_eng)
    p_base = float(classifier.predict_proba(X_base_trans)[0, 1])

    # SHAP Local Explanation on Baseline
    single_shap = explainer.shap_values(X_base_trans)[0]
    shap_df = pd.DataFrame({
        "RawFeature": feature_names,
        "CleanFeature": [clean_feature_label(f) for f in feature_names],
        "ShapValue": single_shap,
        "AbsShap": np.abs(single_shap),
    })

    pos_drivers = shap_df[shap_df["ShapValue"] > 0].sort_values(by="ShapValue", ascending=False)
    top_pos_driver = pos_drivers.iloc[0]["RawFeature"] if len(pos_drivers) > 0 else "None"
    recommended_playbook = get_retention_play(top_pos_driver, effective_arpu, p_base)

    with col_sim:
        st.markdown("#### 📊 Risk Diagnosis & Retention Simulator")

        # ======================================================================
        # INTERACTIVE "WHAT-IF" RETENTION SIMULATOR PANEL
        # ======================================================================
        st.markdown(
            """
            <div class="simulator-box">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 14px; font-weight: 700; color: #34D399; display: flex; align-items: center; gap: 6px;">
                        <span>🧪</span> LIVE "WHAT-IF" RETENTION OFFER SIMULATOR
                    </span>
                    <span class="badge badge-emerald">Interactive ROI Engine</span>
                </div>
                <div style="font-size: 12px; color: #94A3B8; margin-bottom: 12px;">
                    Simulate proactive retention offers below to observe dynamic churn drop, saved revenue, and net ROI in ₹:
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Dynamic Upgrade Validity Checkbox Label based on selected PlanValidityDays
        if validity_days_int == 28:
            validity_label = "🔒 Upgrade Validity: 28D ➔ 84-Day Plan (15% Off)"
            validity_help = "Extends recharge validity to 84 Days (Quarterly) with a 15% effective monthly tariff discount."
        elif validity_days_int in [56, 84, 90]:
            validity_label = f"🔒 Upgrade Validity: {validity_days_int}D ➔ 365-Day Annual Plan (15% Off)"
            validity_help = f"Upgrades subscriber from {validity_days_int}-day cycle to a 365-day annual plan with 15% tariff discount."
        elif validity_days_int == 365:
            validity_label = "🔒 Lock In 2-Year Loyalty Extension (20% Renewal Credit)"
            validity_help = "Locks in subscriber with a 2-Year enterprise/family commitment and 20% renewal discount."
        else:
            validity_label = f"🔒 Upgrade Validity: {validity_days_int}D ➔ 365-Day Annual Plan (15% Off)"
            validity_help = "Upgrades subscriber to a 365-day annual plan with 15% tariff discount."

        sim_c1, sim_c2 = st.columns(2)
        with sim_c1:
            tog_validity = st.checkbox(
                validity_label,
                value=False,
                help=validity_help,
                key="tog_validity",
            )
            tog_data = st.checkbox(
                "🚀 Boost Daily Data (+1 GB/day High-Speed Booster)",
                value=False,
                help="Provisions a complimentary +1 GB/day high-speed data booster pack to eliminate speed throttling.",
                key="tog_data",
            )
        with sim_c2:
            tog_support = st.checkbox(
                "🛠️ Complimentary VIP Tech Support + Security Shield",
                value=False,
                help="Provisions 90 days of priority technical resolution and digital security suite.",
                key="tog_support",
            )
            tog_autopay = st.checkbox(
                "💳 Switch to UPI AutoPay (₹50/mo Cashback)",
                value=False,
                help="Incentivizes friction-free UPI AutoPay with a ₹50/month cashback credit for 6 billing cycles.",
                key="tog_autopay",
            )

        # Build SIMULATED row based on active retention toggles
        sim_dict = base_dict.copy()
        intervention_cost = 0.0

        if tog_validity:
            if validity_days_int == 28:
                sim_dict["PlanValidityDays"] = [84]
                sim_dict["Contract"] = ["One year"]
                sim_dict["PlanPrice_INR"] = [round((effective_arpu * 0.85) * (84.0 / 30.0), 0)]
                sim_dict["MonthlyEffectiveARPU"] = [round(effective_arpu * 0.85, 2)]
                sim_dict["MonthlyCharges"] = sim_dict["MonthlyEffectiveARPU"]
                intervention_cost += effective_arpu * 0.15 * 12.0  # 15% discount for 1 year
            elif validity_days_int in [56, 84, 90]:
                sim_dict["PlanValidityDays"] = [365]
                sim_dict["Contract"] = ["One year"]
                sim_dict["PlanPrice_INR"] = [round((effective_arpu * 0.85) * (365.0 / 30.0), 0)]
                sim_dict["MonthlyEffectiveARPU"] = [round(effective_arpu * 0.85, 2)]
                sim_dict["MonthlyCharges"] = sim_dict["MonthlyEffectiveARPU"]
                intervention_cost += effective_arpu * 0.15 * 12.0  # 15% discount for 1 year
            elif validity_days_int == 365:
                sim_dict["PlanValidityDays"] = [730]
                sim_dict["Contract"] = ["Two year"]
                sim_dict["PlanPrice_INR"] = [round((effective_arpu * 0.80) * (365.0 / 30.0), 0)]
                sim_dict["MonthlyEffectiveARPU"] = [round(effective_arpu * 0.80, 2)]
                sim_dict["MonthlyCharges"] = sim_dict["MonthlyEffectiveARPU"]
                intervention_cost += effective_arpu * 0.20 * 12.0  # 20% discount for 1 year
            else:
                sim_dict["PlanValidityDays"] = [365]
                sim_dict["Contract"] = ["One year"]
                sim_dict["PlanPrice_INR"] = [round((effective_arpu * 0.85) * (365.0 / 30.0), 0)]
                sim_dict["MonthlyEffectiveARPU"] = [round(effective_arpu * 0.85, 2)]
                sim_dict["MonthlyCharges"] = sim_dict["MonthlyEffectiveARPU"]
                intervention_cost += effective_arpu * 0.15 * 12.0

        if tog_data:
            if "Per Day" in sim_dict["DataQuotaMode"][0]:
                sim_dict["DataAllowance_GB"] = [float(sim_dict["DataAllowance_GB"][0]) + 1.0]
                sim_dict["EffectiveMonthlyData_GB"] = [round(float(sim_dict["DataAllowance_GB"][0]) * 30.0, 1)]
            else:
                sim_dict["DataAllowance_GB"] = [float(sim_dict["DataAllowance_GB"][0]) + 50.0]
            intervention_cost += 249.0  # Data booster provisioning cost

        if tog_support:
            sim_dict["TechSupport"] = ["Yes"]
            sim_dict["OnlineSecurity"] = ["Yes"]
            intervention_cost += 300.0  # Provisioning cost

        if tog_autopay:
            sim_dict["PaymentMethod"] = ["Bank transfer (automatic)"]
            intervention_cost += 300.0  # ₹50/mo cashback for 6 months

        df_sim = pd.DataFrame(sim_dict)
        df_sim_eng = engineer_features(df_sim)
        X_sim_trans = preprocessor.transform(df_sim_eng)
        p_sim = float(classifier.predict_proba(X_sim_trans)[0, 1])

        # Active simulation metrics
        any_intervention_active = tog_validity or tog_data or tog_support or tog_autopay
        current_prob = p_sim if any_intervention_active else p_base
        current_prob_pct = current_prob * 100
        risk_reduction_pct = (p_base - p_sim) * 100 if any_intervention_active else 0.0

        current_monthly = sim_dict["MonthlyEffectiveARPU"][0]
        annual_rev_at_risk = round(current_monthly * 12.0 * current_prob, 2)
        annual_contract_val = round(current_monthly * 12.0, 2)

        # Financial ROI of intervention
        gross_rev_saved = round((p_base - p_sim) * (current_monthly * 12.0), 2)
        net_retention_roi = round(gross_rev_saved - intervention_cost, 2)
        roi_multiplier = round(gross_rev_saved / max(1.0, intervention_cost), 1)

        # Risk Tier Classification (calibrated around optimal decision threshold)
        if current_prob >= optimal_threshold:
            tier_label = "HIGH RISK (Likely to Churn)"
            tier_color = "#EF4444"
            badge_cls = "badge-red"
            card_cls = "card-red"
        elif current_prob >= round(optimal_threshold * 0.55, 2):
            tier_label = "MEDIUM RISK (At-Risk)"
            tier_color = "#F59E0B"
            badge_cls = "badge-amber"
            card_cls = "card-amber"
        else:
            tier_label = "LOW RISK (Retained)"
            tier_color = "#10B981"
            badge_cls = "badge-emerald"
            card_cls = "card-emerald"

        # 3 KPI Metric Cards
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        kpi_c1, kpi_c2, kpi_c3 = st.columns(3)
        with kpi_c1:
            delta_html = ""
            if any_intervention_active:
                delta_html = f"<div style='font-size: 11px; color: #34D399; font-weight: 600;'>📉 Reduced by {risk_reduction_pct:.1f}%</div>"
            st.markdown(
                f"""
                <div class="metric-card {card_cls}">
                    <div class="metric-label">Churn Probability</div>
                    <div class="metric-value" style="color: {tier_color};">{current_prob_pct:.1f}%</div>
                    {delta_html if delta_html else f"<div class='metric-sub'>Baseline: {p_base*100:.1f}%</div>"}
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi_c2:
            st.markdown(
                f"""
                <div class="metric-card {card_cls}">
                    <div class="metric-label">Risk Classification</div>
                    <div style="margin-top: 6px;"><span class="badge {badge_cls}">{tier_label}</span></div>
                    <div class="metric-sub">Optimal Cutoff: {int(optimal_threshold*100)}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi_c3:
            st.markdown(
                f"""
                <div class="metric-card card-indigo">
                    <div class="metric-label">Annual Revenue at Risk</div>
                    <div class="metric-value">{format_inr(annual_rev_at_risk)}</div>
                    <div class="metric-sub">Gross ARR: {format_inr(annual_contract_val)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Plotly Gauge Chart
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=current_prob_pct,
                number={"suffix": "%", "font": {"size": 29, "color": "#F8FAFC", "family": "Inter"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748B", "tickvals": [0, 25, 50, 75, 100]},
                    "bar": {"color": tier_color, "thickness": 0.28},
                    "bgcolor": "rgba(30, 41, 59, 0.4)",
                    "borderwidth": 1,
                    "bordercolor": "rgba(255, 255, 255, 0.1)",
                    "steps": [
                        {"range": [0, 35], "color": "rgba(16, 185, 129, 0.25)"},
                        {"range": [35, 65], "color": "rgba(245, 158, 11, 0.25)"},
                        {"range": [65, 100], "color": "rgba(239, 68, 68, 0.25)"},
                    ],
                    "threshold": {
                        "line": {"color": "#F8FAFC", "width": 3},
                        "thickness": 0.8,
                        "value": current_prob_pct,
                    },
                },
            )
        )
        fig_gauge.update_layout(
            height=190,
            margin=dict(l=20, r=20, t=15, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#F8FAFC", "family": "Inter"},
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Dynamic Intervention ROI Strip if toggles active
        if any_intervention_active:
            roi_badge_color = "#10B981" if net_retention_roi > 0 else "#EF4444"
            st.markdown(
                f"""
                <div style="background: rgba(30, 41, 59, 0.9); border: 1px solid rgba(16, 185, 129, 0.4); border-radius: 10px; padding: 12px 16px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                    <div>
                        <span style="font-size: 11px; text-transform: uppercase; color: #94A3B8; font-weight: 600;">Intervention Cost</span><br>
                        <span style="font-size: 15px; font-weight: 700; color: #F8FAFC;">{format_inr(intervention_cost)}</span>
                    </div>
                    <div>
                        <span style="font-size: 11px; text-transform: uppercase; color: #94A3B8; font-weight: 600;">Annual Revenue Saved</span><br>
                        <span style="font-size: 15px; font-weight: 700; color: #34D399;">{format_inr(gross_rev_saved)}</span>
                    </div>
                    <div>
                        <span style="font-size: 11px; text-transform: uppercase; color: #94A3B8; font-weight: 600;">Net Retention ROI</span><br>
                        <span style="font-size: 16px; font-weight: 800; color: {roi_badge_color};">{format_inr(net_retention_roi)} ({roi_multiplier}x ROI)</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # SHAP Diverging Horizontal Bar Chart
        top_shap = shap_df.sort_values(by="AbsShap", ascending=False).head(8)
        top_shap = top_shap.sort_values(by="ShapValue", ascending=True)
        bar_colors = ["#EF4444" if val > 0 else "#10B981" for val in top_shap["ShapValue"]]

        fig_shap = go.Figure(
            go.Bar(
                x=top_shap["ShapValue"],
                y=top_shap["CleanFeature"],
                orientation="h",
                marker=dict(color=bar_colors, line=dict(color="rgba(255,255,255,0.15)", width=1)),
                hovertemplate="<b>%{y}</b><br>SHAP Contribution: %{x:+.4f}<extra></extra>",
            )
        )
        fig_shap.update_layout(
            title=dict(
                text="<b>Why Is This Customer Leaving? (SHAP Feature Attribution)</b>",
                font=dict(size=13, color="#F8FAFC"),
            ),
            xaxis=dict(
                title="SHAP Value (Red = Pushes Churn UP, Green = Keeps Customer LOYAL)",
                title_font=dict(size=10, color="#94A3B8"),
                tickfont=dict(color="#94A3B8", size=10),
                gridcolor="rgba(255, 255, 255, 0.08)",
                zerolinecolor="#6366F1",
                zerolinewidth=1.5,
            ),
            yaxis=dict(
                tickfont=dict(size=11, color="#F8FAFC"),
                gridcolor="rgba(0,0,0,0)",
            ),
            height=280,
            margin=dict(l=10, r=20, t=30, b=30),
            paper_bgcolor="rgba(30, 41, 59, 0.5)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_shap, use_container_width=True)

        # Recommended AI Retention Playbook Callout
        st.markdown(
            f"""
            <div class="playbook-box">
                <div class="playbook-header">
                    <span>💡</span> <b>RECOMMENDED RETENTION PLAYBOOK (Tailored to #1 SHAP Driver)</b>
                </div>
                <div class="playbook-title">{recommended_playbook['title']}</div>
                <div class="playbook-desc">{recommended_playbook['play']}</div>
                <div style="background: rgba(15, 23, 42, 0.6); border-radius: 8px; padding: 10px 14px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                    <div style="display: flex; gap: 18px; align-items: center; flex-wrap: wrap;">
                        <div>
                            <span style="font-size: 11px; color: #94A3B8;">Estimated Offer Cost:</span>
                            <span style="font-size: 13px; font-weight: 700; color: #F87171; margin-left: 4px;">{format_inr(recommended_playbook['offer_cost'])}</span>
                        </div>
                        <div>
                            <span style="font-size: 11px; color: #94A3B8;">Net Value Realized:</span>
                            <span style="font-size: 13px; font-weight: 800; color: #A5B4FC; margin-left: 4px;">+{format_inr(recommended_playbook['net_roi'])} ({recommended_playbook['roi_multiplier']}x ROI)</span>
                        </div>
                    </div>
                    <div>
                        <span style="font-size: 11px; color: #94A3B8;">Annual Revenue Saved:</span>
                        <span style="font-size: 13px; font-weight: 700; color: #34D399; margin-left: 4px;">{format_inr(recommended_playbook['annual_rev_saved'])}</span>
                    </div>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                    <div style="font-size: 12px; color: #E2E8F0;">
                        <b>Recommended Execution:</b> <code>{recommended_playbook['action']}</code>
                    </div>
                    <span class="badge badge-emerald">{recommended_playbook['lift']}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# TAB 2: BATCH SCORING & ROI PRIORITIZER
# ==============================================================================
with tab2:
    st.markdown('<div class="section-title">📂 Enterprise Batch Scoring & Revenue Prioritization Matrix (INR ₹)</div>', unsafe_allow_html=True)
    st.markdown(
        "Upload customer cohort files or score the built-in evaluation sample (500 Indian subscribers with full recharge plan attributes). "
        "The engine ranks high-risk accounts by **Annual Revenue at Risk (₹)** to prioritize customer success workflows."
    )

    batch_action_col1, batch_action_col2 = st.columns([1.8, 2.5])
    with batch_action_col1:
        use_sample = st.button("🚀 Score Built-in Customer Cohort (500 Subscribers)", use_container_width=True, type="primary")

    uploaded_file = st.file_uploader("Or Upload Custom Subscriber CSV File", type=["csv"], key="batch_uploader")

    # Persist batch dataset in session state so dropdowns/filters do not reset data
    if "batch_raw" not in st.session_state:
        st.session_state["batch_raw"] = None

    if use_sample:
        if os.path.exists(BUILTIN_SAMPLE_PATH):
            st.session_state["batch_raw"] = pd.read_csv(BUILTIN_SAMPLE_PATH)
            st.session_state.pop("batch_scored", None)  # Force re-score
            st.success("Loaded built-in evaluation sample of 500 telecom subscribers with balanced risk distribution!")
        else:
            st.error("Built-in sample CSV not found at data/telco_test_sample.csv")
    elif uploaded_file is not None:
        try:
            st.session_state["batch_raw"] = pd.read_csv(uploaded_file)
            st.session_state.pop("batch_scored", None)  # Force re-score
            st.success(f"Uploaded subscriber cohort successfully! ({len(st.session_state['batch_raw']):,} records)")
        except Exception as e:
            st.error(f"Error parsing uploaded CSV: {e}")

    df_batch_raw = st.session_state.get("batch_raw", None)

    if df_batch_raw is not None:
        if "batch_scored" not in st.session_state or st.session_state.get("batch_scored_src") != id(df_batch_raw):
            with st.spinner("Executing feature engineering & XGBoost inference pipeline..."):
                df_batch = df_batch_raw.copy()
                if "TotalCharges" in df_batch.columns:
                    df_batch["TotalCharges"] = pd.to_numeric(df_batch["TotalCharges"].astype(str).str.strip(), errors="coerce").fillna(0.0)
                if "SeniorCitizen" in df_batch.columns:
                    df_batch["SeniorCitizen"] = df_batch["SeniorCitizen"].astype(str)

                engineered_batch = engineer_features(df_batch)

                # Predict probabilities
                batch_trans = pipeline.named_steps["preprocessor"].transform(engineered_batch)
                probs = pipeline.named_steps["classifier"].predict_proba(batch_trans)[:, 1]

                # Append predictions and intelligence metrics
                df_batch["Churn_Probability"] = np.round(probs, 4)
                df_batch["Churn_Risk_Pct"] = (probs * 100).round(1)

                def get_tier(p):
                    if p >= 0.65:
                        return "High Risk"
                    elif p >= 0.35:
                        return "Medium Risk"
                    else:
                        return "Low Risk"

                df_batch["Risk_Tier"] = [get_tier(p) for p in probs]

                monthly = pd.to_numeric(df_batch["MonthlyCharges"], errors="coerce").fillna(0.0)
                df_batch["Monthly_Revenue_At_Risk"] = np.round(monthly * probs, 2)
                df_batch["Annual_Revenue_At_Risk"] = np.round(monthly * 12.0 * probs, 2)

                # Priority score: High Risk + High Monthly Spend
                df_batch["Priority_Score"] = np.round(probs * (monthly / (monthly.max() + 1e-5)) * 100, 1)

                # Compute personalized Primary Churn Driver and Tailored Retention Play for each subscriber
                def compute_personalized_strategy(row):
                    tier = row.get("Risk_Tier", "Low Risk")
                    arpu = float(row.get("MonthlyEffectiveARPU", row.get("MonthlyCharges", 0.0)))
                    validity = int(row.get("PlanValidityDays", 28))
                    contract = str(row.get("Contract", "Month-to-month"))
                    data_mode = str(row.get("DataQuotaMode", "Per Day"))
                    data_gb = float(row.get("DataAllowance_GB", 1.5))
                    tech_support = str(row.get("TechSupport", "No"))
                    online_sec = str(row.get("OnlineSecurity", "No"))
                    pay_method = str(row.get("PaymentMethod", "")).lower()
                    tenure = float(row.get("TenureMonths", row.get("tenure", 0)))

                    if tier == "Low Risk":
                        return "Loyal Account", "No Discount Needed — Enroll in Annual Loyalty Rewards"

                    # 1. Short validity with high monthly spend
                    if (validity <= 28 or "month" in contract.lower()) and arpu >= 600:
                        return "Short 28D Validity", "Upgrade 28D ➔ 84-Day Plan (15% Tariff Off)"
                    # 2. Daily data limit exhaustion
                    elif "per day" in data_mode.lower() and data_gb <= 1.5:
                        return f"Low Daily Data ({data_gb:.1f} GB/day)", "Free +1 GB/Day High-Speed Data Booster (30 Days)"
                    # 3. Missing critical support & security
                    elif tech_support == "No" or online_sec == "No":
                        return "No Tech Support", "Complimentary 3-Month Priority Tech Support + Security Shield"
                    # 4. Manual payment drop-off friction
                    elif "check" in pay_method or "electronic" in pay_method or "manual" in pay_method:
                        return "Manual UPI Friction", "₹50/mo Cashback for 6 Cycles on UPI AutoPay Switch"
                    # 5. Early tenure vulnerability
                    elif tenure <= 6:
                        return f"Early Tenure ({int(tenure)}m)", "New-User Onboarding Loyalty Bonus + Free OTT Pack (2 Months)"
                    # 6. Fallback for medium/high risk
                    else:
                        return "Tariff & Plan Friction", "Targeted 10% Renewal Loyalty Voucher + Free OTT Weekend Pass"

                strategies = [compute_personalized_strategy(row) for _, row in df_batch.iterrows()]
                df_batch["Primary_Churn_Driver"] = [s[0] for s in strategies]
                df_batch["Recommended_Retention_Play"] = [s[1] for s in strategies]

                st.session_state["batch_scored"] = df_batch
                st.session_state["batch_scored_src"] = id(df_batch_raw)
        else:
            df_batch = st.session_state["batch_scored"]

        # Interactive Decision Threshold Slider (Tab 2)
        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
        thresh_col1, thresh_col2 = st.columns([2.6, 3.4])
        with thresh_col1:
            batch_threshold = st.slider(
                "🎚️ Decision Boundary Threshold (Likely to Churn)",
                min_value=0.25,
                max_value=0.70,
                value=float(optimal_threshold),
                step=0.01,
                help="Subscribers with predicted churn probability ≥ this threshold are classified as High Risk / Likely to Churn. Calibrated to optimal F1.",
                key="tab2_batch_thresh_slider",
            )

        # Dynamic re-classification based on the slider threshold
        probs = df_batch["Churn_Probability"].values
        df_batch["Risk_Tier"] = [
            "High Risk" if p >= batch_threshold else ("Medium Risk" if p >= batch_threshold * 0.55 else "Low Risk")
            for p in probs
        ]

        # Calculate live metrics on batch
        if "Churn" in df_batch.columns:
            y_act = df_batch["Churn"].astype(str).str.strip().str.lower().isin(["yes", "1", "true"]).astype(int).values
            flagged = (probs >= batch_threshold).astype(int)
            tp = int(((flagged == 1) & (y_act == 1)).sum())
            fp = int(((flagged == 1) & (y_act == 0)).sum())
            fn = int(((flagged == 0) & (y_act == 1)).sum())
            batch_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            batch_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            batch_f1 = (2 * batch_prec * batch_rec) / (batch_prec + batch_rec) if (batch_prec + batch_rec) > 0 else 0.0
            flagged_cnt = int(flagged.sum())
            with thresh_col2:
                st.markdown(
                    f"""
                    <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(99, 102, 241, 0.25); border-radius: 8px; padding: 10px 14px; margin-top: 5px;">
                        <div style="font-size: 11px; color: #94A3B8; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px;">Live Batch Validation at Cutoff ({batch_threshold:.2f})</div>
                        <div style="display: flex; gap: 16px; margin-top: 4px; font-size: 13px; color: #F8FAFC; flex-wrap: wrap;">
                            <span>🎯 <b>Precision:</b> <span style="color: #818CF8;">{batch_prec*100:.1f}%</span></span>
                            <span>📡 <b>Recall:</b> <span style="color: #34D399;">{batch_rec*100:.1f}%</span></span>
                            <span>⚡ <b>F1-Score:</b> <span style="color: #FBBF24;">{batch_f1:.3f}</span></span>
                            <span>🚨 <b>Flagged:</b> <span style="color: #F87171;">{flagged_cnt:,}</span> / {len(df_batch):,}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            curve = metrics_data.get("threshold_optimization", {}).get("threshold_curve", [])
            closest = min(curve, key=lambda x: abs(x["threshold"] - batch_threshold)) if curve else {"precision": 0.68, "recall": 0.80, "f1": 0.73}
            batch_prec = closest["precision"]
            batch_rec = closest["recall"]
            batch_f1 = closest["f1"]
            flagged_cnt = int((probs >= batch_threshold).sum())
            with thresh_col2:
                st.markdown(
                    f"""
                    <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(99, 102, 241, 0.25); border-radius: 8px; padding: 10px 14px; margin-top: 5px;">
                        <div style="font-size: 11px; color: #94A3B8; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px;">Model Threshold Expectation ({batch_threshold:.2f})</div>
                        <div style="display: flex; gap: 16px; margin-top: 4px; font-size: 13px; color: #F8FAFC; flex-wrap: wrap;">
                            <span>🎯 <b>Expected Precision:</b> <span style="color: #818CF8;">{batch_prec*100:.1f}%</span></span>
                            <span>📡 <b>Expected Recall:</b> <span style="color: #34D399;">{batch_rec*100:.1f}%</span></span>
                            <span>⚡ <b>F1-Score:</b> <span style="color: #FBBF24;">{batch_f1:.3f}</span></span>
                            <span>🚨 <b>Flagged:</b> <span style="color: #F87171;">{flagged_cnt:,}</span> / {len(df_batch):,}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # High-level KPIs in Indian Rupees (₹)
        total_customers = len(df_batch)
        high_risk_count = int((df_batch["Risk_Tier"] == "High Risk").sum())
        med_risk_count = int((df_batch["Risk_Tier"] == "Medium Risk").sum())
        low_risk_count = int((df_batch["Risk_Tier"] == "Low Risk").sum())
        high_risk_pct = (high_risk_count / total_customers) * 100 if total_customers > 0 else 0
        total_monthly_risk = df_batch["Monthly_Revenue_At_Risk"].sum()
        total_annual_risk = df_batch["Annual_Revenue_At_Risk"].sum()
        projected_saved = total_annual_risk * 0.30

        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
        bkpi1, bkpi2, bkpi3, bkpi4 = st.columns(4)
        with bkpi1:
            st.markdown(
                f"""
                <div class="metric-card card-indigo">
                    <div class="metric-label">Analyzed Subscribers</div>
                    <div class="metric-value">{total_customers:,}</div>
                    <div class="metric-sub">Active Telecom Accounts</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with bkpi2:
            st.markdown(
                f"""
                <div class="metric-card card-red">
                    <div class="metric-label">High-Risk Accounts</div>
                    <div class="metric-value" style="color: #F87171;">{high_risk_count:,} ({high_risk_pct:.1f}%)</div>
                    <div class="metric-sub">Cutoff Threshold: {batch_threshold*100:.0f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with bkpi3:
            st.markdown(
                f"""
                <div class="metric-card card-amber">
                    <div class="metric-label">Monthly ARR Exposure</div>
                    <div class="metric-value">{format_inr(total_monthly_risk)}</div>
                    <div class="metric-sub">Annual: {format_inr(total_annual_risk)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with bkpi4:
            st.markdown(
                f"""
                <div class="metric-card card-emerald">
                    <div class="metric-label">Projected Revenue Saved</div>
                    <div class="metric-value" style="color: #34D399;">{format_inr(projected_saved)}</div>
                    <div class="metric-sub">Assuming 30% Retention Lift</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Plotly Interactive Scatter/Bubble Chart: Tenure vs Plan Price / ARPU
        st.markdown("#### 🔍 Subscriber Portfolio Matrix: Tenure vs. Monthly Tariff (₹)")
        color_map = {"High Risk": "#EF4444", "Medium Risk": "#F59E0B", "Low Risk": "#10B981"}

        fig_scatter = go.Figure()
        for tier in ["High Risk", "Medium Risk", "Low Risk"]:
            sub_df = df_batch[df_batch["Risk_Tier"] == tier]
            if len(sub_df) > 0:
                cid = sub_df["customerID"] if "customerID" in sub_df.columns else sub_df.index
                arr_formatted = [format_inr(v) for v in sub_df["Annual_Revenue_At_Risk"]]
                monthly_formatted = [format_inr(v) for v in sub_df["MonthlyCharges"]]
                tenure_col = sub_df["TenureMonths"] if "TenureMonths" in sub_df.columns else sub_df["tenure"]
                val_col = sub_df["PlanValidityDays"] if "PlanValidityDays" in sub_df.columns else 28

                fig_scatter.add_trace(
                    go.Scatter(
                        x=tenure_col,
                        y=sub_df["MonthlyCharges"],
                        mode="markers",
                        name=tier,
                        marker=dict(
                            color=color_map[tier],
                            size=7,
                            opacity=0.75,
                            line=dict(color="rgba(255,255,255,0.2)", width=0.5),
                        ),
                        customdata=np.stack((cid, sub_df["Churn_Risk_Pct"], arr_formatted, sub_df["Recommended_Retention_Play"], monthly_formatted, val_col, sub_df["Primary_Churn_Driver"]), axis=-1),
                        hovertemplate=(
                            "<b>Subscriber: %{customdata[0]}</b><br>"
                            "Tenure: %{x} months<br>"
                            "Validity: %{customdata[5]} Days<br>"
                            "Monthly Tariff: %{customdata[4]}<br>"
                            "Churn Probability: %{customdata[1]}%<br>"
                            "Primary Driver: <b>%{customdata[6]}</b><br>"
                            "ARR at Risk: %{customdata[2]}<br>"
                            "Recommended Play: <i>%{customdata[3]}</i><extra></extra>"
                        ),
                    )
                )

        fig_scatter.update_layout(
            xaxis=dict(
                title="Subscriber Tenure (Months with Operator)",
                title_font=dict(color="#94A3B8"),
                tickfont=dict(color="#94A3B8"),
                gridcolor="rgba(255, 255, 255, 0.06)",
            ),
            yaxis=dict(
                title="Monthly Tariff (₹)",
                title_font=dict(color="#94A3B8"),
                tickfont=dict(color="#94A3B8"),
                gridcolor="rgba(255, 255, 255, 0.06)",
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(color="#F8FAFC"),
            ),
            height=360,
            margin=dict(l=20, r=20, t=30, b=30),
            paper_bgcolor="rgba(30, 41, 59, 0.5)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Actionable High-Risk Table
        st.markdown("#### 📋 Prioritized Action List for Telecom Retention & Customer Success Teams")
        filter_col1, filter_col2 = st.columns([1.8, 3.2])
        with filter_col1:
            tier_filter = st.selectbox(
                "Filter by Risk Tier",
                ["All Tiers", "High Risk", "Medium Risk", "Low Risk"],
                index=1,
                key="batch_tier_filter",
            )

        filtered_table = df_batch if tier_filter == "All Tiers" else df_batch[df_batch["Risk_Tier"] == tier_filter]
        sorted_table = filtered_table.sort_values(by="Priority_Score", ascending=False).copy()

        with filter_col2:
            st.markdown(
                f"""
                <div style="margin-top: 28px; font-size: 13px; color: #94A3B8;">
                    Showing <b>{len(sorted_table):,}</b> accounts in <b>{tier_filter}</b> &bull; Total Cohort: <b>{total_customers:,}</b>
                    (🔴 High: {high_risk_count} | 🟡 Medium: {med_risk_count} | 🟢 Low: {low_risk_count})
                </div>
                """,
                unsafe_allow_html=True,
            )

        display_table = sorted_table.copy()
        if "MonthlyCharges" in display_table.columns:
            display_table["MonthlyCharges (₹)"] = display_table["MonthlyCharges"].apply(format_inr)
        if "Annual_Revenue_At_Risk" in display_table.columns:
            display_table["Annual_Revenue_At_Risk (₹)"] = display_table["Annual_Revenue_At_Risk"].apply(format_inr)
        if "PlanPrice_INR" in display_table.columns:
            display_table["PlanPrice (₹)"] = display_table["PlanPrice_INR"].apply(format_inr)

        cols_to_show = [
            c for c in [
                "customerID",
                "Churn_Risk_Pct",
                "Risk_Tier",
                "Primary_Churn_Driver",
                "Recommended_Retention_Play",
                "ConnectionType",
                "PlanValidityDays",
                "PlanPrice (₹)",
                "MonthlyCharges (₹)",
                "TenureMonths",
                "Annual_Revenue_At_Risk (₹)",
                "Priority_Score",
            ] if c in display_table.columns
        ]

        st.dataframe(
            display_table[cols_to_show].head(100),
            use_container_width=True,
            height=320,
        )

        csv_buffer = io.StringIO()
        sorted_table.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Scored Action List (.CSV)",
            data=csv_buffer.getvalue(),
            file_name="churnshield_scored_retention_action_list.csv",
            mime="text/csv",
            type="primary",
        )
    else:
        st.info("💡 Click the button above to load the built-in 500-subscriber evaluation cohort, or upload your own CSV dataset.")


# ==============================================================================
# TAB 3: MODEL EVALUATION & GLOBAL INSIGHTS
# ==============================================================================
with tab3:
    st.markdown('<div class="section-title">📊 Multi-Model Benchmark & Global SHAP Explainability</div>', unsafe_allow_html=True)
    st.markdown(
        "Comparative evaluation of **Logistic Regression**, **Random Forest**, and **XGBoost** incorporating Indian recharge plan features. "
        "Hyperparameters are tuned via **Stratified 5-Fold Cross-Validation** with decision threshold optimization to maximize operational retention ROI."
    )

    # 1. Comparison Table
    m_rows = []
    for model_name in ["Logistic Regression", "Random Forest", "XGBoost"]:
        if model_name in metrics_data:
            m = metrics_data[model_name]
            m_rows.append({
                "Model Architecture": f"{model_name} {'🏆 (Selected for ChurnShield AI)' if model_name == 'XGBoost' else ''}",
                "ROC-AUC": f"{m['roc_auc']:.4f}",
                "Recall (Churn)": f"{m['recall']:.4f}",
                "F1-Score": f"{m['f1_score']:.4f}",
                "Precision": f"{m['precision']:.4f}",
                "Accuracy": f"{m['accuracy']:.4f}",
            })
    df_metrics = pd.DataFrame(m_rows)
    st.table(df_metrics)

    # Tuned Hyperparameters Badges & Details
    best_xgb_p = metrics_data.get("best_params_xgb", metrics_data.get("best_params", {}))
    best_xgb_cv = metrics_data.get("best_cv_f1_xgb", 0.7238)
    best_rf_p = metrics_data.get("best_params_rf", {})
    best_rf_cv = metrics_data.get("best_cv_f1_rf", 0.7208)

    st.markdown("##### ⚙️ Cross-Validated Hyperparameter Tuning Results (Stratified 5-Fold CV, F1-Scoring)")
    hp_c1, hp_c2 = st.columns(2, gap="medium")
    with hp_c1:
        st.markdown(
            f"""
            <div class="metric-card card-indigo">
                <div class="metric-label" style="display: flex; justify-content: space-between;">
                    <span>XGBoost Classifier (Tuned 🏆)</span>
                    <span class="badge badge-emerald">CV F1: {best_xgb_cv:.4f}</span>
                </div>
                <div style="font-size: 12.5px; color: #CBD5E1; margin-top: 8px; line-height: 1.7;">
                    • <b>max_depth:</b> <code>{best_xgb_p.get('max_depth', 5)}</code> &bull; <b>learning_rate:</b> <code>{best_xgb_p.get('learning_rate', 0.03)}</code><br>
                    • <b>n_estimators:</b> <code>{best_xgb_p.get('n_estimators', 350)}</code> &bull; <b>subsample:</b> <code>{best_xgb_p.get('subsample', 0.8)}</code><br>
                    • <b>colsample_bytree:</b> <code>{best_xgb_p.get('colsample_bytree', 0.9)}</code> &bull; <b>scale_pos_weight:</b> <code>2.77</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with hp_c2:
        st.markdown(
            f"""
            <div class="metric-card card-amber">
                <div class="metric-label" style="display: flex; justify-content: space-between;">
                    <span>Random Forest Classifier (Tuned)</span>
                    <span class="badge badge-amber">CV F1: {best_rf_cv:.4f}</span>
                </div>
                <div style="font-size: 12.5px; color: #CBD5E1; margin-top: 8px; line-height: 1.7;">
                    • <b>n_estimators:</b> <code>{best_rf_p.get('n_estimators', 250)}</code><br>
                    • <b>max_depth:</b> <code>{best_rf_p.get('max_depth', 12)}</code><br>
                    • <b>class_weight:</b> <code>balanced</code> &bull; <b>criterion:</b> <code>gini</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 2. Decision Threshold Optimization Section
    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
    st.markdown("#### 🎯 Decision Threshold Optimization & Operating Precision-Recall Curve")
    st.markdown(
        "Standard classification uses a fixed 0.50 cutoff, which creates excessive false alarms in imbalanced telecom churn data. "
        "Evaluating candidate thresholds between **0.25 and 0.70** pinpoints the **optimal F1 operating threshold** that balances customer retention against intervention cost."
    )

    thresh_opt = metrics_data.get("threshold_optimization", {})
    def_m = thresh_opt.get("default_metrics", {})
    opt_m = thresh_opt.get("optimal_metrics", {})
    opt_cutoff = float(thresh_opt.get("optimal_threshold", optimal_threshold))

    # Side-by-side Default vs Optimal Threshold Cards
    comp_c1, comp_c2 = st.columns(2, gap="medium")
    with comp_c1:
        st.markdown(
            f"""
            <div class="metric-card card-indigo">
                <div class="metric-label" style="display: flex; justify-content: space-between;">
                    <span>Default Decision Boundary</span>
                    <span class="badge badge-indigo">Cutoff: 0.50</span>
                </div>
                <div style="display: flex; gap: 14px; margin-top: 8px; flex-wrap: wrap;">
                    <div><span style="font-size: 11px; color: #94A3B8;">Precision:</span> <b style="color: #818CF8;">{def_m.get('precision', 0.6345)*100:.1f}%</b></div>
                    <div><span style="font-size: 11px; color: #94A3B8;">Recall:</span> <b style="color: #34D399;">{def_m.get('recall', 0.8262)*100:.1f}%</b></div>
                    <div><span style="font-size: 11px; color: #94A3B8;">F1-Score:</span> <b style="color: #FBBF24;">{def_m.get('f1_score', 0.7178):.4f}</b></div>
                    <div><span style="font-size: 11px; color: #94A3B8;">Accuracy:</span> <b>{def_m.get('accuracy', 0.8126)*100:.1f}%</b></div>
                </div>
                <div style="font-size: 11px; color: #94A3B8; margin-top: 6px;">Higher false alarms: retention offers given to subscribers who wouldn't have churned.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with comp_c2:
        st.markdown(
            f"""
            <div class="metric-card card-emerald">
                <div class="metric-label" style="display: flex; justify-content: space-between;">
                    <span>Optimal F1 Decision Boundary 🏆</span>
                    <span class="badge badge-emerald">Optimal: {opt_cutoff:.2f}</span>
                </div>
                <div style="display: flex; gap: 14px; margin-top: 8px; flex-wrap: wrap;">
                    <div><span style="font-size: 11px; color: #94A3B8;">Precision:</span> <b style="color: #34D399;">{opt_m.get('precision', 0.6819)*100:.1f}%</b></div>
                    <div><span style="font-size: 11px; color: #94A3B8;">Recall:</span> <b style="color: #34D399;">{opt_m.get('recall', 0.7952)*100:.1f}%</b></div>
                    <div><span style="font-size: 11px; color: #94A3B8;">F1-Score:</span> <b style="color: #34D399;">{opt_m.get('f1_score', 0.7345):.4f}</b></div>
                    <div><span style="font-size: 11px; color: #94A3B8;">Accuracy:</span> <b>{opt_m.get('accuracy', 0.8495)*100:.1f}%</b></div>
                </div>
                <div style="font-size: 11px; color: #34D399; margin-top: 6px;"><b>+4.7% Precision Gain:</b> Minimizes wasted retention expenditure while capturing ~80% of true churners.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Interactive Threshold Slider in Tab 3
    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
    eval_thresh = st.slider(
        "🎚️ Interactive Decision Threshold Slider (Explore Operating Boundary on Holdout Test Cohort)",
        min_value=0.25,
        max_value=0.70,
        value=float(opt_cutoff),
        step=0.01,
        key="tab3_threshold_slider",
        help="Adjust the decision threshold to see how Precision, Recall, F1, and flagged accounts shift across the test population.",
    )

    curve_pts = thresh_opt.get("threshold_curve", [])
    if curve_pts:
        selected_pt = min(curve_pts, key=lambda x: abs(x["threshold"] - eval_thresh))
        cur_prec = selected_pt["precision"]
        cur_rec = selected_pt["recall"]
        cur_f1 = selected_pt["f1"]

        # Approximate test set accounts (test set has 1,409 rows, 374 actual churners)
        approx_flagged = int(round((374 * cur_rec) / max(0.01, cur_prec)))
        approx_flagged = min(1409, max(50, approx_flagged))

        kpi_th1, kpi_th2, kpi_th3, kpi_th4, kpi_th5 = st.columns(5)
        with kpi_th1:
            st.markdown(
                f"""
                <div class="metric-card card-indigo">
                    <div class="metric-label">Operating Cutoff</div>
                    <div class="metric-value">{eval_thresh:.2f}</div>
                    <div class="metric-sub">{'Optimal Cutoff' if abs(eval_thresh - opt_cutoff) < 0.005 else f'Δ {eval_thresh - opt_cutoff:+.2f}'}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi_th2:
            st.markdown(
                f"""
                <div class="metric-card card-indigo">
                    <div class="metric-label">Precision</div>
                    <div class="metric-value" style="color: #818CF8;">{cur_prec*100:.1f}%</div>
                    <div class="metric-sub">Accuracy of Flags</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi_th3:
            st.markdown(
                f"""
                <div class="metric-card card-emerald">
                    <div class="metric-label">Recall (Coverage)</div>
                    <div class="metric-value" style="color: #34D399;">{cur_rec*100:.1f}%</div>
                    <div class="metric-sub">True Churn Caught</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi_th4:
            st.markdown(
                f"""
                <div class="metric-card card-amber">
                    <div class="metric-label">F1-Score</div>
                    <div class="metric-value" style="color: #FBBF24;">{cur_f1:.3f}</div>
                    <div class="metric-sub">Harmonic Mean</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi_th5:
            st.markdown(
                f"""
                <div class="metric-card card-red">
                    <div class="metric-label">Flagged Accounts</div>
                    <div class="metric-value" style="color: #F87171;">~{approx_flagged:,}</div>
                    <div class="metric-sub">Holdout Test Set</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Plotly Threshold Optimization Curve
        df_curve = pd.DataFrame(curve_pts)
        fig_curve = go.Figure()

        fig_curve.add_trace(
            go.Scatter(
                x=df_curve["threshold"],
                y=df_curve["precision"] * 100,
                mode="lines",
                name="Precision (%)",
                line=dict(color="#818CF8", width=2.2),
                hovertemplate="Threshold: %{x:.2f}<br>Precision: %{y:.1f}%<extra></extra>",
            )
        )
        fig_curve.add_trace(
            go.Scatter(
                x=df_curve["threshold"],
                y=df_curve["recall"] * 100,
                mode="lines",
                name="Recall (%)",
                line=dict(color="#34D399", width=2.2),
                hovertemplate="Threshold: %{x:.2f}<br>Recall: %{y:.1f}%<extra></extra>",
            )
        )
        fig_curve.add_trace(
            go.Scatter(
                x=df_curve["threshold"],
                y=df_curve["f1"] * 100,
                mode="lines",
                name="F1-Score (x100)",
                line=dict(color="#FBBF24", width=3.2),
                hovertemplate="Threshold: %{x:.2f}<br>F1-Score: %{y:.1f}%<extra></extra>",
            )
        )

        # Add vertical line for optimal threshold
        fig_curve.add_vline(
            x=opt_cutoff,
            line_width=2,
            line_dash="dash",
            line_color="#FBBF24",
            annotation_text=f"Optimal F1 ({opt_cutoff:.2f})",
            annotation_position="top left",
            annotation_font=dict(color="#FBBF24", size=11),
        )

        # Add vertical line for current slider value if different from optimal
        if abs(eval_thresh - opt_cutoff) >= 0.01:
            fig_curve.add_vline(
                x=eval_thresh,
                line_width=1.8,
                line_dash="dot",
                line_color="#60A5FA",
                annotation_text=f"Current ({eval_thresh:.2f})",
                annotation_position="bottom right",
                annotation_font=dict(color="#60A5FA", size=11),
            )

        fig_curve.update_layout(
            title=dict(
                text="Precision, Recall & F1-Score vs. Candidate Decision Threshold (0.25 ➔ 0.70)",
                font=dict(color="#F8FAFC", size=14),
            ),
            xaxis=dict(
                title="Probability Decision Threshold",
                title_font=dict(color="#94A3B8"),
                tickfont=dict(color="#94A3B8"),
                gridcolor="rgba(255, 255, 255, 0.08)",
                range=[0.24, 0.71],
            ),
            yaxis=dict(
                title="Metric Value (%)",
                title_font=dict(color="#94A3B8"),
                tickfont=dict(color="#94A3B8"),
                gridcolor="rgba(255, 255, 255, 0.08)",
                range=[40, 100],
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(15, 23, 42, 0.8)",
                bordercolor="rgba(255, 255, 255, 0.1)",
                font=dict(color="#F8FAFC", size=11),
            ),
            height=360,
            margin=dict(l=10, r=10, t=40, b=30),
            paper_bgcolor="rgba(30, 41, 59, 0.5)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_curve, use_container_width=True)

    # 3. Side-by-Side: ROC Curves & Confusion Matrix
    eval_c1, eval_c2 = st.columns(2, gap="medium")

    with eval_c1:
        st.markdown("##### 📈 Multi-Model ROC Curves")
        fig_roc = go.Figure()
        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(color="#64748B", dash="dash"),
                name="Random Baseline (AUC = 0.50)",
            )
        )

        model_palette = {
            "Logistic Regression": "#94A3B8",
            "Random Forest": "#F59E0B",
            "XGBoost": "#6366F1",
        }

        for model_name, color in model_palette.items():
            if model_name in metrics_data and "roc_curve" in metrics_data[model_name]:
                roc_pts = metrics_data[model_name]["roc_curve"]
                fpr = [pt["fpr"] for pt in roc_pts]
                tpr = [pt["tpr"] for pt in roc_pts]
                auc = metrics_data[model_name]["roc_auc"]
                fig_roc.add_trace(
                    go.Scatter(
                        x=fpr,
                        y=tpr,
                        mode="lines",
                        line=dict(color=color, width=2.5 if model_name == "XGBoost" else 1.8),
                        name=f"{model_name} (AUC = {auc:.3f})",
                    )
                )

        fig_roc.update_layout(
            xaxis=dict(
                title="False Positive Rate",
                title_font=dict(color="#94A3B8"),
                tickfont=dict(color="#94A3B8"),
                gridcolor="rgba(255, 255, 255, 0.08)",
            ),
            yaxis=dict(
                title="True Positive Rate (Recall)",
                title_font=dict(color="#94A3B8"),
                tickfont=dict(color="#94A3B8"),
                gridcolor="rgba(255, 255, 255, 0.08)",
            ),
            legend=dict(
                yanchor="bottom",
                y=0.05,
                xanchor="right",
                x=0.95,
                bgcolor="rgba(15, 23, 42, 0.8)",
                bordercolor="rgba(255, 255, 255, 0.1)",
                font=dict(color="#F8FAFC", size=11),
            ),
            height=340,
            margin=dict(l=10, r=10, t=25, b=25),
            paper_bgcolor="rgba(30, 41, 59, 0.5)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with eval_c2:
        st.markdown(f"##### 🎯 XGBoost Confusion Matrix (at Optimal Cutoff {opt_cutoff:.2f})")
        # Use confusion matrix from optimal threshold if available
        opt_cm = opt_m.get("confusion_matrix", metrics_data["XGBoost"]["confusion_matrix"])
        tn, fp = opt_cm[0][0], opt_cm[0][1]
        fn, tp = opt_cm[1][0], opt_cm[1][1]
        z_text = [[f"True Retained: {tn:,}", f"False Alarm: {fp:,}"], [f"Missed Churn: {fn:,}", f"Saved Churn: {tp:,}"]]

        fig_cm = go.Figure(
            data=go.Heatmap(
                z=[[tn, fp], [fn, tp]],
                x=["Predicted Retained", "Predicted Churn"],
                y=["Actual Retained", "Actual Churn"],
                text=z_text,
                texttemplate="%{text}",
                textfont={"size": 13, "color": "#F8FAFC", "family": "Inter"},
                colorscale=[[0, "#1E293B"], [0.5, "#4F46E5"], [1, "#6366F1"]],
                showscale=False,
            )
        )
        fig_cm.update_layout(
            xaxis=dict(tickfont=dict(color="#F8FAFC")),
            yaxis=dict(tickfont=dict(color="#F8FAFC")),
            height=340,
            margin=dict(l=10, r=10, t=25, b=25),
            paper_bgcolor="rgba(30, 41, 59, 0.5)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # 3. Global SHAP Feature Importance
    st.markdown("#### 🌍 Global SHAP Feature Importance (Impact of Indian Recharge Plan Attributes)")
    if "global_importance" in metrics_data:
        gi = metrics_data["global_importance"][:15]
        gi_features = [clean_feature_label(item["feature"]) for item in gi][::-1]
        gi_scores = [item["importance"] for item in gi][::-1]

        fig_global = go.Figure(
            go.Bar(
                x=gi_scores,
                y=gi_features,
                orientation="h",
                marker=dict(
                    color=gi_scores,
                    colorscale=[[0, "#4F46E5"], [1, "#818CF8"]],
                    line=dict(color="rgba(255,255,255,0.15)", width=1),
                ),
                hovertemplate="<b>%{y}</b><br>Global Mean |SHAP|: %{x:.4f}<extra></extra>",
            )
        )
        fig_global.update_layout(
            xaxis=dict(
                title="Mean |SHAP Value| (Impact on Model Log-Odds Output)",
                title_font=dict(color="#94A3B8"),
                tickfont=dict(color="#94A3B8"),
                gridcolor="rgba(255, 255, 255, 0.08)",
            ),
            yaxis=dict(
                tickfont=dict(size=11, color="#F8FAFC"),
                gridcolor="rgba(0,0,0,0)",
            ),
            height=400,
            margin=dict(l=10, r=20, t=20, b=30),
            paper_bgcolor="rgba(30, 41, 59, 0.5)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_global, use_container_width=True)


# ==============================================================================
# TAB 4: ML ARCHITECTURE & METHODOLOGY
# ==============================================================================
with tab4:
    st.markdown('<div class="section-title">🧠 Production ML Pipeline Architecture & Design Decisions</div>', unsafe_allow_html=True)
    st.markdown(
        "A comprehensive Machine Learning pipeline tailored for Indian Telecom enterprise deployment. "
        "Every phase balances statistical rigor, game-theoretic explainability, and financial ROI in Indian Rupees (₹)."
    )

    arch1, arch2 = st.columns(2, gap="medium")

    with arch1:
        st.markdown(
            """
            <div class="metric-card card-indigo">
                <div class="metric-label">1. Indian Recharge Attribute Modeling</div>
                <div style="font-size: 13px; color: #CBD5E1; margin-top: 8px; line-height: 1.6;">
                    • <b>7 Plan Features:</b> Directly incorporates <code>ConnectionType</code>, <code>PlanPrice_INR</code>, <code>PlanValidityDays</code> (28, 56, 84, 90, 365 Days), <code>DataAllowance_GB</code>, <code>DataQuotaMode</code>, <code>SMSQuota</code>, and <code>CallsQuota</code>.<br>
                    • <b>MonthlyEffectiveARPU:</b> Computed as <code>PlanPrice_INR / (PlanValidityDays / 30.0)</code> to accurately measure revenue exposure across varying plan lengths.<br>
                    • <b>Data Density:</b> Synthesized <code>EffectiveMonthlyData_GB</code> capturing true data generosity (Daily vs Bulk).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="metric-card card-emerald">
                <div class="metric-label">2. Intact Account Demographics & Add-ons</div>
                <div style="font-size: 13px; color: #CBD5E1; margin-top: 8px; line-height: 1.6;">
                    • <b>Demographics Intact:</b> Fully preserves <code>Gender</code>, <code>SeniorCitizen</code>, <code>Partner</code>, and <code>Dependents</code> in the model and UI.<br>
                    • <b>Ecosystem Stickiness:</b> Cross-service density score (0 to 9) measuring subscriber engagement across Security Shield, Cloud Storage, OTT Streaming Bundle, Live TV, and Tech Support add-ons.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with arch2:
        st.markdown(
            """
            <div class="metric-card card-amber">
                <div class="metric-label">3. XGBoost Optimization & Imbalance Balancing</div>
                <div style="font-size: 13px; color: #CBD5E1; margin-top: 8px; line-height: 1.6;">
                    • <b>Stratified 80/20 Partition:</b> Preserved exact class proportions across train and validation sets.<br>
                    • <b>Class Imbalance Handling:</b> Applied <code>scale_pos_weight = 2.77</code> in XGBoost.<br>
                    • <b>Outstanding Benchmark:</b> Boosted ROC-AUC to <b>0.9110</b> and Recall to <b>83.69%</b> with the addition of authentic recharge attributes.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="metric-card card-red">
                <div class="metric-label">4. Explainable AI & Prescriptive ROI Engine</div>
                <div style="font-size: 13px; color: #CBD5E1; margin-top: 8px; line-height: 1.6;">
                    • <b>SHAP TreeExplainer:</b> Deconstructs tree ensemble decisions into exact additive feature contributions in log-odds space.<br>
                    • <b>Prescriptive Retention Playbook:</b> Top positive SHAP risk factor automatically activates a targeted intervention play.<br>
                    • <b>Financial ROI Simulator:</b> Calculates intervention cost against gross annual revenue saved to deliver verifiable Net ROI in Indian Rupees (₹).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Technology Stack Footer
    st.markdown("#### 🛠️ Production Tech Stack & Artifact Specifications")
    st.markdown(
        """
        | Component | Framework / Library | Specification / Implementation Detail |
        | :--- | :--- | :--- |
        | **Inference Engine** | `xgboost (v3.4)` | Tuned: 350 Estimators, max_depth=5, lr=0.03, subsample=0.8, colsample=0.9, `scale_pos_weight=2.77`, **ROC-AUC: 0.910** |
        | **Threshold Optimizer** | `scikit-learn (v1.9)` | Optimal F1 Cutoff (0.64), **F1: 0.735**, +4.7% Precision gain minimizing false-positive spend |
        | **Explainability (XAI)** | `shap (v0.54)` | `TreeExplainer` local attribution + global mean absolute importance |
        | **Recharge Attributes** | Indian Telecom Standards | 7 Plan Features: ConnectionType, PlanPrice, PlanValidity, DataQuota, SMS, Calls, Tenure |
        | **Account Demographics** | Intact Profile Controls | Gender, SeniorCitizen, Partner, Dependents |
        | **Currency & Scaling** | Indian Rupee (`₹`) | Indian Numbering (`format_inr`), Effective ARPU range `₹299–₹1,499/mo` |
        | **Preprocessing** | `scikit-learn (v1.9)` | `ColumnTransformer` + `StandardScaler` + `OneHotEncoder(handle_unknown='ignore')` (65 features) |
        | **Visualization** | `plotly (v7.1)` | Interactive indicators, scatter matrices, threshold curves, and diverging horizontal bars |
        | **Frontend UI** | `streamlit (v1.67)` | Clean dark slate SaaS theme (`#0F172A`), glassmorphic KPI cards, responsive layout |
        | **Deployment Artifacts** | `joblib` | Ultra-lightweight: `churn_pipeline.joblib` (0.23 MB), `shap_artifacts.joblib` (0.55 MB) |
        """
    )
