import streamlit as st
import pandas as pd
import numpy as np
import joblib
from validation_engine import calculate_ks_gini, calculate_psi

st.set_page_config(page_title="AutoMRV | Model Risk Governance", layout="wide")

st.title("🛡️ AutoMRV: Automated Model Risk Management & Validation Suite")
st.markdown("Independent Model Audit, Challenger Benchmarking, and Drift Detection Portal.")

# Sidebar Controls & Data Source Selection
st.sidebar.header("Audit Controls & Data")
data_mode = st.sidebar.radio("Select Data Source", ["Default Synthetic Credit Data", "Upload Custom CSV Datasets"])

@st.cache_resource
def load_default_assets():
    train_df = pd.read_csv("train_baseline.csv")
    oot_df = pd.read_csv("oot_population.csv")
    champion = joblib.load("champion_rf.pkl")
    challenger = joblib.load("challenger_lr.pkl")
    return train_df, oot_df, champion, challenger

if data_mode == "Default Synthetic Credit Data":
    train_df, oot_df, champion, challenger = load_default_assets()
    features = ['age', 'income', 'loan_amount', 'credit_history_length', 'utilization_rate']
    target_col = 'default'
else:
    st.sidebar.subheader("Upload Datasets")
    train_file = st.sidebar.file_uploader("Upload Baseline Training CSV", type=["csv"])
    oot_file = st.sidebar.file_uploader("Upload Out-of-Time (OOT) CSV", type=["csv"])
    
    if train_file and oot_file:
        train_df = pd.read_csv(train_file)
        oot_df = pd.read_csv(oot_file)
        
        # Load default models as fallback or use defaults
        champion = joblib.load("champion_rf.pkl")
        challenger = joblib.load("challenger_lr.pkl")
        
        # Auto-detect features (all numeric columns except target if possible)
        target_col = st.sidebar.selectbox("Select Target/Label Column", options=oot_df.columns, index=len(oot_df.columns)-1)
        features = [col for col in oot_df.columns if col != target_col and col in train_df.columns]
    else:
        st.info("Please upload both Baseline and OOT CSV files in the sidebar to proceed with custom data analysis.")
        st.stop()

# Sidebar Model Selection
model_choice = st.sidebar.selectbox("Select Model Architecture", ["Champion (Random Forest)", "Challenger (Logistic Regression)"])
active_model = champion if "Champion" in model_choice else challenger

# Dashboard Tabs
tab1, tab2, tab3 = st.tabs(["📊 Performance & Back-Testing", "📉 Drift & PSI Analysis", "⚖️ Champion vs. Challenger"])

with tab1:
    st.subheader("Out-of-Time (OOT) Performance & Discriminatory Power")
    
    oot_probs = active_model.predict_proba(oot_df[features])[:, 1]
    ks, gini, auc = calculate_ks_gini(oot_df[target_col], oot_probs)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("AUC-ROC Score", f"{auc:.3f}", "Target > 0.70")
    col2.metric("Gini Coefficient", f"{gini:.3f}", "Target > 0.40")
    col3.metric("KS Statistic", f"{ks:.3f}", "Target > 0.30")
    
    st.info("The Kolmogorov-Smirnov (KS) and Gini metrics confirm the model's ability to successfully segregate good records from target defaults on out-of-sample data.")

with tab2:
    st.subheader("Population Stability Index (PSI) & Feature Drift Monitoring")
    
    train_probs = active_model.predict_proba(train_df[features])[:, 1]
    score_psi = calculate_psi(train_probs, oot_probs)
    
    st.metric("Model Score PSI (Baseline vs OOT)", f"{score_psi:.4f}", 
              delta="Stable (< 0.1)" if score_psi < 0.1 else "Moderate/High Drift", 
              delta_color="normal" if score_psi < 0.1 else "inverse")
    
    st.markdown("### Individual Feature Drift Breakdown")
    feature_psi_results = []
    for feat in features:
        f_psi = calculate_psi(train_df[feat], oot_df[feat])
        status = "Stable" if f_psi < 0.1 else ("Moderate Drift" if f_psi < 0.25 else "Action Required")
        feature_psi_results.append({"Feature": feat, "PSI Value": round(f_psi, 4), "Status": status})
        
    st.dataframe(pd.DataFrame(feature_psi_results), use_container_width=True)

with tab3:
    st.subheader("Model Governance Trade-off Matrix")
    
    champ_probs_oot = champion.predict_proba(oot_df[features])[:, 1]
    chal_probs_oot = challenger.predict_proba(oot_df[features])[:, 1]
    
    c_ks, c_gini, c_auc = calculate_ks_gini(oot_df[target_col], champ_probs_oot)
    lr_ks, lr_gini, lr_auc = calculate_ks_gini(oot_df[target_col], chal_probs_oot)
    
    comparison_df = pd.DataFrame({
        "Validation Metric": ["AUC-ROC", "Gini Coefficient", "KS Statistic", "Interpretability Level", "Regulatory Compliance Risk"],
        "Challenger (Logistic Regression)": [f"{lr_auc:.3f}", f"{lr_gini:.3f}", f"{lr_ks:.3f}", "High (Transparent Weights)", "Low"],
        "Champion (Random Forest)": [f"{c_auc:.3f}", f"{c_gini:.3f}", f"{c_ks:.3f}", "Low (Black-Box)", "Medium-High"]
    })
    
    st.table(comparison_df)
    st.warning("Governance Note: While the Random Forest champion model yields incremental predictive lift, the Logistic Regression challenger model remains mandatory for regulatory audit compliance.")