# 🛡️ AutoMRV: Automated Model Risk Management & Validation Suite

An independent model audit, challenger benchmarking, and data drift detection platform engineered for retail credit risk portfolios.

---

##  Executive Summary
In retail banking, predictive models dictate credit approvals and risk provisioning. However, before deploying machine learning models into production, financial institutions require rigorous, independent validation to satisfy regulatory compliance (such as SR 11-7 guidelines). 

**AutoMRV** functions as an automated second line of defense. Rather than building models, it acts as an **independent model risk governance engine** that stress-tests predictive performance, benchmarks models against transparent regulatory baselines, and continuously audits population stability.

---

##  Core Architecture & Validation Workflow

AutoMRV evaluates credit portfolios through a four-stage audit pipeline:

1. **Challenger Benchmarking (Champion vs. Challenger):**
   * **Challenger (Logistic Regression):** A transparent, highly interpretable linear scorecard favored by regulatory auditors.
   * **Champion (Random Forest):** A complex, high-performing non-linear ensemble model. 
   * *Purpose:* Quantifies the predictive lift of black-box machine learning against traditional regulatory baselines.

2. **Out-of-Time (OOT) Back-Testing:**
   * Evaluates model generalization on temporal out-of-sample data.
   * Computes industry-standard banking metrics: **AUC-ROC** (discriminatory power), **Gini Coefficient** (inequality measure), and the **Kolmogorov-Smirnov (KS) Statistic** (maximum separation between good and bad accounts).

3. **Population Stability Index (PSI) Drift Monitoring:**
   * Automatically scans feature distributions and model score outputs to detect covariate and concept drift between baseline training data and recent production/OOT populations.

4. **Interactive Governance Dashboard:**
   * Packed into a real-time **Streamlit** web application, allowing risk managers and data science teams to inspect performance matrices, view drift diagnostics, and upload custom portfolio CSVs dynamically.

---

## Technical Stack
* **Programming Language:** Python
* **Data Science & Machine Learning:** Scikit-Learn, Pandas, NumPy
* **Model Persistence & Serialization:** Joblib
* **Web Framework:** Streamlit
* **Version Control:** Git & GitHub

---

##  Getting Started Locally

Follow these steps to set up and run the validation suite on your local machine:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/monispamacc-ops/AutoMRV.git](https://github.com/monispamacc-ops/AutoMRV.git)
   cd AutoMRV