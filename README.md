# 🛡️ AutoMRV: Automated Model Risk Management & Validation Suite

## 💡 What is this project about?
Imagine a bank uses an AI program to decide whether someone should get a loan or if they might fail to pay it back. 

Usually, data scientists focus entirely on *building* that AI and making it accurate. But in the real world, banks have strict rules. Before an AI can be trusted with millions of rupees, an independent team of **risk auditors** must test it. They need to answer questions like:
* *Will this AI start failing if customer habits change next year?*
* *Is it secretly biased or unstable?*
* *Can we compare this complex AI against a simple, traditional math formula to see if it's actually safer?*

**AutoMRV** is a software tool that acts as that **quality control inspector**. Instead of building predictions, it takes an AI model and runs strict stress tests on it.

---

## 🔍 How Does It Work? (Step-by-Step)

1. **The Contest (Champion vs. Challenger):** 
   * **The Challenger:** A simple, old-school math formula (Logistic Regression) that banks have trusted for decades. It's easy to read and audit.
   * **The Champion:** A complex, modern AI model (Random Forest) that is very smart, but acts like a "black box."
   
2. **The Time-Travel Test (Out-of-Time Back-Testing):** 
   * The tool tests both models using future economic data to see if their accuracy holds up. It calculates professional scores like **AUC-ROC, Gini Coefficient, and KS Statistic** to grade how well they separate good customers from risky ones.

3. **The Habit-Change Test (Drift Detection):** 
   * If inflation rises or people's spending habits change, an old AI can break. Our tool uses a metric called **PSI (Population Stability Index)** to automatically check if the customer data today looks completely different from the data the AI was trained on.

4. **The Web Dashboard:** 
   * All these tests are packed into an interactive web app built with **Streamlit**, where anyone can click around, check charts, and upload custom files.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Data Science & ML:** Pandas, NumPy, Scikit-Learn
* **Web Interface:** Streamlit
* **Version Control:** Git & GitHub

---

## 🚀 How to Run It Locally

If you want to run this project on your computer, follow these simple steps:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/monispamacc-ops/AutoMRV.git](https://github.com/monispamacc-ops/AutoMRV.git)
   cd AutoMRV