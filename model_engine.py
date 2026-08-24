import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

def generate_credit_portfolio():
    print("Generating synthetic retail credit portfolio...")
    np.random.seed(42)
    n_samples = 20000
    
    # Core credit risk feature set
    age = np.random.randint(18, 70, size=n_samples)
    income = np.random.exponential(50000, size=n_samples) + 20000
    loan_amount = np.random.exponential(15000, size=n_samples) + 5000
    credit_history_length = np.random.randint(0, 15, size=n_samples)
    utilization_rate = np.random.beta(2, 5, size=n_samples)
    
    # Temporal partitioning for Out-of-Time (OOT) validation
    time_period = np.random.choice(
        ['Q1_Train', 'Q2_Train', 'Q3_Validation', 'Q4_OOT'], 
        size=n_samples, 
        p=[0.3, 0.3, 0.2, 0.2]
    )
    
    # Default probability calculation using simple math
    log_odds = -3 + 0.00001*loan_amount - 0.00002*income + 3*utilization_rate - 0.05*age
    default_prob = 1 / (1 + np.exp(-log_odds))
    default = np.random.binomial(1, np.clip(default_prob, 0, 1))
    
    portfolio_df = pd.DataFrame({
        'age': age,
        'income': income,
        'loan_amount': loan_amount,
        'credit_history_length': credit_history_length,
        'utilization_rate': utilization_rate,
        'time_period': time_period,
        'default': default
    })
    return portfolio_df

def execute_training_pipeline():
    df = generate_credit_portfolio()
    
    # Split data into training baseline vs out-of-time test set
    train_df = df[df['time_period'].isin(['Q1_Train', 'Q2_Train'])]
    oot_df = df[df['time_period'] == 'Q4_OOT']
    
    features = ['age', 'income', 'loan_amount', 'credit_history_length', 'utilization_rate']
    X_train, y_train = train_df[features], train_df['default']
    
    # Train Challenger Model (Logistic Regression)
    print("Training challenger model (Logistic Regression)...")
    challenger = LogisticRegression(max_iter=1000, random_state=42)
    challenger.fit(X_train, y_train)
    
    # Train Champion Model (Random Forest)
    print("Training champion model (Random Forest)...")
    champion = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    champion.fit(X_train, y_train)
    
    # Save files to your folder
    train_df.to_csv("train_baseline.csv", index=False)
    oot_df.to_csv("oot_population.csv", index=False)
    joblib.dump(challenger, "challenger_lr.pkl")
    joblib.dump(champion, "champion_rf.pkl")
    
    print("Done! Files saved successfully: train_baseline.csv, oot_population.csv, and model files.")

if __name__ == "__main__":
    execute_training_pipeline()