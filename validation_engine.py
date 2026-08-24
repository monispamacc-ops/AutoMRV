import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import joblib

def calculate_ks_gini(y_true, y_prob):
    """Calculates KS statistic, Gini coefficient, and AUC score."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    ks_stat = np.max(tpr - fpr)
    auc = roc_auc_score(y_true, y_prob)
    gini = (2 * auc) - 1
    return float(ks_stat), float(gini), float(auc)

def calculate_psi(expected, actual, num_bins=10):
    """Calculates Population Stability Index (PSI) to check for data drift."""
    expected = np.array(expected)
    actual = np.array(actual)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    # Set breakpoints based on expected population distribution
    percentiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(expected, percentiles)
    bins[0] = -np.inf
    bins[-1] = np.inf

    expected_counts = np.histogram(expected, bins=bins)[0]
    actual_counts = np.histogram(actual, bins=bins)[0]

    # Convert to proportions and smooth zeros to prevent division errors
    expected_pct = np.where(expected_counts == 0, 0.0001, expected_counts) / len(expected)
    actual_pct = np.where(actual_counts == 0, 0.0001, actual_counts) / len(actual)

    # Compute PSI sum
    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_value)

def run_validation_checks():
    print("Loading datasets and trained models...")
    train_df = pd.read_csv("train_baseline.csv")
    oot_df = pd.read_csv("oot_population.csv")
    
    champion = joblib.load("champion_rf.pkl")
    features = ['age', 'income', 'loan_amount', 'credit_history_length', 'utilization_rate']
    
    # Generate probabilities for out-of-time data
    oot_probs = champion.predict_proba(oot_df[features])[:, 1]
    train_probs = champion.predict_proba(train_df[features])[:, 1]
    
    # Run metric calculations
    ks, gini, auc = calculate_ks_gini(oot_df['default'], oot_probs)
    score_psi = calculate_psi(train_probs, oot_probs)
    
    print("\n--- Validation Audit Results ---")
    print(f"AUC-ROC Score: {auc:.4f}")
    print(f"Gini Coefficient: {gini:.4f}")
    print(f"KS Statistic: {ks:.4f}")
    print(f"Score PSI (Drift Check): {score_psi:.4f}")

if __name__ == "__main__":
    run_validation_checks()