import pandas as pd
import numpy as np
from scipy.stats import rankdata, chi2, wilcoxon
from numpy.linalg import pinv
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

def run_skillings_mack(data):
    """
    Performs the Skillings-Mack test for non-parametric blocked data with missing values.
    """
    n_models = data.shape[1]
    
    # Within-block ranking (higher score is better, gets smaller rank)
    def rank_with_nans(row):
        mask = ~np.isnan(row)
        result = np.full_like(row, np.nan)
        if mask.sum() > 0:
            result[mask] = rankdata(-row[mask], method='average')
        return result

    ranks = data.apply(rank_with_nans, axis=1, raw=True).to_numpy()
    
    # Compute rank sums R_j
    R = np.nansum(ranks, axis=0)
    
    # Compute expected rank sums E[R_j]
    block_sizes = np.sum(~np.isnan(ranks), axis=1)
    E = np.zeros(n_models)
    for j in range(n_models):
        present = ~np.isnan(ranks[:, j])
        E[j] = np.sum(block_sizes[present] + 1) / 2
        
    # Compute variance-covariance matrix V
    V = np.zeros((n_models, n_models))
    for i in range(ranks.shape[0]):
        k = block_sizes[i]
        if k < 2:
            continue
        
        present_indices = np.where(~np.isnan(ranks[i, :]))[0]
        
        # Diagonal elements
        diag_val = (k**2 - 1) / 12
        for idx in present_indices:
            V[idx, idx] += diag_val
        
        # Off-diagonal elements
        off_diag_val = -(k + 1) / 12
        
        for j1_idx, j2_idx in combinations(present_indices, 2):
            V[j1_idx, j2_idx] += off_diag_val
            V[j2_idx, j1_idx] += off_diag_val

    # Compute test statistic T
    if np.linalg.matrix_rank(V) < n_models:
        print("Warning: Using pseudo-inverse for singular covariance matrix.")
        V_inv = pinv(V)
        df = np.linalg.matrix_rank(V)
    else:
        V_inv = np.linalg.inv(V)
        df = n_models - 1

    d = R - E
    T = d @ V_inv @ d
    p_value = 1 - chi2.cdf(T, df)
    
    return T, p_value, df

def calculate_descriptive_stats(data):
    """
    Calculates median, IQR, mean, SD for continuous data.
    """
    models = data.columns
    results = []
    
    for model in models:
        scores = data[model].dropna()
        if len(scores) > 0:
            median = scores.median()
            q1 = scores.quantile(0.25)
            q3 = scores.quantile(0.75)
            iqr = q3 - q1
            mean = scores.mean()
            sd = scores.std()
            n = len(scores)
        else:
            median, iqr, mean, sd, n = np.nan, np.nan, np.nan, np.nan, 0
            
        results.append({
            "Model": model,
            "Median": median,
            "IQR": iqr,
            "Mean": mean,
            "SD": sd,
            "N": n
        })
        
    return pd.DataFrame(results).sort_values("Median", ascending=False)

def run_pairwise_wilcoxon(data):
    """
    Performs pairwise Wilcoxon signed-rank tests for continuous data.
    """
    models = data.columns
    results = []
    
    for model1, model2 in combinations(models, 2):
        paired_data = data[[model1, model2]].dropna()
        
        if len(paired_data) < 10: # Not enough samples for meaningful test
            continue
        
        # Check if differences are all zero
        if (paired_data[model1] == paired_data[model2]).all():
            continue

        stat, p_value = wilcoxon(paired_data[model1], paired_data[model2], zero_method='zsplit')
        
        results.append({
            "Model 1": model1,
            "Model 2": model2,
            "p_value": p_value,
            "n_pairs": len(paired_data)
        })
        
    results_df = pd.DataFrame(results)
    
    # Bonferroni correction
    if not results_df.empty:
        results_df["p_value_corr"] = results_df["p_value"] * len(results_df)
        results_df["p_value_corr"] = results_df["p_value_corr"].clip(upper=1.0)
    
    return results_df.sort_values("p_value_corr")

# --- Main Execution ---
if __name__ == "__main__":
    # Load and preprocess data
    try:
        data = pd.read_csv("semantic quality.csv", sep=";", index_col=0)
        data.replace('', np.nan, inplace=True)
        data = data.astype(float)
    except FileNotFoundError:
        print("Error: 'semantic quality.csv' not found. Please ensure the file is in the correct directory.")
        exit()

    print("### 1. Global Test: Skillings-Mack for Semantic Quality ###")
    print("="*60)
    sm_stat, sm_p, sm_df = run_skillings_mack(data)
    print(f"Skillings-Mack Statistic (T): {sm_stat:.2f}")
    print(f"Degrees of Freedom: {sm_df}")
    print(f"P-value: {sm_p:.2e}")
    if sm_p < 0.05:
        print("Conclusion: Statistically significant difference among models.")
    else:
        print("Conclusion: No statistically significant difference found.")

    print("\n### 2. Descriptive Statistics: Median, IQR, Mean, SD ###")
    print("="*60)
    descriptive_df = calculate_descriptive_stats(data)
    print(descriptive_df.to_string(index=False))

    print("\n### 3. Pairwise Comparisons: Wilcoxon Signed-Rank Test ###")
    print("="*70)
    print("Significant differences (p < 0.05 after Bonferroni correction):")
    
    pairwise_results = run_pairwise_wilcoxon(data)
    if not pairwise_results.empty:
        significant_pairs = pairwise_results[pairwise_results["p_value_corr"] < 0.05]
        if not significant_pairs.empty:
            print(f"\n{len(significant_pairs)} significant pairs found:")
            print(significant_pairs[["Model 1", "Model 2", "p_value_corr", "n_pairs"]].to_string(index=False))
        else:
            print("No significant pairs found after correction.")
    else:
        print("No pairs had sufficient data for comparison.") 