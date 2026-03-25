import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

def wind_to_compass8(series):
    # series in degrees [0,360), NaNs allowed
    bins = np.arange(-22.5, 360+45, 45)  # -22.5..22.5 -> N, etc.
    labels = ['N','NE','E','SE','S','SW','W','NW','N']  # last 'N' for wrap; we'll map mod 8
    # shift degrees so bins align, then digitize
    deg = series % 360
    idx = np.digitize(deg, bins) - 1
    # map indices 0..7 to labels 0..7
    idx = idx % 8
    return pd.Categorical([labels[i] for i in idx], categories=labels[:8])

def ks_test_by_heatwave(
    df,
    column,
    heatwave_col='heat_wave',
    alpha=0.05,
    verbose=True
):
    """
    Apply two-sample Kolmogorov–Smirnov test between
    heatwave and no-heatwave days for a given column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    column : str
        Numerical column to test.
    heatwave_col : str, default 'heat_wave'
        Binary column indicating heatwave days (True/False).
    alpha : float, default 0.05
        Significance level.
    verbose : bool, default True
        Whether to print interpretation.

    Returns
    -------
    dict with keys:
        'ks_statistic', 'p_value', 'n_heatwave', 'n_no_heatwave'
    """

    # Split samples
    hw = df.loc[df[heatwave_col] == True, column].dropna()
    nhw = df.loc[df[heatwave_col] == False, column].dropna()

    ks_stat, p_value = ks_2samp(hw, nhw)

    if verbose:
        print(f"KS Statistic: {ks_stat:.4f}")
        print(f"P-value: {p_value:.4e}")
        if p_value < alpha:
            print("Reject H0: Distributions differ.")
        else:
            print("Fail to reject H0: No strong evidence distributions differ.")

    return {
        "column": column,
        "ks_statistic": ks_stat,
        "p_value": p_value,
        "n_heatwave": len(hw),
        "n_no_heatwave": len(nhw)
    }


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def pca_by_feature_groups(
    feature_groups: pd.Series,
    df: pd.DataFrame,
    unusable_cols: list
):
    """
    Applies full PCA to each correlated feature group.

    Parameters
    ----------
    feature_groups : pd.Series
        Each value is an iterable of column names (correlated group).
    df : pd.DataFrame
        DataFrame containing all feature columns.
    unusable_cols : list
        Columns that must be excluded from PCA (model-specific).

    Returns
    -------
    pca_results : dict
        Keys are feature_groups indices.
        Values are dicts containing:
            - 'pca': fitted PCA object
            - 'components': PCA-transformed DataFrame
            - 'explained_variance_ratio'
            - 'cumulative_pve'
            - 'used_columns'
    feature_groups_with_pve : pd.Series
        Original feature_groups with an added column 'cumulative_pve'
    """

    pca_results = {}
    group_sizes = []
    groups = []
    cumulative_pves = []

    for idx, group in feature_groups.items():
        group_cols = list(group)

        # Drop unusable columns
        group_cols = [c for c in group_cols if c not in unusable_cols]

        # If nothing usable remains
        if len(group_cols) < 2:
            cumulative_pves.append(np.nan)
            groups.append(group_cols)
            group_sizes.append(0)
            continue

        X = df[group_cols]

        # Drop rows with NaNs (PCA cannot handle missing values)
        X_clean = X.dropna()

        if X_clean.shape[0] < 2:
            cumulative_pves.append(np.nan)
            continue

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)

        # Full PCA
        pca = PCA(n_components=X_scaled.shape[1])
        X_pca = pca.fit_transform(X_scaled)

        explained_var = pca.explained_variance_ratio_
        cumulative_pve = np.cumsum(explained_var).tolist()

        # Store results
        pca_results[idx] = {
            "pca": pca,
            "components": pd.DataFrame(
                X_pca,
                index=X_clean.index,
                columns=[f"PC{i+1}" for i in range(X_pca.shape[1])]
            ),
            "explained_variance_ratio": explained_var,
            "cumulative_pve": cumulative_pve,
            "used_columns": group_cols,
        }

        cumulative_pves.append(cumulative_pve)
        groups.append(group_cols)
        group_sizes.append(len(cumulative_pve))

    # Attach cumulative PVE info back to feature_groups
    feature_groups_with_pve = feature_groups.copy()
    feature_groups_with_pve = feature_groups_with_pve.to_frame(name="feature_group")
    feature_groups_with_pve.loc[:, 'feature_group'] = pd.Series(groups)
    feature_groups_with_pve["group_size"] = group_sizes
    feature_groups_with_pve["cumulative_pve"] = cumulative_pves

    return pca_results, feature_groups_with_pve

def combine_train_test_metrics(metric_result_train, metric_result_test):
    "Combines train and test metric dictionaries into one dataframe."
    return pd.DataFrame({
            'train': metric_result_train,
            'test': metric_result_test
        }, index=['recall', 'precision', 'pr_auc', 'roc_auc', 'pr_auc_adj']).round(4)

from scipy.stats import wasserstein_distance
def compute_wasserstein_report(train_set, test_set, 
                               h_train, h_test,
                               nh_train, nh_test,
                               all_cols, used_cols, 
                               feature_importances,
                               n_bootstrap=30, random_state=4
                               ):
    """ Compute wasserstein distances of whole train test sets, and heatwave/non-heatwave subsets for full feature space.
    Uses bootstrapping for heatwave test set.
    Returns the results in a dataframe ordered by feature importances in descending order.
    """
    results = []

    for i, col in enumerate(all_cols):
        # All instances
        w_all = wasserstein_distance(train_set[col], test_set[col])

        # Non heatwave instances
        w_nh = wasserstein_distance(nh_train[col], nh_test[col])

        
        # Heatwave instances with bootstrapping test set
        rng = np.random.default_rng(random_state) # For reproducibility
        boot_distances = []
        n_test_samples = len(h_test[col])

        for _ in range(n_bootstrap):
            boot_sample = rng.choice(h_test[col], size=n_test_samples, replace=True)
            w_boot = wasserstein_distance(h_train[col], boot_sample)
            boot_distances.append(w_boot)

        w_h_mean = np.mean(boot_distances)
        w_h_std = np.std(boot_distances)

        importance = 0
        if col in used_cols:
            idx = used_cols.index(col)
            importance = feature_importances[idx]

        results.append({
            "feature": col,
            "importance": importance,
            "w_all": w_all,
            "w_nh": w_nh,
            "w_h_mean": w_h_mean,
            "w_h_std": w_h_std
        })

    df_results = pd.DataFrame(results).round(4)
    df_results = df_results.sort_values(
        by="importance",
        ascending=False
    ).reset_index(drop=True)

    return df_results