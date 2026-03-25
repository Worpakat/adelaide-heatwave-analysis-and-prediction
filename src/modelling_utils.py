import numpy as np
import pandas as pd
import ast
from time import perf_counter
import random

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid

from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve, roc_curve, auc

RANDOM_STATE = 4 # For reproducibility


def adjust_pr_curve_from_roc(fpr, tpr, target_prevalence):
    """
    Adjust PR curve to a new prevalence using ROC curve, target_prevalence with Bayes Rule.

    Parameters
    ----------
    fpr : array-like
        False positive rates from ROC curve.
    tpr : array-like
        True positive rates from ROC curve.
    target_prevalence : float
        New prevalence to adjust PR curve to.

    Returns
    -------
    dict with:
        'precision': adjusted precision values
        'recall': recall values (same as TPR)
        'pr_auc': adjusted PR-AUC
    """
    pi = target_prevalence

    # Avoid division by zero
    denominator = (pi * tpr) + ((1 - pi) * fpr)
    precision = np.divide(
        pi * tpr,
        denominator,
        out=np.ones_like(tpr),
        where=denominator != 0
    )

    recall = tpr # For naming convenience
    pr_auc = auc(recall, precision)

    return {
        "precision": precision,
        "recall": recall,
        "pr_auc": pr_auc
    }

def get_classification_metrics(y, y_pred, y_score, target_prevalence=None):
    """Calculates classification metrics for a binary classification problem.
    Returns two dict. First contains precision, recall, f1, f2, and pr-auc.
    Second contains precision and recall.
    """
    precision, recall, _, _ = precision_recall_fscore_support(y, y_pred, zero_division=0)

    # PR-Curve
    precisions, recalls, _ = precision_recall_curve(y, y_score)
    pr_auc = auc(recalls, precisions)
    pr_auc_dict = {'recall': recalls, 'precision': precisions, 'pr_auc': pr_auc}

    # ROC
    fpr, tpr, _ = roc_curve(y, y_score)
    roc_auc = auc(fpr, tpr)
    roc_auc_dict = {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}
    
    # Adjusted PR-AUC
    if target_prevalence is not None:
        pr_auc_adj = adjust_pr_curve_from_roc(fpr, tpr, target_prevalence)

    metrics_dict = {
        'recall': recall[1],
        'precision': precision[1],
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'pr_auc_adj':pr_auc_adj['pr_auc'] if target_prevalence is not None else None
    }

    return metrics_dict, pr_auc_dict, roc_auc_dict


def apply_pca_to_group(
    df_group: pd.DataFrame,
    pc_count: int,
    pca: PCA | None = None,
    prefix: str = "PC"
):
    """
    Fits or applies PCA to a feature group.

    Parameters
    ----------
    df_group : pd.DataFrame
        Feature group data (rows = samples, cols = features).
    pc_count : int
        Number of principal components to keep.
    pca : PCA or None, optional
        Pre-fitted PCA object. If None, a new PCA is fitted.
    prefix : str
        Prefix for principal component column names.

    Returns
    -------
    pca : PCA
        Fitted PCA object.
    pcs_df : pd.DataFrame
        DataFrame containing principal components.
    """

    if pca is None:
        pca = PCA(n_components=pc_count)
        pcs = pca.fit_transform(df_group)
    else:
        pcs = pca.transform(df_group)[:, :pc_count]

    pcs_df = pd.DataFrame(
        pcs,
        index=df_group.index,
        columns=[f"{prefix}{i+1}" for i in range(pc_count)]
    )

    return pca, pcs_df

def extract_pcas_from_groups(
        feature_groups_df, 
        X_train_scaled, 
        X_train_parts, 
        X_test_scaled=None, 
        X_test_parts=None, 
        _pcas_=None):
    """
    Extracts principal components from feature groups and adds them to training and test sets.
    If _pcas_ is provided (as fitted PCAs), uses previously fitted PCAs.
    Returns a dictionary of fitted PCA objects, X_train_parts, and X_test_parts.
    """
    pcas = {} if _pcas_ is None else _pcas_ # In case of fitted PCAs are provided.
    test_not_inclueded = X_test_scaled is None or X_test_parts is None

    for i, row in feature_groups_df.iterrows():
        features = ast.literal_eval(row["feature_group"])
        pc_count = int(row["to_be_used_pcs"])
        Xtr_group = X_train_scaled[features]

        pca = pcas[i] if i in pcas else None
        # In case of it is provided, use previously fitted PCAs
        # Otherwise, assigning None will make ``apply_pca_to_group`` fit a new PCA.

        pca, Xtr_pcs = apply_pca_to_group( # ! Returned fitted PCA
            Xtr_group, pc_count=pc_count, pca=pca, prefix=f"G{i}_PC"
        )
        X_train_parts.append(Xtr_pcs)
        pcas[i] = pca
        
        if test_not_inclueded: continue # In case of test data is not provided, we pass.

        Xte_group = X_test_scaled[features]

        _, Xte_pcs = apply_pca_to_group(
            Xte_group, pc_count=pc_count, pca=pca, prefix=f"G{i}_PC"
        )

        X_test_parts.append(Xte_pcs)


    return pcas, X_train_parts, X_test_parts


def get_final_feature_matrices(
        X_train: pd.DataFrame | None = None,
        X_test: pd.DataFrame | None = None,    
        feature_groups_df: pd.DataFrame = pd.DataFrame(),
        non_standardized_features: list[str] | None = None,
        features_not_to_be_used: list[str] | None = None,
        scaler: StandardScaler | None = None,
        pcas: dict | None = None
    ):
    """
    Scales features and applies PCA to groups. 
    Can fit on X_train and return both X_train_final and X_test_final,
    or use only X_test with provided scaler/pcas for inference purposes.
    """
    non_standardized_features = non_standardized_features or []
    X_train_final, X_test_final = None, None

    # ---- Parse feature groups ----
    group_features = set()
    for g in feature_groups_df["feature_group"]:
        group_features.update(ast.literal_eval(g))

    # Identify features based on whichever dataframe is provided
    reference_df = X_train if X_train is not None else X_test
    all_features = set(reference_df.columns)
    standardizable_features = list(all_features - set(non_standardized_features))
    non_group_standardized = list(set(standardizable_features) - group_features)

    # ---- Scaling Logic ----
    X_train_scaled, X_test_scaled = None, None

    if X_train is not None:
        # Fit mode: Use or create a new scaler
        scaler = scaler or StandardScaler()
        X_train_scaled = X_train.copy()
        X_train_scaled[standardizable_features] = scaler.fit_transform(X_train[standardizable_features])
        
        if X_test is not None:
            X_test_scaled = X_test.copy()
            X_test_scaled[standardizable_features] = scaler.transform(X_test[standardizable_features])
    else:
        # Inference mode: Must have a scaler provided
        if scaler is None:
            raise ValueError("Inference mode: 'scaler' must be provided if 'X_train' is None.")
        
        if X_test is not None:
            X_test_scaled = X_test.copy()
            X_test_scaled[standardizable_features] = scaler.transform(X_test[standardizable_features])

    # ---- Setup Parts Processing ----
    def process_set(original_df, scaled_df):
        parts = []
        if non_group_standardized:
            parts.append(scaled_df[non_group_standardized])
        if non_standardized_features:
            parts.append(original_df[non_standardized_features])
        return parts

    # ---- Logic Branch: Training vs. Prediction ----
    if X_train is not None:
        # 1. TRAINING MODE (Train + Evaluate)
        X_train_parts = process_set(X_train, X_train_scaled)
        
        # Prepare test parts if X_test was provided for evaluation
        X_test_parts = []
        if X_test is not None:
            X_test_parts = process_set(X_test, X_test_scaled)
        
        # Fit PCAs on train and transform both sets at once
        pcas, X_train_parts, X_test_parts = extract_pcas_from_groups(
            feature_groups_df, 
            X_train_scaled, X_train_parts, 
            X_test_scaled if X_test is not None else None, 
            X_test_parts
        )
        
        X_train_final = pd.concat(X_train_parts, axis=1)
        if X_test is not None:
            X_test_final = pd.concat(X_test_parts, axis=1)

    else:
        # 2. PREDICTION MODE (Inference only)
        # We assume X_test contains the data to be predicted
        if X_test_scaled is not None:
            X_test_parts = process_set(X_test, X_test_scaled)
            
            # Use the specific inference signature for PCAs
            _, X_test_parts, _ = extract_pcas_from_groups(
                feature_groups_df, 
                X_test_scaled, 
                X_test_parts, 
                _pcas_=pcas
            )
            
            X_test_final = pd.concat(X_test_parts, axis=1)
            X_train_final = None # Ensure train is returned as None

    # ---- Final Cleanup ----
    for df in [X_train_final, X_test_final]:
        if df is not None and features_not_to_be_used:
            df.drop(columns=features_not_to_be_used, inplace=True, errors='ignore')

    return X_train_final, X_test_final, scaler, pcas


def get_preds_and_scores(model, X):
    """Produces predictions and scores from a model and a feature matrix, then returns them."""
    y_pred = None
    y_score = None

    if type(model) == SVC: # For 'SVC'
        # ! SVC predict_proba() method uses internal CV, which is not compatible with TimeSeriesSplit.
        # ! Therefore, we use decision_function() method instead.
        y_score = model.decision_function(X) 
    else:
        y_score = model.predict_proba(X)[:, 1]

    threshold = 0 if type(model) == SVC else 0.5
    # 'SVC' predicts 1 for positive class, and 0 for negative class.
    # For other models, default probability threshold is 0.5.   
    
    y_pred = y_score.copy()
    y_pred[y_pred > threshold] = 1 
    y_pred[y_pred <= threshold] = 0 

    return y_pred, y_score


def train_test_with_groups_and_pca(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train,
    y_test,
    model,
    feature_groups_df: pd.DataFrame,
    non_standardized_features: list[str] | None = None,
    features_not_to_be_used: list[str] | None = None,
    metric_fn=None,
    metric_kwargs: dict | None = None,
):
    """
    Train and test a model using grouped PCA features with proper scaling.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Feature matrices.
    y_train, y_test : array-like
        Target vectors.
    model : sklearn-like estimator
        Unfitted model with fit/predict interface.
    feature_groups_df : pd.DataFrame
        Must contain:
          - 'feature_group' (stringified list of column names)
          - 'to_be_used_pcs' (int)
    non_standardized_features : list[str], optional
        Features excluded from standardization.
    features_not_to_be_used : list[str], optional
        List of features not to be used, selected during feature selection.
    metric_fn : callable, optional
        Function taking (y_true, y_pred, **kwargs).
    metric_kwargs : dict, optional
        Extra keyword arguments for metric_fn.

    Returns
    -------
    model : fitted model
    metric_result : output of metric_fn (or None)
    artifacts : dict
        Contains scalers, PCAs, and feature names.
    """
    (
    X_train_final,
    X_test_final, 
    scaler, 
    pcas) = get_final_feature_matrices(X_train, 
                                       X_test, 
                                       feature_groups_df, 
                                       non_standardized_features, 
                                       features_not_to_be_used)

    # ---- Train ----
    if type(model) == XGBClassifier and model.get_params()['early_stopping_rounds'] is not None:
        # In case of 'XGBoost' early stopping is used.
        model.fit(X_train_final, y_train, 
                  eval_set=[(X_test_final, y_test)],
                  verbose=False)
    else:
        model.fit(X_train_final, y_train)

    # ---- Evaluate ----
    y_pred_train, y_score_train = get_preds_and_scores(model, X_train_final)
    y_pred_test, y_score_test = get_preds_and_scores(model, X_test_final)

    metric_kwargs = metric_kwargs or {}
    metric_result_train = None
    metric_result_test = None
    if metric_fn is not None:
        metric_result_train = metric_fn(y_train, y_pred_train, y_score_train, **metric_kwargs)
        metric_result_test = metric_fn(y_test, y_pred_test, y_score_test, **metric_kwargs)

    artifacts = {
        "scaler": scaler,
        "pcas": pcas,
        "y_pred_train": y_pred_train,
        "y_score_train": y_score_train,
        "y_pred_test": y_pred_test,
        "y_score_test": y_score_test,
        "feature_names": X_train_final.columns.tolist(),
    }

    return model, metric_result_train, metric_result_test, artifacts

def train_models_with_cv(
        X_train, 
        y_train, 
        feature_groups_df, 
        models_info_and_params,
        non_standardized_features,
        not_to_be_used_at_linear,
        not_to_be_used_at_nonlinear,
        cv_splits,
        results_path
        ):
    tscv = TimeSeriesSplit(n_splits=cv_splits, test_size=365*2) 
    records = []
    all_model_results = {}
    try:
        for model_info in models_info_and_params:
            print("Model: ", model_info['name'])

            grid = ParameterGrid(model_info['params']) # Associated parameter grid

            for params in grid:
                print("Parameters: ", params)
                for fold, (train_idx, test_idx) in enumerate(tscv.split(X_train)):
                    X_train_cv, X_test_cv = X_train.iloc[train_idx], X_train.iloc[test_idx]
                    y_train_cv, y_test_cv = y_train.iloc[train_idx], y_train.iloc[test_idx]

                    if 'scale_pos_weight' in params.keys(): # For XGBoost
                        if params['scale_pos_weight']:
                            params['scale_pos_weight'] = y_train_cv.value_counts()[False] / y_train_cv.value_counts()[True]
                        else:
                            params['scale_pos_weight'] = None

                    model = model_info['model'](**params)
                    features_not_to_be_used = not_to_be_used_at_linear if model_info['type'] == 'linear' else not_to_be_used_at_nonlinear

                    start = perf_counter() # For timing

                    (model, 
                    metric_result_train, 
                    metric_result_test, 
                    _) = train_test_with_groups_and_pca(
                        X_train=X_train_cv,
                        X_test=X_test_cv,
                        y_train=y_train_cv,
                        y_test=y_test_cv,
                        model=model,
                        feature_groups_df=feature_groups_df,
                        non_standardized_features=non_standardized_features,
                        features_not_to_be_used=features_not_to_be_used,
                        metric_fn=get_classification_metrics,
                        metric_kwargs={'target_prevalence': y_train_cv.value_counts(normalize=True)[True]}
                    )

                    stop = perf_counter()

                    records.append({
                        "model": model_info['name'],
                        "param": params,
                        "fold": fold+1,
                    })
            
                    # Add train and test metrics with prefix
                    records[-1].update({'test_'+ k: v for k, v in metric_result_test[0].items()}) 
                    records[-1].update({'train_'+ k: v for k, v in metric_result_train[0].items()}) 

                    records[-1]["time"] = stop - start # Total training and evaluation time.    

                    if type(model) == XGBClassifier: # For 'XGBoost'
                        records[-1]['n_estimators'] = model.best_iteration + 1 # Actual estimator count after model training.

            result_df = pd.DataFrame(records)
            all_model_results[model_info['name']] = result_df
            records.clear()

    except Exception as e:
        print(e)

    finally: # ! In case of any error occurs, we save the results of executed training-evaluation process.
        pd.concat(all_model_results.values()).to_csv(results_path, index=False)




def optimize_threshold_cv(
        model_class,
        model_params,
        X_train, 
        y_train, 
        feature_groups_df, 
        non_standardized_features,
        features_not_to_be_used,
        cv_splits,
        precision_limit=0.4
        ):
    """
    Runs CV to find the best threshold per fold where Precision > precision_limit,
    then maximizes Recall.
    """
    tscv = TimeSeriesSplit(n_splits=cv_splits, test_size=365*2) 
    records = []
    fold_metrics_with_thresholds = {}
    # In case of we want to generate plots or exmaine in details, we return all precisions, recalls and thresholds of each fold.

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_train)):
        X_train_cv, X_test_cv = X_train.iloc[train_idx], X_train.iloc[test_idx]
        y_train_cv, y_test_cv = y_train.iloc[train_idx], y_train.iloc[test_idx]
        
        model = model_class(**model_params)

        np.random.seed(RANDOM_STATE)
        random.seed(RANDOM_STATE)
        # Train and Test CV model
        _, _, _, artifacts = train_test_with_groups_and_pca( # We don't need trained model, metrics
            X_train=X_train_cv,
            X_test=X_test_cv,
            y_train=y_train_cv,
            y_test=y_test_cv,
            model=model,
            feature_groups_df=feature_groups_df,
            non_standardized_features=non_standardized_features,
            features_not_to_be_used=features_not_to_be_used,
        )

        # Get Precision-Recall with thresholds.
        precisions, recalls, thresholds = precision_recall_curve(y_test_cv, artifacts['y_score_test'])
        
        # Filter thresholds that satisfy the precision constraint
        valid_indices = np.where(precisions[:-1] >= precision_limit)[0]
        best_idx = valid_indices[np.argmax(recalls[valid_indices])]
        best_threshold = thresholds[best_idx]
        best_precision = precisions[best_idx]
        best_recall = recalls[best_idx]

        records.append({
            "fold": fold+1,
            "best_threshold": best_threshold,
            "best_precision": best_precision,
            "best_recall": best_recall
        })

        fold_metrics_with_thresholds[fold+1] = {
            "precisions": precisions,
            "recalls": recalls,
            "thresholds": thresholds
        }

    result_df = pd.DataFrame(records)
    return result_df, fold_metrics_with_thresholds       
    





###=========DEPRECATED==========###
def get_final_feature_matrices_deprecated(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,    
        feature_groups_df: pd.DataFrame,
        non_standardized_features: list[str] | None = None,
        features_not_to_be_used: list[str] | None = None
    ):
    """
    Scales features and applies PCA to feature groups.
    Returns feature matrices, fitted standard scaler and fitted PCAs.
    """
    non_standardized_features = non_standardized_features or []

    # ---- Parse feature groups ----
    group_features = set()
    for g in feature_groups_df["feature_group"]:
        group_features.update(ast.literal_eval(g))

    all_features = set(X_train.columns)

    # ---- Feature categories ----
    standardizable_features = list(all_features - set(non_standardized_features))
    non_group_standardized = list(
        set(standardizable_features) - group_features
    )

    # ---- Scale (once, globally) ----
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[standardizable_features] = scaler.fit_transform(
        X_train[standardizable_features]
    )
    X_test_scaled[standardizable_features] = scaler.transform(
        X_test[standardizable_features]
    )

    X_train_parts = []
    X_test_parts = []

    # ---- Pass-through features ----
    if non_group_standardized:
        X_train_parts.append(X_train_scaled[non_group_standardized])
        X_test_parts.append(X_test_scaled[non_group_standardized])

    if non_standardized_features:
        X_train_parts.append(X_train[non_standardized_features])
        X_test_parts.append(X_test[non_standardized_features])

    # ---- PCA per group ----
    pcas, X_train_parts, X_test_parts = extract_pcas_from_groups(
        feature_groups_df, X_train_scaled, X_train_parts, X_test_scaled, X_test_parts
    )

    # ---- Final matrices ----
    X_train_final = pd.concat(X_train_parts, axis=1)
    X_test_final = pd.concat(X_test_parts, axis=1)
    
    if features_not_to_be_used is not None and features_not_to_be_used: # For selected features
        X_train_final.drop(columns=features_not_to_be_used, inplace=True)
        X_test_final.drop(columns=features_not_to_be_used, inplace=True)

    return X_train_final, X_test_final, scaler, pcas


def predict_deprecated(
    X: pd.DataFrame, 
    y,
    model,
    scaler: StandardScaler,
    pcas,
    feature_groups_df: pd.DataFrame,
    non_standardized_features: list[str] | None = None,
    features_not_to_be_used: list[str] | None = None,
    metric_fn=None,
    metric_kwargs: dict | None = None,
):
    """
    Train and test a model using grouped PCA features with proper scaling.

    Parameters
    ----------
    X: pd.DataFrame
        Feature matrice.
    y: array-like
        Target vectors.
    model : sklearn-like estimator
        Fitted model with fit/predict interface.
    scaler : StandardScaler
        Sklearn standardizer fitted with feature to be standardized.
    pcas : 
        PCA dictionary retrieved from 'train_test_with_groups_and_pca()'.
    feature_groups_df : pd.DataFrame
        Must contain:
          - 'feature_group' (stringified list of column names)
          - 'to_be_used_pcs' (int)
    non_standardized_features : list[str], optional
        Features excluded from standardization.
    features_not_to_be_used : list[str], optional
        List of features not to be used, selected during feature selection.
    metric_fn : callable, optional
        Function taking (y_true, y_pred, **kwargs).
    metric_kwargs : dict, optional
        Extra keyword arguments for metric_fn.

    Returns
    -------
    metric_result : output of metric_fn (or None)
    y_hat : array-like
        Output of model.predict().
    """
    non_standardized_features = non_standardized_features or []
    metric_kwargs = metric_kwargs or {}

    # ---- Parse feature groups ----
    group_features = set()
    for g in feature_groups_df["feature_group"]:
        group_features.update(ast.literal_eval(g))

    all_features = set(X.columns)

    # ---- Feature categories ----
    standardizable_features = list(all_features - set(non_standardized_features))
    non_group_standardized = list(
        set(standardizable_features) - group_features
    )

    X_scaled = X.copy()
    X_scaled[standardizable_features] = scaler.transform(X[standardizable_features])
    X_parts = []

    # ---- Pass-through features ----
    if non_group_standardized:
        X_parts.append(X_scaled[non_group_standardized])

    if non_standardized_features:
        X_parts.append(X[non_standardized_features])

    # ---- PCA per group ----
    _, X_parts, _ = extract_pcas_from_groups(
        feature_groups_df, X_scaled, X_parts, _pcas_=pcas
    )

    # ---- Final matrices ----
    X_final = pd.concat(X_parts, axis=1)
    
    if features_not_to_be_used: # For selected features
        X_final.drop(columns=features_not_to_be_used, inplace=True)

    # ---- Evaluate ----
    y_pred, y_score = get_preds_and_scores(model, X_final)

    metric_result = None
    if metric_fn is not None:
        metric_result = metric_fn(y, y_pred, y_score, **metric_kwargs)

    artifacts = {
        "y_pred": y_pred,
        "y_score": y_score,
        "feature_names": X_final.columns.tolist(),
    }

    return model, metric_result, artifacts
    

    
