import pandas as pd
from time import sleep
import ast

def get_results_summary(results_df):
    """Get a summary of CV results"""
    results_summary = (results_df
        .groupby("param")
        .agg(
            # PR-AUC
            tr_pr_auc_mean=("train_pr_auc", "mean"),
            tr_pr_auc_std=("train_pr_auc", "std"),
            te_pr_auc_mean=("test_pr_auc", "mean"),
            te_pr_auc_std=("test_pr_auc", "std"),
            adj_pr_auc_mean=("test_pr_auc_adj", "mean"), # Only for test, train's is equal to original
            adj_pr_auc_std=("test_pr_auc_adj", "std"),

            # ROC-AUC
            tr_roc_auc_mean=("train_roc_auc", "mean"),
            tr_roc_auc_std=("train_roc_auc", "std"),
            te_roc_auc_mean=("test_roc_auc", "mean"),
            te_roc_auc_std=("test_roc_auc", "std"),
            
            # Recall
            tr_recall_mean=("train_recall", "mean"),
            tr_recall_std=("train_recall", "std"),
            te_recall_mean=("test_recall", "mean"),
            te_recall_std=("test_recall", "std"),
            
            # Precision
            tr_precision_mean=("train_precision", "mean"),
            tr_precision_std=("train_precision", "std"),
            te_precision_mean=("test_precision", "mean"),
            te_precision_std=("test_precision", "std"),
            
            # Time
            time_mean=("time", "mean"),
            time_std=("time", "std")
        )
        .sort_values("te_pr_auc_mean", ascending=False)
        .reset_index()
        ).round(4)
    return results_summary

def expand_params(df, exclude_list=[], param_col='param'):
    """
    Unpacks a column of dictionaries into separate columns, 
    filtering out specified keys.
    """
    expanded_df = df[param_col].apply(ast.literal_eval).apply(pd.Series)

    if exclude_list:
        cols_to_keep = [c for c in expanded_df.columns if c not in exclude_list] 
    else:
        cols_to_keep = expanded_df.columns
    
    expanded_df = expanded_df[cols_to_keep]
    expanded_df.rename(columns=(lambda c: 'params_' + c), inplace=True) # Add 'params_' prefix

    result_df = pd.concat([expanded_df, df.drop(columns=[param_col])], axis=1)
    return result_df

# Result Extractor and Loader Functions
def get_cv_results(dir):
    "Loads CV results of an rolling origin folds for all models."
    cv_results = {}
    for model in ['logreg', 'adaboost', 'gradboost', 'svc', 'xgb']:
        cv_results[model] = pd.read_csv(f"{dir}/{model}_cv_results.csv")
    
    return cv_results

def remove_np_float64(x): # For XGBoost's adjusting 'scale_pos_weight' param
    splits = x.split(',')
    if 'np.float64' in splits[-2]:
        splits[-2] = "'scale_pos_weight': True" 
    else :
        splits[-2] = "'scale_pos_weight': False" 
    return ','.join(splits)


def extract_summaries_and_save(cv_results, dir):
    "Extracts a summary of CV results and saves them."
    summaries = {}

    for model in cv_results.keys():
        if model == 'svc': 
            for kernel in ['linear', 'rbf', 'poly']: # We want to analyze each kernel as a different algorithm.
                summaries[f'svc_{kernel}'] = get_results_summary(cv_results[model].loc[cv_results[model]['model'] == f'SVC_{kernel}', :])
        else: # Other algorithms
            summaries[model] = get_results_summary(cv_results[model])
        
        # For handling model spesific issues
        if model == 'logreg':
            summaries[model] = expand_params(summaries[model], exclude_list=['penalty', 'solver', 'random_state'])

        elif model == 'adaboost': 
            summaries[model].loc[:, 'param'] = summaries[model].loc[:, 'param'].apply(lambda x: str(x) 
                                        .replace("'estimator': DecisionTreeClassifier(max_depth=", "'max_depth': ")
                                        .replace(', random_state=4)', ''))
            summaries[model].loc[0:1, 'param']
            summaries[model] = expand_params(summaries[model], exclude_list=['random_state'])
        
        elif model == 'gradboost':
            summaries[model] = expand_params(summaries[model], exclude_list=['random_state'])
        
        elif model == 'svc':
            for kernel in ['linear', 'rbf', 'poly']:
                summaries[f'svc_{kernel}'] = expand_params(summaries[f'svc_{kernel}'], exclude_list=['kernel', 'random_state'])

        elif model == 'xgb':
            param_n_estimators = cv_results[model].groupby('param').agg(
                param_n_estimators_mean=("n_estimators", "mean"),
                param_n_estimators_std=("n_estimators", "std"))
            summaries[model] = summaries[model].merge(param_n_estimators, how='left', on='param')
            summaries[model].loc[:, 'param'] = summaries[model].loc[:, 'param'].apply(remove_np_float64)
            summaries[model] = expand_params(summaries[model], 
                                             exclude_list=['early_stopping_rounds', 'n_estimators', 
                                                           'seed', 'colsample_bytree', 
                                                           'colsample_bylevel', 'nthread'])
    
            cols = []
            cols.extend((summaries[model].columns[:5].tolist() 
                        +['param_n_estimators_mean', 'param_n_estimators_std'] + summaries[model].columns[5:-2].tolist()))
            summaries[model] = summaries[model][cols]

    for model in summaries.keys():
        summaries[model].to_csv(f"{dir}/{model}_summary.csv", index=False)
    
    return summaries


def get_summaries(dir):
    "Loads summary of CV results for a given end year."
    summaries = {}
    for model in ['logreg', 'adaboost', 'gradboost', 'svc_linear', 'svc_rbf', 'svc_poly', 'xgb']:
        summaries[model] = pd.read_csv(f"{dir}/{model}_summary.csv")
    return summaries