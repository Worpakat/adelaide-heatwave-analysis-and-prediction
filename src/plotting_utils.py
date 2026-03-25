import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from typing import Literal

x_ax_types = Literal['index', 'datetime_local']

def plot_with_time_range(df, col_names, x_axis:x_ax_types = 'index'):
    """
    Plots a dataframe column with a time range slider.
    """
    x = df.index if x_axis == 'index' else df[x_axis]

    fig = px.line(df, x=x, y=col_names)
    fig.update_xaxes(rangeslider_visible=True,
                     rangeselector=dict(
                         buttons=list([
                             dict(count=1,  # 1 day
                                  label="1d",
                                  step="day",
                                  stepmode="backward"),
                             dict(count=1,  # 1 month
                                  label="1m",
                                  step="month",
                                  stepmode="backward"),
                             dict(count=6,  # 6 months
                                  label="6m",
                                  step="month",
                                  stepmode="backward"),
                             dict(count=1,  # 1 year
                                  label="1y",
                                  step="year",
                                  stepmode="backward"),
                             dict(step="all")
                         ])))
    fig.show(config={
    'toImageButtonOptions': {
        'format': 'png', 
        'filename': 'plot',
        'scale': 3 # Multiply resolution by 3
        }
    })

def plot_time_slice(df,
                    start_date,
                    end_date,
                    columns,
                    subplots=True,
                    datetime_type="utc"  # "utc" or "local"
                    ):
    """
    Plot a time slice of selected columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data.
    start_date : str or datetime
        Start of time window.
    end_date : str or datetime
        End of time window.
    columns : list
        List of column names to plot.
    subplots : bool, default True
        Whether to plot each variable in a separate subplot.
    datetime_type : str, default "utc"
        "utc" -> use index
        "local" -> use 'datetime_local' column
    """
    # Select datetime source
    if datetime_type == "utc":
        data = df.loc[start_date:end_date, columns]
    elif datetime_type == "local":
        data = (
            df
            .set_index("datetime_local")
            .loc[start_date:end_date, columns]
        )
    else:
        raise ValueError("datetime_type must be 'utc' or 'local'")

    n_vars = len(columns)

    if subplots:
        fig, axes = plt.subplots(n_vars, 1, figsize=(10, 2.4*n_vars), sharex=True)

        if n_vars == 1: 
            axes = [axes]

        for i, (ax, col) in enumerate(zip(axes, columns)):
            color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i % len(plt.rcParams['axes.prop_cycle'].by_key()['color'])]
            data[col].plot(ax=ax, color=color)
            ax.set_title(col)
            ax.grid(True)
    else:
        data.plot(figsize=(10, 4))
        plt.grid(True)
        plt.legend(columns)

    plt.tight_layout()
    plt.show()

def plot_components(col_name, components, resid=True):
    """Plot components of a time series decomposition"""
    r= 4 if resid else 3
    fig, ax = plt.subplots(r, 1, figsize=(10, 2*r), sharex=True)
    fig.suptitle(col_name)

    ax[0].plot(components.observed, label='Original', color='black')
    ax[0].legend()
    ax[1].plot(components.trend, label='Trend', color='blue')
    ax[1].legend()
    ax[2].plot(components.seasonal, label='Seasonal', color='orange')
    ax[2].legend()
    
    if resid: # Conditional plot
        ax[3].plot(components.resid, label='Residual', color='green')
        ax[3].legend()
    
    ax[0].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    fig.tight_layout()  
    plt.show()

def plot_winddir_heatmap(cross_tab, title, conditional=True, condition_on='x', figsize=(10, 3.2)):
    """Plots wind direction and other categorical variables as heatmap. 
    Whether to condition on x or y is optional."""
    if conditional:
        if condition_on == 'x':
            cross_tab = cross_tab / cross_tab.sum()
        elif condition_on == 'y':
            denom = cross_tab.sum(axis=1)
            cross_tab[True] = cross_tab[True] / denom
            cross_tab[False] = cross_tab[False] / denom
    else: 
        cross_tab = cross_tab / cross_tab.sum().sum()
    
    # Sorting by compass order
    compass_order = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    cross_tab = cross_tab.reindex(compass_order, fill_value=0)

    plt.figure(figsize=figsize)    
    sns.heatmap(cross_tab.T, annot=True, cmap='Reds')
    plt.title(title)
    plt.show()

def joint_plots_with_severity(data, x_cols, y, hue=None, palette=None, type='scatter'): 
    rows = (len(x_cols)/3).__ceil__()
    fig, ax = plt.subplots(rows, 3, figsize=(15, rows*4))

    for i, c in enumerate(x_cols):
        if type == 'scatter':
            if rows > 1:
                sns.scatterplot(data=data, x=c, y=y, hue=hue, palette=palette, s=10, ax=ax[i//3][i%3])
            else:
                sns.scatterplot(data=data, x=c, y=y, hue=hue, palette=palette, s=10, ax=ax[i%3])
        elif type == 'kde':
            if rows > 1:
                sns.kdeplot(data=data, x=c, y=y, hue=hue, fill=True, palette=palette, ax=ax[i//3][i%3])
            else:
                sns.kdeplot(data=data, x=c, y=y, hue=hue, fill=True, palette=palette, ax=ax[i%3])

    fig.tight_layout()
    plt.show()

def plot_precision_recall_curve(curves):
    plt.figure(figsize=(6, 4))
    for curve in curves:
        recall = curve['recall']
        precision = curve['precision']
        pr_auc = curve['pr_auc']
        plt.plot(recall, precision, label=f'{curve["name"]} (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()

def plot_roc_curve(curves):
    plt.figure(figsize=(6, 4))
    for curve in curves:
        fpr = curve['fpr']
        tpr = curve['tpr']
        roc_auc = curve['roc_auc']
        plt.plot(fpr, tpr, label=f'{curve["name"]} (AUC = {roc_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

def params_prauc_parallel_coordinates(data, color_column):
    fig = px.parallel_coordinates(
        data, 
        color=color_column, 
        range_color=[3, 5])

    # Hide the color scale that is useless in this case
    fig.update_layout(coloraxis_showscale=False)
    fig.show()

def plot_ehf_proba(proba, EHF, title, decision_threshold=0.5): # REMOVE THIS, IMPORTING FROM UTILS
    EHF_85 = np.float64(28.116478278152794) # Severity threshold. 
    # This value has been calculated during derivation of heatwave indices in the notebook 
    # "03_heatwave_definition_and_identification.ipynb".
    
    plt.figure(figsize=(12, 5))
    plt.axhline(0, c='r', linestyle='--', lw=1, label='Heatwave Threshold') # Heatwave threshold
    plt.axhline(EHF_85, c='orange', linestyle='--', lw=1, label=f'Severity Threshold ({EHF_85:.2f})') 
    plt.axvline(decision_threshold, c='black', linestyle='--', lw=1, label=f'Decision Threshold ({decision_threshold:.2f})')
    plt.legend()
    plt.ylim(-70, 140)
    plt.scatter(proba, EHF, s=2)

    plt.title(title)
    plt.ylabel('EHF')
    plt.xlabel('Predicted Probability')
    plt.show()

def probability_severity_distribution(df_train, df_test, title):
    """Plot the probability box plots by severity for train and test sets"""
    colors = {"No-heatwave": "green", "Low-intensity": "yellow", "Severe": "orange", "Extreme": "red"}
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(12, 4)
    sns.boxplot(data=df_train, y='proba', hue='severity', ax=ax[0], palette=colors, gap=0.2, legend='brief')
    sns.boxplot(data=df_test, y='proba', hue='severity', ax=ax[1], palette=colors, gap=0.2, legend='brief')
    plt.suptitle(title)
    ax[0].set_title("Train")
    ax[1].set_title("Test")
    plt.show()