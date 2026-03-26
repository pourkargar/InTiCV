import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from darts.timeseries import concatenate


def pad_series_pair(target_series, covariate_series, required_length):
    current_length = len(target_series)
    if current_length < required_length:
        pad_length = required_length - current_length
        
        last_target = target_series.slice_n_points_before(current_length, n=1)
        last_covariate = covariate_series.slice_n_points_before(current_length, n=1)
        
        target_padding = concatenate([last_target] * pad_length, ignore_time_axis=True)
        covariate_padding = concatenate([last_covariate] * pad_length, ignore_time_axis=True)
        
        padded_target = concatenate([target_series, target_padding], ignore_time_axis=True)
        padded_covariate = concatenate([covariate_series, covariate_padding], ignore_time_axis=True)
        
        return padded_target, padded_covariate
    return target_series, covariate_series


def calc_nrmse(y_true, y_pred):
    """Calculate Normalized RMSE (range-based)."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    data_range = np.max(y_true) - np.min(y_true)
    return rmse / data_range if data_range > 0 else 0


def calc_metrics(y_true, y_pred):
    """Calculate NRMSE, MAE, and Mean Bias."""
    nrmse = calc_nrmse(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mb = np.mean(y_pred - y_true)
    return {'NRMSE_range': nrmse, 'MAE': mae, 'MB': mb}


def plot_timeseries_with_splits(ax, df, actual_col, pred_col, plt_conf):
    """Plot time series with train/test background colors."""
    df = df.reset_index(drop=True)
    
    train_len = len(df[df['split'] == 'train'])
    test_len = len(df[df['split'] == 'test'])
    
    # Background colors
    ax.axvspan(0, train_len, facecolor='red', alpha=0.1)
    ax.axvspan(train_len, train_len + test_len, facecolor='green', alpha=0.1)
    
    # Plot train data
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    ax.plot(train_df.index, train_df[actual_col], color='black', linewidth=1.5)
    ax.plot(train_df.index, train_df[pred_col], color='blue', linewidth=1.5)
    
    # Plot test data (offset by train length)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    test_idx = test_df.index + train_len
    ax.plot(test_idx, test_df[actual_col], color='black', linewidth=1.5)
    ax.plot(test_idx, test_df[pred_col], color='blue', linewidth=1.5)
    
    ax.tick_params(labelsize=plt_conf['tick_fontsize'])
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight(plt_conf['tick_fontweight'])


def plot_error_distribution(ax, y_true, y_pred, color='red', plt_conf=None, alpha=0.5):
    """Plot error distribution histogram with KDE curve."""
    import seaborn as sns
    errors = y_pred - y_true
    sns.histplot(errors, ax=ax, color=color, alpha=alpha, kde=True, 
                 edgecolor='black', linewidth=0.5, stat='count')
    
    # Style the KDE line
    for line in ax.lines:
        line.set_color(color)
        line.set_linewidth(2)
    
    if plt_conf:
        ax.tick_params(labelsize=plt_conf['tick_fontsize'])
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight(plt_conf['tick_fontweight'])


def get_metrics_df(df, actual_cols, pred_cols, labels, split_name, full_df):
    """Calculate metrics for multiple targets and return as DataFrame."""
    rows = []
    for actual, pred, label in zip(actual_cols, pred_cols, labels):
        y_true = df[actual].values
        y_pred = df[pred].values
        metrics = calc_metrics(y_true, y_pred)
        data_range = f'{full_df[actual].max().round(2)} - {full_df[actual].min().round(2)}'
        rows.append({
            'Model_name': label,
            'MAE': metrics['MAE'],
            'NRMSE_range': metrics['NRMSE_range'],
            'MB': metrics['MB'],
            'Split': split_name,
            'data_range': data_range
        })
    return pd.DataFrame(rows)
