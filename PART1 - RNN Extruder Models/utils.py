import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def calc_nrmse(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    data_range = y_true.max() - y_true.min()
    return rmse / data_range if data_range > 0 else 0


def calc_metrics(y_true, y_pred):
    nrmse = calc_nrmse(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mb = np.mean(y_pred - y_true)
    return {'NRMSE': nrmse, 'MAE': mae, 'MB': mb}


def plot_timeseries_with_splits(ax, df, actual_col, pred_col, title=''):
    df = df.reset_index(drop=True)
    
    # Get split lengths
    train_len = len(df[df['Split'] == 'train'])
    val_len = len(df[df['Split'] == 'val'])
    test_len = len(df[df['Split'] == 'test'])
    
    # Color backgrounds
    ax.axvspan(0, train_len, facecolor='red', alpha=0.1)
    ax.axvspan(train_len, train_len + val_len, facecolor='blue', alpha=0.1)
    ax.axvspan(train_len + val_len, train_len + val_len + test_len, facecolor='green', alpha=0.1)
    
    # Plot actual and predicted
    ax.plot(df.index, df[pred_col], color='blue', linewidth=2, label='Predicted')
    ax.plot(df.index, df[actual_col], color='black', linewidth=2, label='Actual', linestyle='--')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.tick_params(labelsize=12)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
