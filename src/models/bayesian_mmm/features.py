import numpy as np
import pandas as pd

def apply_adstock(series, alpha):
    """
    Computes the geometric decay of marketing spend over time.
    x_t = spend_t + (alpha * x_{t-1})
    """
    adstocked_series = np.zeros(len(series))
    for i in range(len(series)):
        if i == 0:
            adstocked_series[i] = series[i]
        else:
            adstocked_series[i] = series[i] + alpha * adstocked_series[i-1]
    return adstocked_series

def process_features(df):
    """
    Adjusts raw media spend for adstock effects and returns a cleaned DataFrame.
    Target: profit
    """
    df_adstocked = df.copy()

    # Apply specific decay rates established in the notebook
    # TV and Print have higher 'memory' (0.7, 0.6) than digital (0.3, 0.2)
    df_adstocked['tv_s'] = apply_adstock(df['tv_s'].values, 0.7)
    df_adstocked['ooh_s'] = apply_adstock(df['ooh_s'].values, 0.5)
    df_adstocked['print_s'] = apply_adstock(df['print_s'].values, 0.6)
    df_adstocked['facebook_s'] = apply_adstock(df['facebook_s'].values, 0.3)
    df_adstocked['search_s'] = apply_adstock(df['search_s'].values, 0.2)

    # Return only the columns needed for modeling to maintain data hygiene
    columns_to_keep = [
        'date', 'profit', 'tv_s', 'ooh_s', 'print_s', 'facebook_s', 'search_s'
    ]
    
    return df_adstocked[columns_to_keep]