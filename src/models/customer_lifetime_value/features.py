import pandas as pd
import lifetimes as lf

def create_rfm_features(df):
    """
    Transforms transactional data into RFM format and segments outliers.
    Returns: df_core, df_whales, df_high_freq, df_high_mon
    """
    # 1. Transform to RFM
    df['OrderDate'] = pd.to_datetime(df['OrderDate'], format='mixed')
    df_rfm = lf.utils.summary_data_from_transaction_data(
        df,
        customer_id_col     = 'CustomerKey',
        datetime_col        = 'OrderDate',
        monetary_value_col  = 'Profit'
    ).reset_index()

    # 2. Filter: Remove "One-Hit Wonders" (Frequency must be > 0 for Gamma-Gamma)
    df_rfm = df_rfm[df_rfm['frequency'] > 0]

    # 3. Identify Outlier Segments
    # High Frequentists: Customers buying significantly more often than the mean
    df_high_freq = df_rfm[df_rfm['frequency'] > 5].copy().reset_index(drop=True)
    
    # High Monetarists: Customers with exceptionally high average profit per transaction
    df_high_mon = df_rfm[df_rfm['monetary_value'] > 1000].copy().reset_index(drop=True)

    # 4. Create Whales (The Union of both outlier groups)
    # We use CustomerKey to ensure we don't duplicate if someone is both
    whale_keys = set(df_high_freq['CustomerKey']).union(set(df_high_mon['CustomerKey']))
    
    if whale_keys:
        df_whales = df_rfm[df_rfm['CustomerKey'].isin(whale_keys)].copy().reset_index(drop=True)
    else:
        # Return a single row of NaNs as requested if empty
        df_whales = pd.DataFrame([[pd.NA] * len(df_rfm.columns)], columns=df_rfm.columns)

    # 5. Create Core Dataframe (Exclude Whales)
    df_core = df_rfm[~df_rfm['CustomerKey'].isin(whale_keys)].copy().reset_index(drop=True)

    # Final reset and safety check for CustomerKey
    return df_core, df_whales, df_high_freq, df_high_mon