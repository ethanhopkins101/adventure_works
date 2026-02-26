import pandas as pd
import numpy as np
import lifetimes as lf
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def run_lifetimes_models(df_core):
    """Fits BG/NBD and Gamma-Gamma to predict future value."""
    bgf = lf.BetaGeoFitter(penalizer_coef=0.1)
    bgf.fit(df_core['frequency'], df_core['recency'], df_core['T'])

    ggf = lf.GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(df_core['frequency'], df_core['monetary_value'])

    df_core['prob_alive'] = bgf.conditional_probability_alive(df_core['frequency'], df_core['recency'], df_core['T'])
    df_core['pred_purchases_90d'] = bgf.conditional_expected_number_of_purchases_up_to_time(90, df_core['frequency'], df_core['recency'], df_core['T'])
    df_core['exp_avg_profit'] = ggf.conditional_expected_average_profit(df_core['frequency'], df_core['monetary_value'])
    
    df_core['CLV_90d'] = ggf.customer_lifetime_value(
        bgf, df_core['frequency'], df_core['recency'], df_core['T'], df_core['monetary_value'],
        time=3, discount_rate=0.01
    )
    return df_core

def run_kmeans_clustering(df_core):
    """Clusters core customers into 4 behavioral segments."""
    features = ['frequency', 'recency', 'monetary_value', 'CLV_90d', 'prob_alive']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_core[features])
    
    kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42, n_init=10)
    df_core['Segment'] = kmeans.fit_predict(scaled_features)
    return df_core

def segment_outliers(df_whales, df_high_freq, df_high_mon):
    """
    Assigns a single Segment ID per customer to avoid duplication.
    Priority: Whales (4) covers all outliers.
    """
    # If a customer is in df_whales, they are already accounted for.
    # We only return df_whales to prevent the 300+ row duplication.
    
    processed_outliers = []
    
    if df_whales is not None and not df_whales.dropna(how='all').empty:
        # We ensure we don't have duplicates inside df_whales itself
        df_whales = df_whales.drop_duplicates(subset=['CustomerKey']).copy()
        df_whales['Segment'] = 4  # All outliers categorized as 'Whales'
        processed_outliers.append(df_whales)
            
    return processed_outliers

def combine_and_impute(df_core, outlier_list):
    """Merges core and unique outliers to maintain original customer count."""
    # Concatenate only the unique sets
    full_df = pd.concat([df_core] + outlier_list, axis=0, ignore_index=True)
    
    # Validation: Ensure no duplicate CustomerKeys were created
    if full_df['CustomerKey'].duplicated().any():
        full_df = full_df.drop_duplicates(subset=['CustomerKey'])

    # Define columns for MICE
    rfm_cols = ['frequency', 'recency', 'T', 'monetary_value']
    target_cols = ['prob_alive', 'pred_purchases_90d', 'exp_avg_profit', 'CLV_90d']
    impute_cols = rfm_cols + target_cols

    # Apply MICE Imputer
    mice_imputer = IterativeImputer(max_iter=10, random_state=42)
    full_df[impute_cols] = mice_imputer.fit_transform(full_df[impute_cols])

    # Final Financial Metrics
    full_df['CLV_at_Risk'] = full_df['CLV_90d'] * 0.10
    full_df['Marketing_Budget_90d'] = full_df['CLV_90d'] * 0.20
    
    return full_df