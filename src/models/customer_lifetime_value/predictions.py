# combine behavioral + demographic,
import os
import joblib
import pandas as pd
import lifetimes as lf
# This line MUST come before the Imputer import
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer

def run_predictions(df_core, model_path='../../../models/customer_lifetime_value'):
    """
    Loads saved models and applies them to the core dataframe.
    """
    # 1. Load Assets
    bgf = lf.BetaGeoFitter()
    bgf.load_model(os.path.join(model_path, 'bgf_model.pkl'))
    
    ggf = lf.GammaGammaFitter()
    ggf.load_model(os.path.join(model_path, 'ggf_model.pkl'))
    
    scaler = joblib.load(os.path.join(model_path, 'scaler.pkl'))
    kmeans = joblib.load(os.path.join(model_path, 'kmeans_model.pkl'))

    # 2. Probabilistic Predictions
    df_core['prob_alive'] = bgf.conditional_probability_alive(df_core['frequency'], df_core['recency'], df_core['T'])
    df_core['pred_purchases_90d'] = bgf.conditional_expected_number_of_purchases_up_to_time(90, df_core['frequency'], df_core['recency'], df_core['T'])
    df_core['exp_avg_profit'] = ggf.conditional_expected_average_profit(df_core['frequency'], df_core['monetary_value'])
    
    df_core['CLV_90d'] = ggf.customer_lifetime_value(
        bgf, df_core['frequency'], df_core['recency'], df_core['T'], df_core['monetary_value'],
        time=3, discount_rate=0.01
    )

    # 3. K-Means Clustering
    features = ['frequency', 'recency', 'monetary_value']
    scaled_data = scaler.transform(df_core[features])
    df_core['Segment'] = kmeans.predict(scaled_data)
    
    return df_core

def segment_outliers(df_whales, df_high_freq, df_high_mon):
    """
    Categorizes outliers into a single 'Whale' segment (4) to prevent duplication.
    """
    if df_whales is None or df_whales.dropna(how='all').empty:
        return []
    
    # Priority: One row per customer. Segment 4 = Outliers/Whales
    df_outliers = df_whales.drop_duplicates(subset=['CustomerKey']).copy()
    df_outliers['Segment'] = 4 
    
    return [df_outliers]

def combine_and_impute(df_core, outlier_list, customer_demographics):
    """
    Combines all customers, runs MICE imputation for Whales, 
    and merges with demographic data.
    """
    # 1. Concatenate Core and Outliers
    full_df = pd.concat([df_core] + outlier_list, axis=0, ignore_index=True)
    full_df = full_df.drop_duplicates(subset=['CustomerKey'])

    # 2. MICE Imputation for predictive columns
    rfm_cols = ['frequency', 'recency', 'T', 'monetary_value']
    target_cols = ['prob_alive', 'pred_purchases_90d', 'exp_avg_profit', 'CLV_90d']
    impute_cols = rfm_cols + target_cols

    # MICE fills NaNs in Whales by modeling them based on Core patterns
    mice_imputer = IterativeImputer(max_iter=10, random_state=42)
    full_df[impute_cols] = mice_imputer.fit_transform(full_df[impute_cols])

    # 3. Financial Metrics Calculation
    full_df['CLV_at_Risk'] = full_df['CLV_90d'] * 0.10
    full_df['Marketing_Budget_90d'] = full_df['CLV_90d'] * 0.20

    # 4. Merge with Demographics
    # We use a Left Join to ensure we keep all 5,000 customers from our RFM logic
    full_df = full_df.merge(customer_demographics, on='CustomerKey', how='left')
    
    return full_df

def apply_business_logic(full_df):
    """
    Maps numeric segments to business strategies and qualitative descriptions.
    """
    # 1. Define the behavioral mapping dictionary
    segment_mapping = {
        'Segment': [4, 5, 1, 3, 2, 0],
        'segment name': [
            'High-Value Whales', 
            'Operational Whales', 
            'Elite Champions', 
            'Loyal Daily-Drivers', 
            'Slipping High-Spenders', 
            'Potential Growth'
        ],
        'reason (behavioral logic)': [
            'Heuristic: Extreme Monetary outliers. High-ticket, high-equity clients.',
            'Heuristic: Extreme Frequency outliers. High-volume, habitual buyers.',
            'Core Model: High Frequency & High Recency. Most active core customers.',
            'Core Model: High Frequency / Low Monetary Value. Frequent small-basket buyers.',
            'Core Model: High Historical Spend / Low Recency. At-risk premium customers.',
            'Core Model: Low Frequency. Recent first-time buyers.'
        ],
        'strategy': [
            'VIP Treatment: Retention of high-equity assets.',
            'Cross-Sell/Education: Increasing category depth.',
            'Loyalty Protection: Maintain high engagement.',
            'Margin Optimization: Increase basket size.',
            'Win-Back: Urgent re-activation.',
            'Conversion: Secure the 2nd purchase habit.'
        ],
        'plan': [
            'Provide white-glove service and dedicated VIP support. No model needed; value is inherently maximum.',
            'Send targeted emails to educate on business-grade items. Offer discounts on complementary items they are not yet buying.',
            'Permanent loyalty status and early access to new product launches.',
            'Bundle low-margin items with high-margin accessories to optimize profit per order.',
            'High-incentive "We Miss You" offers tailored to their historically preferred luxury categories.',
            'Welcome sequence focused on brand story and a 2nd-purchase discount trigger.'
        ]
    }

    # 2. Convert to DataFrame for merging
    mapping_df = pd.DataFrame(segment_mapping)

    # 3. Merge with full_df on the 'Segment' column
    # This automatically adds the 'segment name', 'reason', 'strategy', and 'plan' cols
    final_output = full_df.merge(mapping_df, on='Segment', how='left')

    return final_output