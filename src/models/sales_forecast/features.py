import pandas as pd
import numpy as np
import logging
import os
import sys

# Setup for encoder access
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data')))
try:
    from encoder import sync_subcategory_encoding
except ImportError:
    logging.warning("Encoder module not found. apply_persistent_encoding will fail.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def apply_persistent_encoding(df):
    """
    Assigns unique numerical IDs to subcategories and replaces the original names.
    Ensures the 'SubcategoryName' column becomes purely numerical.
    """
    logging.info("Applying persistent encoding and replacing names with IDs...")
    
    # 1. Clean data to prevent sorting errors (float vs str)
    if 'SubcategoryName' in df.columns:
        df['SubcategoryName'] = df['SubcategoryName'].fillna('Unknown').astype(str)
    else:
        logging.error("SubcategoryName column missing during encoding!")
        return df

    # 2. Get the encoded dataframe and the mapping dictionary
    df_encoded, mapping = sync_subcategory_encoding(df)
    
    # 3. REPLACEMENT LOGIC
    # Overwrite the original name column with the numerical IDs
    df_encoded['SubcategoryName'] = df_encoded['SubcategoryEncoded']
    
    # 4. Cleanup: Remove the redundant 'SubcategoryEncoded' column
    df_encoded = df_encoded.drop(columns=['SubcategoryEncoded'])
    
    return df_encoded

def time_series_split(df, days=30):
    """Helper to split the last 30 days for testing."""
    if df.empty: return pd.DataFrame(), pd.DataFrame()
    cutoff = df['OrderDate'].max() - pd.Timedelta(days=days)
    train = df[df['OrderDate'] <= cutoff]
    test = df[df['OrderDate'] > cutoff]
    return train, test

def create_daily_skeleton(df_raw, m_date, df_subs):
    """Ensures every date exists for every subcategory. Fills gaps with zeros."""
    df_daily = df_raw.groupby(['OrderDate', 'SubcategoryName']).agg({
        'OrderQuantity': 'sum',
        'StockDate': 'min'
    }).reset_index()

    all_days = pd.date_range(start=df_daily['OrderDate'].min(), 
                             end=df_daily['OrderDate'].max(), freq='D')
    all_items = df_subs['SubcategoryName'].unique()
    grid = pd.MultiIndex.from_product([all_days, all_items], names=['OrderDate', 'SubcategoryName'])
    df_grid = pd.DataFrame(index=grid).reset_index()

    df_final = pd.merge(df_grid, df_daily, on=['OrderDate', 'SubcategoryName'], how='left')
    df_final['OrderQuantity'] = df_final['OrderQuantity'].fillna(0).astype(int)
    df_final['StockDate'] = df_final['StockDate'].fillna(m_date)
    
    return df_final

def route_and_split(df_final):
    """
    Statistically analyzes subcategories to route to ARIMA, Prophet, or ColdStart.
    Includes 'Trend Check' to avoid misrouting sparse items with high volume.
    """
    # 1. Routing Metrics
    metrics = df_final.groupby('SubcategoryName').agg(
        TotalSales=('OrderQuantity', 'sum'),
        ZeroDays=('OrderQuantity', lambda x: (x == 0).sum()),
        TotalDays=('OrderQuantity', 'count'),
        MaxSingleDay=('OrderQuantity', 'max'), # For Trend Check
        FirstStock=('StockDate', 'min'),
        LastDate=('OrderDate', 'max')
    ).reset_index()

    metrics['ZeroRatio'] = metrics['ZeroDays'] / metrics['TotalDays']
    
    # Trend Check: If a single day accounts for more than 50% of total sales,
    # it is a 'Trendy' spike, not a consistent statistical pattern.
    metrics['IsSpiky'] = (metrics['MaxSingleDay'] / metrics['TotalSales']) > 0.50

    def determine_route(row):
        # 1. Spiky/Trendy items or Very Sparse items -> ColdStart
        if row['IsSpiky'] or row['ZeroRatio'] > 0.60:
            return 'ColdStart'
        
        # 2. Consistent high-volume sales -> AutoARIMA
        elif row['ZeroRatio'] <= 0.35:
            return 'AutoARIMA'
        
        # 3. Moderate sparsity -> Prophet
        else:
            return 'Prophet'

    route_map = metrics.set_index('SubcategoryName').apply(determine_route, axis=1).to_dict()
    df_final['TargetModel'] = df_final['SubcategoryName'].map(route_map)

    # 2. Create Bundles
    bundles = {
        'arima': time_series_split(df_final[df_final['TargetModel'] == 'AutoARIMA']),
        'prophet': time_series_split(df_final[df_final['TargetModel'] == 'Prophet']),
        'cold_start': df_final[df_final['TargetModel'] == 'ColdStart'].copy()
    }

    logging.info(f"Routing Complete. Spiky items identified: {metrics['IsSpiky'].sum()}")
    return df_final, bundles