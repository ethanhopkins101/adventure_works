import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def time_series_split(df, days=30):
    """Helper to split the last 30 days for testing."""
    if df.empty: return pd.DataFrame(), pd.DataFrame()
    cutoff = df['OrderDate'].max() - pd.Timedelta(days=days)
    train = df[df['OrderDate'] <= cutoff]
    test = df[df['OrderDate'] > cutoff]
    return train, test

def generate_features(df_raw, m_date, df_subs):
    """
    Accepts raw data from orchestrator and routes items to models.
    """
    try:
        # --- 1. DATA DENSIFICATION ---
        # Group by Day/Item and sum quantities
        df_daily = df_raw.groupby(['OrderDate', 'SubcategoryName']).agg({
            'OrderQuantity': 'sum',
            'StockDate': 'min'
        }).reset_index()

        # Create the Grid (Every Day x Every Item)
        all_days = pd.date_range(start=df_daily['OrderDate'].min(), 
                                 end=df_daily['OrderDate'].max(), freq='D')
        all_items = df_subs['SubcategoryName'].unique()
        grid = pd.MultiIndex.from_product([all_days, all_items], names=['OrderDate', 'SubcategoryName'])
        df_grid = pd.DataFrame(index=grid).reset_index()

        # Merge and Impute
        df_final = pd.merge(df_grid, df_daily, on=['OrderDate', 'SubcategoryName'], how='left')
        df_final['OrderQuantity'] = df_final['OrderQuantity'].fillna(0).astype(int)
        df_final['StockDate'] = df_final['StockDate'].fillna(m_date)

        # --- 2. ROUTING LOGIC ---
        route_metrics = df_final.groupby('SubcategoryName').agg(
            TotalSales=('OrderQuantity', 'sum'),
            ZeroDays=('OrderQuantity', lambda x: (x == 0).sum()),
            TotalDays=('OrderQuantity', 'count'),
            FirstStock=('StockDate', 'min'),
            LastDate=('OrderDate', 'max')
        ).reset_index()

        route_metrics['DaysSinceStocked'] = (route_metrics['LastDate'] - route_metrics['FirstStock']).dt.days + 1
        route_metrics['DailyVelocity'] = route_metrics['TotalSales'] / route_metrics['DaysSinceStocked']
        route_metrics['ZeroRatio'] = route_metrics['ZeroDays'] / route_metrics['TotalDays']
        
        # Threshold for high-velocity items
        v_thresh = route_metrics['DailyVelocity'].quantile(0.95) 

        def determine_route(row):
            if row['DailyVelocity'] >= v_thresh: return 'AutoARIMA'
            if row['ZeroRatio'] < 0.20: return 'AutoARIMA'
            elif row['ZeroRatio'] < 0.60: return 'Prophet'
            else: return 'ColdStart'

        route_map = route_metrics.set_index('SubcategoryName').apply(determine_route, axis=1).to_dict()
        df_final['TargetModel'] = df_final['SubcategoryName'].map(route_map)

        # --- 3. BUNDLING ---
        arima_full = df_final[df_final['TargetModel'] == 'AutoARIMA'].copy()
        prophet_full = df_final[df_final['TargetModel'] == 'Prophet'].copy()
        cold_start_full = df_final[df_final['TargetModel'] == 'ColdStart'].copy()

        arima_train, arima_test = time_series_split(arima_full)
        prophet_train, prophet_test = time_series_split(prophet_full)
        
        bundles = {
            'arima': (arima_train, arima_test),
            'prophet': (prophet_train, prophet_test),
            'cold_start': cold_start_full
        }

        logging.info(f"Features Ready. Stats: ARIMA({arima_full['SubcategoryName'].nunique()}), "
                     f"Prophet({prophet_full['SubcategoryName'].nunique()}), "
                     f"ColdStart({cold_start_full['SubcategoryName'].nunique()})")

        # IMPORTANT: Return df_final as first element for predict.py to use max_date
        return df_final, bundles

    except Exception as e:
        logging.error(f"Feature engineering failed: {e}")
        raise

if __name__ == "__main__":
    # bundles = generate_features()
    # print("\n--- Pipeline Verification ---")
    # for key, val in bundles.items():
    #     if isinstance(val, tuple):
    #         print(f"{key.upper()} -> Train: {val[0].shape}, Test: {val[1].shape}")
    #     else:
    #         print(f"{key.upper()} -> Full: {val.shape}")
    pass