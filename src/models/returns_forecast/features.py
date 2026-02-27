import pandas as pd
import numpy as np
import sys
import os
from itertools import product

# Add src/data to path to import the encoder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data')))
from encoder import sync_subcategory_encoding

import pandas as pd
from itertools import product

def create_daily_skeleton(df, subcategories_df):
    """
    Creates a full daily date range for every subcategory in the master list.
    Fills missing entries with 0 for quantities.
    
    Args:
        df (pd.DataFrame): The merged sales/returns data.
        subcategories_df (pd.DataFrame): The master list of 37 subcategories.
    """
    # 1. Convert to Datetime
    df['ReturnDate'] = pd.to_datetime(df['ReturnDate'])
    if 'OrderDate' not in df.columns:
        df['OrderDate'] = df['ReturnDate']
    else:
        df['OrderDate'] = pd.to_datetime(df['OrderDate']).fillna(df['ReturnDate'])

    # 2. Setup Dates and Master Categories
    all_dates = pd.date_range(df['ReturnDate'].min(), df['ReturnDate'].max(), freq='D')
    
    # SCIENTIFICALLY OBJECTIVE: Use the master list to define the categories
    # This ensures 37/37 items exist even if they have 0 returns/sales history.
    master_subcats = subcategories_df['SubcategoryName'].unique()

    # 3. Create Skeleton using the Master List
    skeleton = pd.DataFrame(
        list(product(all_dates, master_subcats)), 
        columns=['ReturnDate', 'SubcategoryName']
    )

    # 4. Aggregate original data to avoid duplicates before merging
    df_daily = df.groupby(['ReturnDate', 'SubcategoryName']).agg({
        'ReturnQuantity': 'sum',
        'OrderQuantity': 'sum',
        'OrderDate': 'first'
    }).reset_index()

    # 5. Merge and Fill Gaps
    # Using 'left' on the skeleton ensures we keep all 37 items for every date
    final_df = skeleton.merge(df_daily, on=['ReturnDate', 'SubcategoryName'], how='left')
    
    # Fill quantities with 0 if they don't exist
    final_df['ReturnQuantity'] = final_df['ReturnQuantity'].fillna(0).astype(int)
    final_df['OrderQuantity'] = final_df['OrderQuantity'].fillna(0).astype(int)
    
    # Fill missing OrderDates with the ReturnDate of that row
    final_df['OrderDate'] = final_df['OrderDate'].fillna(final_df['ReturnDate'])

    return final_df.sort_values(by=['ReturnDate', 'SubcategoryName'])

def apply_persistent_encoding(df):
    """
    Calls the master encoder script to assign/retrieve unique IDs.
    """
    df_encoded, mapping = sync_subcategory_encoding(df)
    return df_encoded

def route_by_sparsity(df):
    """
    Statistically analyzes subcategories to route to ARIMA, Prophet, or ColdStart.
    Adjusted for Return patterns to ensure all 37 items are retained.
    """
    # 1. Routing Metrics
    metrics = df.groupby('SubcategoryName').agg(
        TotalReturns=('ReturnQuantity', 'sum'),
        ZeroDays=('ReturnQuantity', lambda x: (x == 0).sum()),
        TotalDays=('ReturnQuantity', 'count'),
        MaxSingleDay=('ReturnQuantity', 'max'),
        FirstDate=('ReturnDate', 'min'),
        LastDate=('ReturnDate', 'max')
    ).reset_index()

    metrics['ZeroRatio'] = metrics['ZeroDays'] / metrics['TotalDays']
    
    # Trend/Spiky Check: Returns are volatile. 
    # If one day is > 70% of all returns, it's a 'Spike', not a pattern.
    metrics['IsSpiky'] = (metrics['MaxSingleDay'] / metrics['TotalReturns'].replace(0, 1)) > 0.70

    def determine_route(row):
        # Priority 1: No data or extremely spiky -> ColdStart
        if row['TotalReturns'] == 0 or row['IsSpiky'] or row['ZeroRatio'] > 0.85:
            return 'ColdStart'
        
        # Priority 2: Consistent signal -> AutoARIMA
        elif row['ZeroRatio'] <= 0.40:
            return 'AutoARIMA'
        
        # Priority 3: Moderate signal -> Prophet
        else:
            return 'Prophet'

    # 2. Map back to main DF
    route_map = metrics.set_index('SubcategoryName').apply(determine_route, axis=1).to_dict()
    
    df = df.copy()
    df['TargetModel'] = df['SubcategoryName'].map(route_map)

    # 3. Create Bundles
    # We return the full DF as the first argument to ensure execute.py has all 37 items
    bundles = {
        'arima': df[df['TargetModel'] == 'AutoARIMA'].copy(),
        'prophet': df[df['TargetModel'] == 'Prophet'].copy(),
        'cold_start': df[df['TargetModel'] == 'ColdStart'].copy()
    }

    return df, bundles, route_map

# def time_series_split(df, days=30):
#     """
#     Splits data into train and test sets based on the last N days.
#     """
#     if df.empty: 
#         return pd.DataFrame(), pd.DataFrame()
    
#     cutoff = df['ReturnDate'].max() - pd.Timedelta(days=days)
#     train = df[df['ReturnDate'] <= cutoff].copy()
#     test = df[df['ReturnDate'] > cutoff].copy()
    
#     return train, test

def add_lagged_features(df, lag_days=1):
    """
    Creates a lagged sales feature to represent the delay between 
    an item being sold and it being returned.
    """
    # We must sort by Subcategory and Date to ensure the shift happens within groups
    df = df.sort_values(['SubcategoryName', 'ReturnDate']).copy()
    
    # Scientifically: Today's returns are often driven by yesterday's (or previous) sales volume
    df['OrderQuantity_Lag1'] = df.groupby('SubcategoryName')['OrderQuantity'].shift(lag_days)
    
    # Handle the first row of each group which will now be NaN
    # We fill with the mean of that subcategory to maintain data density
    df['OrderQuantity_Lag1'] = df.groupby('SubcategoryName')['OrderQuantity_Lag1'].transform(
        lambda x: x.fillna(x.mean())
    )
    
    # Ensure it's an integer for consistency
    df['OrderQuantity_Lag1'] = df['OrderQuantity_Lag1'].astype(int)
    
    return df