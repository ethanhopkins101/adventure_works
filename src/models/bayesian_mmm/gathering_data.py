import pandas as pd
import numpy as np
import os

def gather_mmm_data():
    """
    Generates synthetic Marketing Mix Modeling data with Profit as the target.
    This represents the raw data collection phase of the pipeline.
    """
    # 1. Setup Timeframe (47 weeks)
    dates = pd.date_range(start="2016-08-08", periods=47, freq='W-MON')

    # 2. Media Spending Generator
    np.random.seed(42) 
    
    def get_pulsed_spend(mean_val, max_val, probability=0.3):
        pulses = np.random.choice([0, 1], size=47, p=[1-probability, probability])
        raw_spend = pulses * np.random.uniform(low=mean_val*0.5, high=max_val, size=47)
        # Scale to ensure the average spend matches our desired mean_val
        return raw_spend * (mean_val / raw_spend.mean())

    # Generate Channel Spends
    tv_s = get_pulsed_spend(57, 612, probability=0.25)
    ooh_s = get_pulsed_spend(42, 484, probability=0.20)
    print_s = get_pulsed_spend(14, 123, probability=0.15)
    facebook_s = get_pulsed_spend(33, 238, probability=0.60) 

    # Search Spend: Trend + Seasonality
    search_s_base = np.linspace(15, 25, 47) 
    search_s_seasonal = 3 * np.sin(np.arange(47) * (2 * np.pi / 52))
    search_s = search_s_base + search_s_seasonal + np.random.normal(0, 1, 47)
    search_s = np.clip(search_s, 0, 70)
    search_s = search_s * (23 / search_s.mean()) 

    # 3. Target Variable: PROFIT (Net Earnings)
    # We simulate Profit based on organic base, seasonality, and ad-driven margins
    time_index = np.arange(47)
    profit_base = 800  # Base profit from organic sales
    profit_seasonal = 180 * np.sin((time_index - 10) * (2 * np.pi / 52))
    profit_trend = np.linspace(0, 80, 47)
    
    # Media Impact Coefficients (Reflecting Profit ROI)
    # Search (2.2) > FB (1.1) because Search captures high-margin intent
    media_impact = (tv_s * 0.45) + (facebook_s * 1.1) + (search_s * 2.2)

    profit = profit_base + profit_seasonal + profit_trend + media_impact + np.random.normal(0, 40, 47)

    # 4. Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'profit': profit,
        'tv_s': tv_s,
        'ooh_s': ooh_s,
        'print_s': print_s,
        'facebook_i': facebook_s * 1000, 
        'facebook_s': facebook_s,
        'search_clicks_p': search_s * 3, 
        'search_s': search_s,
        'competitor_sales_b': 4100 + np.random.normal(0, 100, 47)
    })

    return df
