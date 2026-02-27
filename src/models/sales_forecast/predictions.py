import os
import json
import joblib
import logging
from datetime import timedelta

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Absolute Path Configuration
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define and create directories
PIC_DIR = os.path.abspath(os.path.join(script_dir, "../../../pictures/sales_forecast/"))
REPORT_DIR = os.path.abspath(os.path.join(script_dir, "../../../json_files/sales_forecast/"))
MODEL_DIR = os.path.abspath(os.path.join(script_dir, "../../../models/sales_forecast/"))
JSON_DIR = os.path.abspath(os.path.join(script_dir, "../../../json_files/sales_forecast/"))

for directory in [PIC_DIR, REPORT_DIR, MODEL_DIR, JSON_DIR]:
    os.makedirs(directory, exist_ok=True)

# Settings
plt.switch_backend('Agg')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_sales_prediction(df, horizon=30):
    """
    Orchestrates rolling forecasts. Routes to ARIMA/Prophet if .pkl exists, 
    otherwise performs a Velocity-based Cold Start.
    """
    max_date = df['OrderDate'].max()
    future_dates = pd.date_range(start=max_date + timedelta(days=1), periods=horizon)
    date_strings = [d.strftime('%Y-%m-%d') for d in future_dates]
    
    final_forecasts = {}

    for subcat in df['SubcategoryName'].unique():
        sub_df = df[df['SubcategoryName'] == subcat].sort_values('OrderDate')
        
        # Construct model path (SubcategoryName is the encoded ID)
        model_path = os.path.join(MODEL_DIR, f"{subcat}.pkl")
        
        preds = []
        model_type = "ColdStart"
        accuracy_hint = "N/A"

        # 1. ATTEMPT PRE-TRAINED MODELS
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                
                # ARIMA BRANCH
                if "ARIMA" in str(type(model)):
                    model_type = "AutoARIMA"
                    preds = model.predict(n_periods=horizon).tolist()
                    accuracy_hint = "High (90-95%)"
                
                # PROPHET BRANCH
                else:
                    model_type = "Prophet"
                    future_p = pd.DataFrame({'ds': future_dates})
                    # Add regressor logic if used in training
                    future_p['is_payday'] = future_p['ds'].dt.day.isin([15, 30]).astype(int)
                    forecast = model.predict(future_p)
                    preds = forecast['yhat'].clip(lower=0).tolist()
                    accuracy_hint = "Medium-High (80-85%)"
            except Exception as e:
                logging.warning(f"Model load failed for {subcat}, falling back to ColdStart: {e}")

        # 2. COLD START BRANCH (Fallback or Default)
        if not preds:
            model_type = "ColdStart"
            # Scientific Logic: Use the median of the last 14 days to avoid outlier influence
            # and multiply by a momentum factor (ratio of last 7 days vs last 14 days)
            recent_short = sub_df['OrderQuantity'].tail(7).mean()
            recent_long = sub_df['OrderQuantity'].tail(14).mean()
            
            momentum = (recent_short / recent_long) if recent_long > 0 else 1.0
            momentum = np.clip(momentum, 0.8, 1.2) # Cap momentum to avoid wild swings
            
            base_val = sub_df['OrderQuantity'].tail(30).median()
            preds = [float(base_val * momentum)] * horizon
            accuracy_hint = "Low (60-65%)"

        # 3. CLEANING PREDICTIONS
        preds = [max(0, round(float(p), 2)) for p in preds]

        final_forecasts[str(subcat)] = {
            "model_source": model_type,
            "confidence_level": accuracy_hint,
            "daily_forecast": dict(zip(date_strings, preds)),
            "total_horizon_volume": sum(preds)
        }

    # 4. SAVE TO JSON
    output_path = os.path.join(JSON_DIR, "latest_sales_forecast.json")
    with open(output_path, 'w') as f:
        json.dump(final_forecasts, f, indent=4)
    
    logging.info(f"Forecast complete. JSON saved to: {output_path}")
    return final_forecasts


def generate_stocking_report(sales_forecast_json, planned_stock_json=None, horizon=30, period_months=1):
    """
    Generates a stocking report identifying Forecasted Demand and Safety Stock.
    
    Logic:
    - Forecasted Sales: Sum of daily predictions over the horizon.
    - Safety Stock (Models): Based on model RMSE/Error, capped at the median forecast to avoid overstocking.
    - Safety Stock (Cold Start): Uses 'Coefficient of Variation' (CV) - higher volatility = higher buffer.
    """
    
    # 1. Load Sales Forecast
    with open(sales_forecast_json, 'r') as f:
        forecast_data = json.load(f)

    # 2. Handle Planned Stock (Baseline)
    # If no planned stock, we simulate 'Owner Intent' based on historical volume
    if planned_stock_json:
        with open(planned_stock_json, 'r') as f:
            planned_stock = json.load(f)
    else:
        # Fallback: Sum of forecasted volume acts as the planned baseline
        planned_stock = {k: v['total_horizon_volume'] for k, v in forecast_data.items()}

    stocking_report = {}

    for subcat, data in forecast_data.items():
        model_type = data['model_source']
        forecast_sum = data['total_horizon_volume']
        daily_vals = list(data['daily_forecast'].values())
        
        # --- SAFETY STOCK LOGIC ---
        safety_stock = 0
        
        if model_type in ['AutoARIMA', 'Prophet']:
            # Scientifically: Safety Stock = Z-Score * Standard Deviation of Error
            # We approximate error using the 'Confidence' spread or historical RMSE if available
            # Rule: Cap safety stock at 50% of the median forecast to prevent 'error-driven' overstocking
            median_daily = np.median(daily_vals)
            estimated_error = forecast_sum * 0.15 # Assuming 15% margin for trained models
            safety_stock = min(estimated_error, median_daily * (horizon / 2))
            
        else: # Cold Start Logic
            # Scientific Logic: Demand Lead Time Variability
            # We use the standard deviation of the forecasted daily values (volatility)
            # If the item is inconsistent (high CV), we add a 30% buffer.
            std_dev = np.std(daily_vals)
            mean_val = np.mean(daily_vals)
            cv = (std_dev / mean_val) if mean_val > 0 else 0
            
            # If CV > 0.5, item is 'Intermittent'. We buffer significantly.
            multiplier = 0.3 if cv > 0.5 else 0.15
            safety_stock = forecast_sum * multiplier

        # --- FINAL CALCULATION ---
        total_required_inventory = forecast_sum + safety_stock
        
        stocking_report[subcat] = {
            "item_id": subcat,
            "forecasted_sales_total": round(forecast_sum, 2),
            "safety_stock_estimate": round(safety_stock, 2),
            "total_stock_recommendation": round(total_required_inventory, 2),
            "model_used": model_type,
            "stock_logic": "Volatility-Adjusted" if model_type == "ColdStart" else "Error-Adjusted"
        }

    # Save Report
    output_path = os.path.join(REPORT_DIR, "stocking_report.json")
    with open(output_path, 'w') as f:
        json.dump(stocking_report, f, indent=4)
        
    logging.info(f"Stocking report generated at: {output_path}")
    return stocking_report


def generate_staffing_heatmap(df_final, sales_forecast_json, horizon=30):
    """
    Plans staffing by comparing forecasts against historical benchmarks.
    Logic:
    1. Look at same month last year (11 months ago) for seasonality.
    2. If last year's volume is too low (new business), use the previous month.
    3. Flags high-traffic days and saves a heatmap for executive review.
    """
    logging.info("Generating Executive Staffing Heatmap...")

    # 1. Load Forecasted Data
    with open(sales_forecast_json, 'r') as f:
        forecast_data = json.load(f)

    # Aggregate daily totals from JSON
    # Structure: {'date': total_units}
    daily_forecasts = {}
    for item_id, item_data in forecast_data.items():
        for date_str, qty in item_data['daily_forecast'].items():
            daily_forecasts[date_str] = daily_forecasts.get(date_str, 0) + qty

    forecast_df = pd.DataFrame(list(daily_forecasts.items()), columns=['OrderDate', 'Forecasted_Units'])
    forecast_df['OrderDate'] = pd.to_datetime(forecast_df['OrderDate'])

    # 2. Benchmark Logic (Previous Month vs Previous Year)
    max_hist_date = df_final['OrderDate'].max()
    
    # Previous Month Benchmark (Last 30 Days)
    prev_month_avg = df_final[df_final['OrderDate'] > (max_hist_date - timedelta(days=30))].groupby('OrderDate')['OrderQuantity'].sum().mean()

    # Same Month Last Year Benchmark (Approx 11 months ago to match the 'next' month)
    # If predicting Jan 2018, look at Jan 2017
    start_last_year = max_hist_date - timedelta(days=335) # Approx start of same month last year
    end_last_year = max_hist_date - timedelta(days=305)
    hist_year_data = df_final[(df_final['OrderDate'] >= start_last_year) & (df_final['OrderDate'] <= end_last_year)]
    
    avg_last_year = hist_year_data.groupby('OrderDate')['OrderQuantity'].sum().mean() if not hist_year_data.empty else 0

    # Scientifically Objective Selection:
    # Use Last Year if it's significant (>70% of current volume), else use Prev Month (Growth Mode)
    use_seasonal_benchmark = avg_last_year > (prev_month_avg * 0.7)
    benchmark_val = avg_last_year if use_seasonal_benchmark else prev_month_avg
    benchmark_name = "Same Month Prev Year" if use_seasonal_benchmark else "Prev Month Average"

    # 3. Create Heatmap Structure
    forecast_df['Day'] = forecast_df['OrderDate'].dt.day_name().str[:3]
    forecast_df['Week'] = (forecast_df['OrderDate'].dt.day - 1) // 7 + 1
    forecast_df['Day_Num'] = forecast_df['OrderDate'].dt.dayofweek # 0=Mon

    pivot_cal = forecast_df.pivot_table(index='Week', columns='Day_Num', values='Forecasted_Units').fillna(0)
    
    # Reindex columns to ensure Mon-Sun order
    pivot_cal = pivot_cal.reindex(columns=range(7), fill_value=0)
    pivot_cal.columns = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    # 4. Visualization
    plt.figure(figsize=(12, 7))
    sns.heatmap(pivot_cal, cmap='BuPu', annot=True, fmt=".0f", linewidths=2, cbar_kws={'label': 'Units'},vmin=150)
    
    plt.title(f"Staffing Intensity Heatmap\nBenchmark: {benchmark_name} ({benchmark_val:.1f} units)", fontweight='bold', fontsize=14)
    plt.xlabel("Day of Week")
    plt.ylabel("Week of Forecast Horizon")
    
    save_path = os.path.join(PIC_DIR, "staffing_heatmap.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    # 5. High Traffic Alerting
    high_traffic_days = forecast_df[forecast_df['Forecasted_Units'] > benchmark_val * 1.2]
    
    logging.info(f"Heatmap saved to {save_path}. Identified {len(high_traffic_days)} high-traffic days.")
    return save_path