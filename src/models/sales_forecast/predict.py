import pandas as pd
import numpy as np
import joblib
import os
import logging
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns


# Setup logging to track pipeline progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_forecast_pipeline(df, horizon_days, model_dir):
    """
    Orchestrated prediction function for main.py.
    
    Args:
        df (pd.DataFrame): The feature-engineered dataframe containing 'OrderDate' and 'SubcategoryName'.
        horizon_days (int): Number of days to forecast into the future.
        model_dir (str): Path to the folder containing .pkl models.
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty. Pipeline halted.")

    # 1. Setup Horizon Dates
    last_date = df['OrderDate'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon_days)
    logging.info(f"Generating forecast for window: {future_dates.min().date()} to {future_dates.max().date()}")

    forecast_results = []
    
    # 2. Identify and Process Saved Models
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory '{model_dir}' does not exist.")

    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
    
    for file in model_files:
        path = os.path.join(model_dir, file)
        
        # Load the model package
        # Assumes package is a dict: {'model': obj, 'rmse': value} or just the model object
        try:
            payload = joblib.load(path)
            if isinstance(payload, dict):
                model = payload.get('model')
                daily_rmse = payload.get('rmse', 1.0) # Default to 1.0 if missing
            else:
                model = payload
                daily_rmse = 1.0 # Fallback RMSE
        except Exception as e:
            logging.error(f"Error loading {file}: {e}")
            continue

        # Parse filename: modeltype_SubcategoryName.pkl
        parts = file.replace('.pkl', '').split('_', 1)
        m_type = parts[0].lower()
        subcat = parts[1].replace('_', ' ')

        # 3. Rolling Forecast Execution
        try:
            if m_type == 'arima':
                # ARIMA predicts n steps forward from its internal state
                preds_raw = model.predict(n_periods=horizon_days)
                preds = np.maximum(preds_raw, 0)
            
            elif m_type == 'prophet':
                # Prophet requires a future dataframe with 'ds' column
                future_df = pd.DataFrame({'ds': future_dates})
                # Check for holiday/payday regressors used during training
                future_df['is_payday'] = future_df['ds'].dt.day.isin([15, 30]).astype(int)
                forecast = model.predict(future_df)
                preds = np.maximum(forecast['yhat'].values, 0)
            
            else:
                continue
        except Exception as e:
            logging.warning(f"Prediction failed for {subcat} using {m_type}: {e}")
            continue

        # 4. Statistical Calculations (Rounding, Bounds, Confidence)
        total_predicted = np.sum(preds)
        
        # Sigma for the horizon: Daily RMSE * sqrt(Horizon)
        horizon_sigma = daily_rmse * np.sqrt(horizon_days)
        
        # Lower Bound: 95% Confidence Interval (2-tailed Z=1.96)
        lower_bound = np.maximum(0, total_predicted - (1.96 * horizon_sigma))
        
        # Upper Bound (Safety Stock): 95% Service Level (1-tailed Z=1.65)
        upper_bound = total_predicted + (1.65 * horizon_sigma)
        
        # Confidence: Ratio of predicted volume to error margin
        variation = (horizon_sigma / total_predicted) if total_predicted > 0 else 1.0
        confidence = max(0, min(100, (1 - variation) * 100))

        forecast_results.append({
            'SubcategoryName': subcat,
            'Model': m_type.upper(),
            'Predicted_Sales': int(np.round(total_predicted)),
            'Lower_Bound_SS': int(np.round(lower_bound)),
            'Upper_Bound_SS': int(np.round(upper_bound)),
            'Confidence_Rating_%': round(confidence, 2)
        })

    # 5. Export Results
    results_df = pd.DataFrame(forecast_results)
    if not results_df.empty:
        output_path = "final_future_forecast.csv"
        results_df.to_csv(output_path, index=False)
        logging.info(f"Pipeline complete. {len(results_df)} items exported to {output_path}")
    
    return results_df

def generate_restock_report(df_final, forecast_df, bundles):
    """
    Compares last month's actuals vs future predictions.
    Assigns Stock_Status (0-5) based on the risk of under-stocking.
    """
    logging.info("Generating Restock Comparison Report...")
    
    # 1. Calculate Last Month Actuals
    max_date = df_final['OrderDate'].max()
    last_month_start = max_date - timedelta(days=30)
    
    actuals_df = df_final[df_final['OrderDate'] > last_month_start].groupby('SubcategoryName')['OrderQuantity'].sum().reset_index()
    actuals_df.columns = ['SubcategoryName', 'Last_Month_Actual']

    # 2. Merge with Future Forecast
    report = pd.merge(forecast_df[['SubcategoryName', 'Predicted_Sales']], actuals_df, on='SubcategoryName', how='left')
    report['Last_Month_Actual'] = report['Last_Month_Actual'].fillna(0).astype(int)
    report.rename(columns={'Predicted_Sales': 'Future_Forecast'}, inplace=True)

    # 3. Stock Status Logic (0 to 5)
    def calculate_status(row):
        actual = row['Last_Month_Actual']
        forecast = row['Future_Forecast']
        
        if forecast <= 0: return 0
        if actual <= 0: return 5 # High risk: No previous sales but forecast exists
        
        ratio = forecast / actual
        
        if ratio <= 1.0: return 0  # No extra restock needed
        if ratio <= 1.2: return 1  # Minimal increase
        if ratio <= 1.5: return 2
        if ratio <= 2.0: return 3
        if ratio <= 3.0: return 4
        return 5                   # High significance: Forecast >> Actual

    report['Stock_Status'] = report.apply(calculate_status, axis=1)

    # 4. Handle Cold Start items (Rules: Status 0)
    cold_start_items = []
    if 'cold_start' in bundles:
        cs_df = bundles['cold_start']
        cold_start_items = cs_df['SubcategoryName'].unique()
    
    # Create entries for Cold Start if not already in report
    for item in cold_start_items:
        if item not in report['SubcategoryName'].values:
            # Calculate actuals for cold start items
            cs_actual = actuals_df[actuals_df['SubcategoryName'] == item]['Last_Month_Actual'].values
            val = cs_actual[0] if len(cs_actual) > 0 else 0
            
            new_row = pd.DataFrame([{
                'SubcategoryName': item,
                'Last_Month_Actual': int(val),
                'Future_Forecast': 0,
                'Stock_Status': 0
            }])
            report = pd.concat([report, new_row], ignore_index=True)
        else:
            # If item is already there but is a cold start item, force 0
            report.loc[report['SubcategoryName'] == item, 'Stock_Status'] = 0

    # Export
    report.to_csv("restock_risk_report.csv", index=False)
    logging.info("Restock report exported: restock_risk_report.csv")
    
    return report

plt.switch_backend('Agg')


def generate_labor_heatmap(df_final, model_dir):
    logging.info("Generating Staffing/Labor Intensity Heatmap...")
    try:
        # 1. Timeline Setup
        last_date = pd.to_datetime(df_final['OrderDate'].max())
        next_month_dates = pd.date_range(start=last_date + timedelta(days=1), periods=31)
        daily_predictions = pd.DataFrame({'OrderDate': next_month_dates, 'Total_Units': 0.0})

        # 2. Calculate Benchmark (Previous Month's Daily Average)
        # Scientifically objective: Compare against the most recent 30 days of actual data
        prev_month_start = last_date - timedelta(days=30)
        recent_actuals = df_final[df_final['OrderDate'] > prev_month_start]
        # Sum total units per day across all subcategories
        daily_actuals = recent_actuals.groupby('OrderDate')['OrderQuantity'].sum()
        avg_prev_month_load = daily_actuals.mean()

        # 3. Load and Aggregate Future Predictions
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
        if not model_files:
            logging.error("No .pkl files found in model_dir!")
            return

        for file in model_files:
            path = os.path.join(model_dir, file)
            payload = joblib.load(path)
            model = payload['model'] if isinstance(payload, dict) and 'model' in payload else payload
            
            if 'arima' in file.lower():
                p = model.predict(n_periods=31)
                vals = p.values if hasattr(p, 'values') else p
                daily_predictions['Total_Units'] += np.maximum(vals, 0)
                
            elif 'prophet' in file.lower():
                future = pd.DataFrame({'ds': next_month_dates})
                future['is_payday'] = future['ds'].dt.day.isin([15, 30]).astype(int)
                forecast = model.predict(future)
                daily_predictions['Total_Units'] += np.maximum(forecast['yhat'].values, 0)

        # 4. Calendar Structure
        calendar_df = daily_predictions.copy()
        calendar_df['Day_of_Month'] = calendar_df['OrderDate'].dt.day
        calendar_df['Day_of_Week_Num'] = calendar_df['OrderDate'].dt.dayofweek
        calendar_df['Week_of_Month'] = (calendar_df['Day_of_Month'] - 1) // 7 + 1

        # 5. Pivot for Heatmap
        pivot_cal = calendar_df.pivot_table(
            index='Week_of_Month', 
            columns='Day_of_Week_Num', 
            values='Total_Units', 
            aggfunc='sum'
        ).fillna(0)

        # 6. Visualization
        days_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        pivot_cal = pivot_cal.reindex(columns=range(7), fill_value=0)
        pivot_cal.columns = days_labels

        plt.figure(figsize=(12, 8))
        # cmap='BuPu' as requested
        sns.heatmap(pivot_cal, cmap='BuPu', annot=True, fmt=".0f", linewidths=1.5, vmin=150,vmax=250)
        plt.gca().invert_yaxis() 
        
        plt.title(f"Staffing Heatmap vs. Prev Month Avg ({avg_prev_month_load:.0f} units)\nTotal Load: {daily_predictions['Total_Units'].sum():.0f} Units", fontweight='bold')
        plt.tight_layout()
        plt.savefig("staffing_heatmap.png")
        plt.close()
        
        # 7. Print ALERTS (Only days 20% higher than PREVIOUS month average)
        high_days = calendar_df[calendar_df['Total_Units'] > avg_prev_month_load * 1.2]
        if not high_days.empty:
            print(f"\n--- STAFFING ALERT: {len(high_days)} Days Exceeding Prev Month Avg by >20% ---")
            print(high_days[['OrderDate', 'Total_Units']].to_string(index=False))

        logging.info(f"Heatmap saved. Forecast Sum: {daily_predictions['Total_Units'].sum():.0f}")

    except Exception as e:
        logging.error(f"Heatmap Generation failed: {e}")
        raise
if __name__ == "__main__":
    # Integration test - main.py will bypass this and call run_forecast_pipeline directly
    print("Predict module ready for orchestration.")