import os
import logging
import pandas as pd
from datetime import timedelta

# Import custom modules
from data_gathering import gather_sales_data
from features import create_daily_skeleton, route_and_split, apply_persistent_encoding
from train import train_arima_models, train_prophet_models
from predictions import run_sales_prediction, generate_stocking_report, generate_staffing_heatmap

# Path Configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(script_dir, "../../../models/sales_forecast/"))
JSON_DIR = os.path.abspath(os.path.join(script_dir, "../../../json_files/sales_forecast/"))
FORECAST_JSON = os.path.join(JSON_DIR, "latest_sales_forecast.json")

# Ensure directories exist before execution
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_pipeline():
    # 1. DATA GATHERING
    logging.info("Step 1: Gathering Raw Data...")
    df_raw, m_date, df_subs = gather_sales_data()

    # 2. DATA WINDOW LOGIC (1-Year Freshness Filter)
    # Scientifically objective: Avoid learning from obsolete market trends
    max_date = df_raw['OrderDate'].max()
    min_date_allowed = max_date - timedelta(days=365)
    
    logging.info(f"Applying 1-Year Window: Filtering data prior to {min_date_allowed.date()}")
    df_raw = df_raw[df_raw['OrderDate'] >= min_date_allowed].copy()

    # 3. DATA CLEANING (Skeleton Creation)
    # We do this first so string matches work correctly between Sales and Subcategory references
    logging.info("Step 2: Creating Daily Skeleton (Densification)...")
    df_skeleton = create_daily_skeleton(df_raw, m_date, df_subs)
    
    # 4. ENCODING & REPLACEMENT
    # Converts 'Mountain Bikes' -> 1 directly in the SubcategoryName column
    logging.info("Step 3: Applying Persistent Encoding (Replacing Names with IDs)...")
    df_final_numeric = apply_persistent_encoding(df_skeleton)
    
    # 5. CONDITIONAL EXECUTION (Train vs. Predict)
    existing_models = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl')]
    
    if not existing_models:
        logging.info("--- NO MODELS FOUND: INITIATING FULL TRAINING PIPE ---")
        # Route the numerical data
        df_routed, bundles = route_and_split(df_final_numeric)
        
        # Training (Files will be saved as {ID}.pkl)
        train_arima_models(bundles['arima'][0], bundles['arima'][1])
        train_prophet_models(bundles['prophet'][0], bundles['prophet'][1])
        logging.info("Training complete. Models saved with Numerical IDs.")
    else:
        logging.info(f"--- MODELS DETECTED ({len(existing_models)}): SKIPPING TO PREDICTION ---")

    # 6. PREDICTION & REPORTING
    # All downstream functions now receive numerical IDs
    logging.info("Step 4: Running Rolling Forecasts...")
    run_sales_prediction(df_final_numeric, horizon=30)
    
    logging.info("Step 5: Generating Stocking Report...")
    generate_stocking_report(FORECAST_JSON, horizon=30)
    
    logging.info("Step 6: Generating Staffing Heatmap...")
    generate_staffing_heatmap(df_final_numeric, FORECAST_JSON, horizon=30)

    logging.info("PIPELINE EXECUTION COMPLETE.")

if __name__ == "__main__":
    run_pipeline()


# Crucial functionality to add: a script next to main.py that is triggered weekly by github actions
# instead of main.py, call it retrain.py, it does the following, it moves the pretrained models
# to a subfolder within their exact path, then runs main.py this forces each pipeline to retrain
# since they are configured to retrain if the models aren't in that locations.