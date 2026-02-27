import pandas as pd
import numpy as np
import logging
import joblib
import os
from pmdarima import auto_arima
from prophet import Prophet

# Absolute Path Configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(script_dir, "../../../models/sales_forecast/"))
os.makedirs(MODEL_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_arima_models(arima_train, arima_test):
    """Trains ARIMA and saves using the numerical ID provided in SubcategoryName."""
    subcategories = arima_train['SubcategoryName'].unique()
    
    for subcat_id in subcategories:
        try:
            # Full data for production training
            full_series = pd.concat([arima_train, arima_test])
            s_train = full_series[full_series['SubcategoryName'] == subcat_id].sort_values('OrderDate').set_index('OrderDate')['OrderQuantity']
            
            # Fit Model
            model = auto_arima(s_train, seasonal=True, m=7, suppress_warnings=True, error_action="ignore", stepwise=True)
            
            # Save using the ID directly (casted to int for clean filenames like 1.pkl)
            model_path = os.path.join(MODEL_DIR, f"{int(float(subcat_id))}.pkl")
            joblib.dump(model, model_path)
            logging.info(f"Saved ARIMA: {model_path}")
            
        except Exception as e:
            logging.error(f"ARIMA failed for ID {subcat_id}: {e}")

def train_prophet_models(prophet_train, prophet_test):
    """Trains Prophet and saves using the numerical ID provided in SubcategoryName."""
    subcategories = prophet_train['SubcategoryName'].unique()

    for subcat_id in subcategories:
        try:
            full_df = pd.concat([prophet_train, prophet_test])
            p_train = full_df[full_df['SubcategoryName'] == subcat_id][['OrderDate', 'OrderQuantity']].rename(columns={'OrderDate': 'ds', 'OrderQuantity': 'y'})

            if p_train['y'].sum() < 5: continue

            model = Prophet(yearly_seasonality=True, weekly_seasonality=True, seasonality_mode='multiplicative')
            p_train['is_payday'] = p_train['ds'].dt.day.isin([15, 30]).astype(int)
            model.add_regressor('is_payday')
            model.fit(p_train)

            # Save using the ID directly
            model_path = os.path.join(MODEL_DIR, f"{int(float(subcat_id))}.pkl")
            joblib.dump(model, model_path)
            logging.info(f"Saved Prophet: {model_path}")
            
        except Exception as e:
            logging.error(f"Prophet failed for ID {subcat_id}: {e}")