import pandas as pd
import os
import joblib
from pmdarima import auto_arima
from prophet import Prophet
import logging

# Suppress Prophet logging for cleaner training output
logging.getLogger('prophet').setLevel(logging.ERROR)

def get_model_path():
    """Helper to manage absolute path and directory creation."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.abspath(os.path.join(base_dir, '../../../models/returns_forecast/'))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return model_dir

def train_arima_models(df):
    """
    Trains AutoARIMA and saves using ONLY the encoded ID as filename.
    """
    path = get_model_path()
    for sub_id, group in df.groupby('SubcategoryEncoded'):
        group = group.sort_values('ReturnDate')
        try:
            model = auto_arima(
                y=group['ReturnQuantity'],
                X=group[['OrderQuantity_Lag1']],
                seasonal=True, m=7,
                suppress_warnings=True,
                error_action='ignore'
            )
            # Filename is strictly the ID (e.g., 10.pkl)
            filename = f"{int(sub_id)}.pkl"
            joblib.dump(model, os.path.join(path, filename))
        except Exception as e:
            print(f"Failed to train ARIMA for ID {sub_id}: {e}")

def train_prophet_models(df):
    """
    Trains Prophet and saves using ONLY the encoded ID as filename.
    """
    path = get_model_path()
    for sub_id, group in df.groupby('SubcategoryEncoded'):
        try:
            train_p = group.rename(columns={'ReturnDate': 'ds', 'ReturnQuantity': 'y'})
            
            m = Prophet(daily_seasonality=True)
            m.add_regressor('OrderQuantity_Lag1')
            m.fit(train_p[['ds', 'y', 'OrderQuantity_Lag1']])
            
            # Filename is strictly the ID (e.g., 15.pkl)
            filename = f"{int(sub_id)}.pkl"
            joblib.dump(m, os.path.join(path, filename))
        except Exception as e:
            print(f"Failed to train Prophet for ID {sub_id}: {e}")