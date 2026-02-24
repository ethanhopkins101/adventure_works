import pandas as pd
import numpy as np
import logging
import joblib
import os
from pmdarima import auto_arima
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from features import generate_features

# Global Config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
MODEL_DIR = "models/sales_forecast"
os.makedirs(MODEL_DIR, exist_ok=True)

def calculate_smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred) + 1e-10)
    return 100 / len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / denominator)

def train_arima_models(arima_train, arima_test):
    arima_metrics = []
    subcategories = arima_train['SubcategoryName'].unique()
    
    for subcat in subcategories:
        try:
            s_train = arima_train[arima_train['SubcategoryName'] == subcat].sort_values('OrderDate').set_index('OrderDate')['OrderQuantity']
            s_test = arima_test[arima_test['SubcategoryName'] == subcat].sort_values('OrderDate').set_index('OrderDate')['OrderQuantity']
            
            if s_test.empty: continue

            # Fit Model
            model = auto_arima(s_train, seasonal=True, m=7, suppress_warnings=True, error_action="ignore", stepwise=True)
            
            # --- SAVE STEP ---
            filename = f"arima_{subcat.replace(' ', '_')}.pkl"
            joblib.dump(model, os.path.join(MODEL_DIR, filename))
            
            # Predict & Evaluate
            preds = model.predict(n_periods=len(s_test))
            rmse = np.sqrt(mean_squared_error(s_test, preds))
            smape = calculate_smape(s_test.values, preds.values)
            
            arima_metrics.append({'Subcategory': subcat, 'RMSE': round(rmse, 4), 'sMAPE': round(smape, 2)})
            logging.info(f"Saved ARIMA: {subcat}")
            
        except Exception as e:
            logging.error(f"ARIMA failed for {subcat}: {e}")

    return pd.DataFrame(arima_metrics)

def train_prophet_models(prophet_train, prophet_test):
    prophet_metrics = []
    subcategories = prophet_train['SubcategoryName'].unique()

    for subcat in subcategories:
        try:
            p_train = prophet_train[prophet_train['SubcategoryName'] == subcat][['OrderDate', 'OrderQuantity']].rename(columns={'OrderDate': 'ds', 'OrderQuantity': 'y'})
            p_test = prophet_test[prophet_test['SubcategoryName'] == subcat][['OrderDate', 'OrderQuantity']].rename(columns={'OrderDate': 'ds', 'OrderQuantity': 'y'})

            if p_train['y'].sum() < 10: continue

            # Initialize and Fit
            model = Prophet(yearly_seasonality=True, weekly_seasonality=True, seasonality_mode='multiplicative', changepoint_prior_scale=0.05)
            model.add_country_holidays(country_name='US')
            p_train['is_payday'] = p_train['ds'].dt.day.isin([15, 30]).astype(int)
            model.add_regressor('is_payday')
            
            model.fit(p_train)

            # --- SAVE STEP ---
            filename = f"prophet_{subcat.replace(' ', '_')}.pkl"
            joblib.dump(model, os.path.join(MODEL_DIR, filename))

            # Forecast & Evaluate
            future = p_test[['ds']].copy()
            future['is_payday'] = future['ds'].dt.day.isin([15, 30]).astype(int)
            forecast = model.predict(future)
            preds = np.round(forecast['yhat'].clip(lower=0)).values
            actuals = p_test['y'].values
            
            rmse = np.sqrt(mean_squared_error(actuals, preds))
            smape = calculate_smape(actuals, preds)

            prophet_metrics.append({'Subcategory': subcat, 'RMSE': round(rmse, 4), 'sMAPE': round(smape, 2)})
            logging.info(f"Saved Prophet: {subcat}")
            
        except Exception as e:
            logging.error(f"Prophet failed for {subcat}: {e}")

    return pd.DataFrame(prophet_metrics)

def establish_cold_start_baselines(cold_start_full):
    """
    Calculates and persists the daily velocity for sparse/new items.
    """
    cold_metrics = []
    subcategories = cold_start_full['SubcategoryName'].unique()
    logging.info(f"Establishing baselines for {len(subcategories)} Cold Start items.")

    for subcat in subcategories:
        try:
            item_data = cold_start_full[cold_start_full['SubcategoryName'] == subcat]
            
            # Scientific Velocity: Total Sales / Days since Stocked
            total_sales = item_data['OrderQuantity'].sum()
            # We use the earliest StockDate and latest OrderDate in the record
            days_active = (item_data['OrderDate'].max() - item_data['StockDate'].min()).days + 1
            daily_velocity = total_sales / max(days_active, 1)

            # --- SAVE STEP ---
            # We save a simple dictionary as the "model"
            model_path = os.path.join(MODEL_DIR, f"cold_start_{subcat.replace(' ', '_')}.pkl")
            baseline = {
                'subcategory': subcat,
                'daily_velocity': daily_velocity,
                'status': 'sparse_baseline'
            }
            joblib.dump(baseline, model_path)
            
            cold_metrics.append({
                'Subcategory': subcat,
                'Daily_Velocity': round(daily_velocity, 4),
                'Total_Sales': total_sales
            })
            
        except Exception as e:
            logging.error(f"Cold Start baseline failed for {subcat}: {e}")

    return pd.DataFrame(cold_metrics)

if __name__ == "__main__":
    bundles = generate_features()
    
    # 1. Train ARIMA
    a_train, a_test = bundles['arima']
    arima_results = train_arima_models(a_train, a_test) if not a_train.empty else pd.DataFrame()

    # 2. Train Prophet
    p_train, p_test = bundles['prophet']
    prophet_results = train_prophet_models(p_train, p_test) if not p_train.empty else pd.DataFrame()

    # 3. Establish Cold Start Baselines
    c_full = bundles['cold_start']
    cold_results = establish_cold_start_baselines(c_full) if not c_full.empty else pd.DataFrame()

    print("\n" + "="*30)
    print("ALL MODELS & BASELINES PERSISTED")
    print("="*30)
    print(f"ARIMA Models: {len(arima_results)}")
    print(f"Prophet Models: {len(prophet_results)}")
    print(f"Cold Start Baselines: {len(cold_results)}")