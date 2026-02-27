import pandas as pd
import numpy as np
import os
import json
import joblib
from datetime import timedelta
from prophet import Prophet

def get_forecast_path():
    """Manages directory for the final return forecast JSON."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.abspath(os.path.join(base_dir, '../../../json_files/returns_forecast/'))
    os.makedirs(path, exist_ok=True)
    return path

def run_future_forecast(df, sales_forecast_json_path, horizon=30):
    """
    Generates return forecasts by matching SubcategoryEncoded IDs 
    to the provided Sales Forecast JSON structure.
    """
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../models/returns_forecast/'))
    
    # 1. Load Sales Forecast
    with open(sales_forecast_json_path, 'r') as f:
        sales_forecast = json.load(f)

    max_date = df['ReturnDate'].max()
    future_dates = [max_date + timedelta(days=i) for i in range(1, horizon + 1)]
    
    final_results = []

    # 2. Iterate through every subcategory
    for subcat_name in df['SubcategoryName'].unique():
        sub_df = df[df['SubcategoryName'] == subcat_name].sort_values('ReturnDate')
        sub_id = str(int(sub_df['SubcategoryEncoded'].iloc[0])) # Key in JSON is a string ID
        
        # Extract sales forecast from JSON using the ID key
        try:
            item_data = sales_forecast.get(sub_id, {})
            daily_dict = item_data.get('daily_forecast', {})
            # Sort by date and take the values
            future_sales = [daily_dict[d] for d in sorted(daily_dict.keys())[:horizon]]
            
            # Fallback if JSON list is shorter than horizon
            if len(future_sales) < horizon:
                padding = [sub_df['OrderQuantity'].mean()] * (horizon - len(future_sales))
                future_sales.extend(padding)
        except Exception:
            future_sales = [sub_df['OrderQuantity'].mean()] * horizon

        # 3. Model Matching (Strict ID-based filenames)
        model_path = os.path.join(model_dir, f'{sub_id}.pkl')
        preds = []
        model_type = "ColdStart"

        if os.path.exists(model_path):
            model = joblib.load(model_path)
            
            # Scientifically differentiate between ARIMA and Prophet objects
            if isinstance(model, Prophet):
                model_type = "Prophet"
                future_p = pd.DataFrame({
                    'ds': future_dates, 
                    'OrderQuantity_Lag1': future_sales
                })
                forecast = model.predict(future_p)
                preds = forecast['yhat'].values
            else:
                model_type = "AutoARIMA"
                # ARIMA uses X for exogenous future values
                preds = model.predict(n_periods=horizon, X=np.array(future_sales).reshape(-1, 1))
        
        else:
            # Enhanced Cold Start
            model_type = "ColdStart"
            hist_avg_returns = sub_df['ReturnQuantity'].tail(15).mean()
            hist_avg_sales = sub_df['OrderQuantity'].tail(15).mean()
            hist_avg_sales = hist_avg_sales if hist_avg_sales > 0 else 1
            
            for sale_qty in future_sales:
                multiplier = sale_qty / hist_avg_sales
                preds.append(hist_avg_returns * multiplier)

        # 4. Formatting Output
        total_pred = int(round(max(0, sum(preds))))
        conf = 95.0 if model_type == "AutoARIMA" else 85.0 if model_type == "Prophet" else 65.0

        final_results.append({
            'SubcategoryName': subcat_name,
            'SubcategoryID': sub_id,
            'Predicted_Returns_Total': total_pred,
            'Confidence_Rating': f"{conf}%",
            'Model_Used': model_type,
            'Forecast_Start': str(future_dates[0].date()),
            'Forecast_End': str(future_dates[-1].date())
        })

    # 5. Save Output
    output_path = os.path.join(get_forecast_path(), 'final_returns_forecast.json')
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=4)
    
    return final_results