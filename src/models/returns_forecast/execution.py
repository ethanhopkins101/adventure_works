import os
import glob
import pandas as pd

# Absolute imports starting from the project root (src)
from src.models.returns_forecast.data_gathering import get_returns_raw_data
import src.models.returns_forecast.features as ft
import src.models.returns_forecast.train as tr
import src.models.returns_forecast.predictions as pr

def run_pipeline(force_retrain=False):
    """
    Coordinates the returns forecasting pipeline with full 37-item coverage.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.abspath(os.path.join(base_dir, '../../../models/returns_forecast/'))
    
    sales_forecast_path = os.path.abspath(os.path.join(
        base_dir, 
        '../../../json_files/sales_forecast/encoded/latest_sales_forecast.json'
    ))
    
    existing_models = glob.glob(os.path.join(model_dir, "*.pkl"))
    needs_training = force_retrain or (len(existing_models) == 0)

    print(f"--- Starting Pipeline (Training Required: {needs_training}) ---")

    try:
        # 2. Data Gathering 
        # Unpack both the historical data and the master 37-item list
        raw_df, subcategories_df = get_returns_raw_data()
        
        # 3. Feature Engineering
        # Inject the master list into the skeleton to ensure zero-data items are created
        df_skeleton = ft.create_daily_skeleton(raw_df, subcategories_df)
        df_lagged = ft.add_lagged_features(df_skeleton)
        df_encoded = ft.apply_persistent_encoding(df_lagged)

        if needs_training:
            print("Action: Routing items and training models...")
            
            # Unpack the full DF, training bundles, and the routing map
            df_encoded, bundles, route_map = ft.route_by_sparsity(df_encoded)
            
            tr.train_arima_models(bundles['arima'])
            tr.train_prophet_models(bundles['prophet'])
            print(f"Training complete. Models saved to: {model_dir}")
        else:
            print("Action: Fast-tracking to inference using existing models...")

        # 4. Generate Future Forecast
        # Passing df_encoded now guarantees 37 subcategories in the final JSON
        forecast_results = pr.run_future_forecast(
            df=df_encoded, 
            sales_forecast_json_path=sales_forecast_path,
            horizon=30
        )
        
        print(f"Success! Forecast generated for {len(forecast_results)} subcategories.")
        return forecast_results

    except Exception as e:
        print(f"Pipeline Failed: {str(e)}")
        return None

if __name__ == "__main__":
    run_pipeline()