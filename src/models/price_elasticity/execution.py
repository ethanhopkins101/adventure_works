# src/model/price_elasticity/execute.py
import os
import glob
import pandas as pd
import numpy as np
import joblib

# Absolute imports starting from the project root (src)
from src.models.price_elasticity.gathering_data import get_cleaned_data
from src.models.price_elasticity.train import train_and_save_models
from src.models.price_elasticity.predictions import generate_optimization_plots, generate_performance_csv

def run_pipeline():
    # 1. PATH CONFIGURATION
    # Defining the absolute path to the models directory
    models_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../models/price_elasticity/'))
    
    # 2. LOGIC CHECK: Do the models exist?
    # We look for serialized .pkl files in the expected directory
    model_exists = False
    if os.path.exists(models_path):
        if glob.glob(os.path.join(models_path, "baseline_*.pkl")):
            model_exists = True

    # 3. PIPELINE ORCHESTRATION
    if not model_exists:
        print(">>> [STATUS] Models not found. Initializing FULL PIPELINE...")
        
        # Step A: Data Ingestion
        df = get_cleaned_data()
        
        # Step B: Model Training (Saves to folder)
        train_and_save_models(df)
        print(">>> [SUCCESS] Training complete. Models serialized to disk.")
    else:
        print(">>> [STATUS] Models detected. Initializing PREDICTION PIPELINE...")
        
        # Step A: Data Ingestion (Required for context and baseline)
        df = get_cleaned_data()

    # 4. RESULTS GENERATION
    # Step C: Run performance table generation
    csv_path = generate_performance_csv(df, models_path)
    print(f">>> [EXPORT] Performance table generated at: {csv_path}")

    # Step D: Optimization Simulation for Plotting
    # To generate the curves, we simulate a range of prices per category
    all_gam_results = pd.DataFrame()
    best_profit_points = []

    unique_items = df['CategoryName'].unique()
    for item in unique_items:
        baseline_model_path = os.path.join(models_path, f'baseline_{item}.pkl')
        if os.path.exists(baseline_model_path):
            gam = joblib.load(baseline_model_path)
            
            # Fetch item specifics for profit calculation
            item_data = df[df['CategoryName'] == item]
            cost = (item_data['ProductPrice'] - (item_data['profit'] / item_data['OrderQuantity'])).median()
            
            # Create a dense price grid for optimization visualization
            p_min, p_max = item_data['ProductPrice'].min() * 0.7, item_data['ProductPrice'].max() * 1.3
            price_grid = np.linspace(p_min, p_max, 100).reshape(-1, 1)
            
            # Predict and calculate simulated profit
            qty_preds = gam.predict(price_grid)
            
            res_df = pd.DataFrame({
                'ProductPrice': price_grid.flatten(),
                'CategoryName': item,
                'profit_pred_0.5': (price_grid.flatten() - cost) * qty_preds,
                'profit_pred_0.025': (price_grid.flatten() - cost) * (qty_preds * 0.9), # Mocking CI for plotting
                'profit_pred_0.975': (price_grid.flatten() - cost) * (qty_preds * 1.1)
            })
            
            all_gam_results = pd.concat([all_gam_results, res_df])
            best_profit_points.append(res_df.loc[res_df['profit_pred_0.5'].idxmax()])

    # Step E: Visualization
    best_profit_df = pd.DataFrame(best_profit_points)
    plot_path = generate_optimization_plots(all_gam_results, best_profit_df)
    print(f">>> [VISUAL] Optimization charts saved at: {plot_path}")

if __name__ == "__main__":
    run_pipeline()