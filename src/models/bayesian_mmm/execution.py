import os
import pickle

# Absolute imports starting from the project root (src)
from src.models.bayesian_mmm.gathering_data import gather_mmm_data
from src.models.bayesian_mmm.features import process_features
from src.models.bayesian_mmm.train import train_and_save_model
from src.models.bayesian_mmm.predictions import (
    get_roi_analysis, 
    generate_waterfall_chart, 
    run_budget_simulations
)

# Absolute path to the model file
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, '../../../models/bayesian_mmm/mmm_orbit_model.pkl')

def run_pipeline():
    print("🚀 Starting Bayesian MMM Pipeline...")

    # 1. Data Gathering
    df_raw = gather_mmm_data()
    print("✅ Phase 1: Data Gathering Complete.")

    # 2. Feature Engineering
    df_adstocked = process_features(df_raw)
    print("✅ Phase 2: Feature Engineering (Adstock) Complete.")

    # 3. Training Check (Conditional Logic)
    if os.path.exists(MODEL_PATH):
        print(f"🔄 Model found at {MODEL_PATH}. Skipping training phase.")
        model_to_use_path = MODEL_PATH
    else:
        print("⚠️ No saved model found. Initiating training...")
        # Fits model and saves to path
        model_to_use_path = train_and_save_model(df_adstocked)
        print(f"✅ Phase 3: Model Training Complete. Model saved at: {model_to_use_path}")

    # 4. Predictions & Reporting
    # A. Get ROI Table (Internally saves to ../../../data/models/bayesian_mmm/)
    roi_table = get_roi_analysis(df_raw, df_adstocked)
    print("\n--- PROFIT ROI ANALYSIS ---")
    print(roi_table)

    # B. Generate & Save Waterfall Chart
    generate_waterfall_chart(df_adstocked)
    print("\n✅ Phase 4a: Waterfall Chart generated and saved.")

    # C. Run Budget Simulations
    run_budget_simulations(budgets=[15000, 20000])
    print("✅ Phase 4b: Budget Simulations ($15k & $20k) saved to JSON.")

    print("\n🎉 Pipeline Execution Successful!")

if __name__ == "__main__":
    run_pipeline()