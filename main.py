import pandas as pd
import os
import glob
from src.data.clean import (
    clean_customers, clean_products, clean_calendar, 
    clean_subcategories, clean_categories, clean_territories, 
    clean_sales, clean_returns, clean_price_elasticity  # Added import
)

# Import Model Pipelines
from src.models.sales_forecast.execution import run_pipeline as run_sales_pipeline
from src.models.returns_forecast.execution import run_pipeline as run_returns_pipeline
from src.models.bayesian_mmm.execution import run_pipeline as run_mmm_pipeline
from src.models.association_rule.execution import run_pipeline as run_rules_pipeline
from src.models.price_elasticity.execution import run_pipeline as run_elasticity_pipeline
from src.models.customer_lifetime_value.execution import run_pipeline as run_clv_pipeline

# Import Decoder
from src.data.decoder import run_decoder

def run_main_pipeline():
    # --- Part 1: Data Cleaning ---
    raw_path = 'data/raw/'
    clean_path = 'data/cleaned/'
    os.makedirs(clean_path, exist_ok=True)

    # Added price_elasticity.csv to the tasks dictionary
    tasks = {
        'AdventureWorks_Customers.csv': clean_customers,
        'AdventureWorks_Products.csv': clean_products,
        'AdventureWorks_Calendar.csv': clean_calendar,
        'AdventureWorks_Product_Subcategories.csv': clean_subcategories,
        'AdventureWorks_Product_Categories.csv': clean_categories,
        'AdventureWorks_Territories.csv': clean_territories,
        'AdventureWorks_Returns.csv': clean_returns,
        'price_elasticity.csv': clean_price_elasticity  
    }

    print("🚀 Starting Data Cleaning Pipeline...")
    for file_name, clean_func in tasks.items():
        full_path = os.path.join(raw_path, file_name)
        if os.path.exists(full_path):
            df = pd.read_csv(full_path, encoding='latin1')
            df_cleaned = clean_func(df)
            
            # Ensures file becomes 'price_elasticity.csv' in cleaned folder
            output_name = file_name.replace('AdventureWorks_', 'Cleaned_')
            if file_name == 'price_elasticity.csv':
                output_name = 'price_elasticity.csv'
                
            df_cleaned.to_csv(os.path.join(clean_path, output_name), index=False)
            print(f"✅ Success: {file_name}")

    # Process Sales files
    sales_files = glob.glob(os.path.join(raw_path, 'AdventureWorks_Sales*.csv'))
    if sales_files:
        combined_sales = pd.concat([pd.read_csv(f, encoding='latin1') for f in sales_files])
        df_sales_cleaned = clean_sales(combined_sales)
        df_sales_cleaned.to_csv(os.path.join(clean_path, 'Cleaned_Sales.csv'), index=False)
        print(f"✅ Success: Combined {len(sales_files)} Sales files")

    # (Remaining model execution code follows...)
# --- Part 2: Execute Model Pipelines (Sequential & Verified) ---
    print("\n🧠 Running Model Intelligence Suites...")
    
    # 1. Sales Forecast (Foundation)
    print("📈 Running Sales Forecast...")
    run_sales_pipeline()
    
    # Optional: Verify critical output exists before moving to Returns
    sales_output = 'json_files/sales_forecast/latest_sales_forecast.json'
    if not os.path.exists(sales_output):
        print(f"⚠️ Warning: Sales Forecast did not produce {sales_output}. Returns Forecast may fail.")

    # 2. Returns Forecast (Depends on Sales data)
    print("🔄 Running Returns Forecast...")
    run_returns_pipeline()
    
    # 3. Bayesian MMM
    print("📊 Running Bayesian MMM...")
    run_mmm_pipeline()
    
    # 4. Association Rules
    print("🛒 Running Association Rules...")
    run_rules_pipeline()
    
    # 5. Price Elasticity
    print("🏷️ Running Price Elasticity...")
    run_elasticity_pipeline()
    
    # 6. Customer Lifetime Value
    print("👥 Running Customer Lifetime Value...")
    run_clv_pipeline()



# --- Part 3: Decode JSON Outputs ---
    print("\n🔐 Decoding Encoded JSON Outputs...")
    
    # 1. Get the absolute path to the project root
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Define absolute paths for inputs (encoded) and outputs (decoded)
    # Source: adventure_works/json_files/sales_forecast/encoded/
    encoded_json_dir = os.path.abspath(os.path.join(base_dir, 'json_files/sales_forecast/encoded/'))
    # Destination: adventure_works/json_files/sales_forecast/decoded/
    decoded_output_dir = os.path.abspath(os.path.join(base_dir, 'json_files/sales_forecast/decoded/'))
    
    os.makedirs(decoded_output_dir, exist_ok=True)

    files_to_decode = ['latest_sales_forecast.json', 'stocking_report.json']

    for file_name in files_to_decode:
        input_path = os.path.join(encoded_json_dir, file_name)
        output_path = os.path.join(decoded_output_dir, f"decoded_{file_name}")
        
        if os.path.exists(input_path):
            run_decoder(input_path, output_path)
            print(f"✅ Success: Decoded {file_name} into {output_path}")
        else:
            # Helps you verify if the file exists in the 'encoded' folder
            print(f"❌ Error: File not found in encoded folder: {input_path}")



    print("\n✨ All Systems Complete!")

if __name__ == "__main__":
    run_main_pipeline()