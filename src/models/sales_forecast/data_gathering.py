import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def gather_sales_data():
    """Rule 1 & 2: Pulls raw data and prepares the baseline."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, "../../../"))
        base_path = os.path.join(project_root, "data", "cleaned")
        
        # Load raw files
        df_sales = pd.read_csv(os.path.join(base_path, 'Cleaned_Sales.csv'))
        df_products = pd.read_csv(os.path.join(base_path, 'Cleaned_Products.csv'))
        df_subcat = pd.read_csv(os.path.join(base_path, 'Cleaned_Product_Subcategories.csv'))

        # Merge to get Subcategory names
        df_merged = df_sales[['OrderDate', 'StockDate', 'OrderQuantity', 'ProductKey']].merge(
            df_products[['ProductKey', 'ProductSubcategoryKey']], on='ProductKey', how='left'
        ).merge(
            df_subcat[['ProductSubcategoryKey', 'SubcategoryName']], on='ProductSubcategoryKey', how='left'
        )

        # Scientific Standard: Proper Date objects
        df_merged['OrderDate'] = pd.to_datetime(df_merged['OrderDate'], dayfirst=True)
        df_merged['StockDate'] = pd.to_datetime(df_merged['StockDate'], dayfirst=True)
        
        # Cutoff: Aug 2016
        cutoff_date = pd.to_datetime('2016-08-01')
        df_final = df_merged[df_merged['OrderDate'] >= cutoff_date].copy()
        
        # Capture global baseline for imputation
        min_stock_date = df_final['StockDate'].min()

        return df_final, min_stock_date, df_subcat

    except Exception as e:
        logging.error(f"Gathering failed: {e}")
        raise

if __name__ == "__main__":
    pass