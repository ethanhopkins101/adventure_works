import pandas as pd
import os

def get_returns_raw_data():
    """
    Retrieves and joins sales, returns, and product metadata.
    
    Returns:
        tuple: (final_df, subcategories_df)
            - final_df: Merged dataset for forecasting.
            - subcategories_df: Master list of all 37 subcategory names.
    """
    # Define absolute path relative to this script's location
    base_path = os.path.join(os.path.dirname(__file__), '../../../data/cleaned/')

    # 1. Load datasets
    df_sales = pd.read_csv(os.path.join(base_path, 'Cleaned_Sales.csv'))
    df_returns = pd.read_csv(os.path.join(base_path, 'Cleaned_Returns.csv'))
    df_products = pd.read_csv(os.path.join(base_path, 'Cleaned_Products.csv'))
    df_subcat = pd.read_csv(os.path.join(base_path, 'Cleaned_Product_Subcategories.csv'))

    # 2. Master Subcategory List (The Truth List)
    # Extract only the names to ensure all 37 items are tracked
    subcategories_df = df_subcat[['SubcategoryName']].drop_duplicates()

    # 3. Map ProductKey to SubcategoryName
    product_map = df_products[['ProductKey', 'ProductSubcategoryKey']].merge(
        df_subcat[['ProductSubcategoryKey', 'SubcategoryName']], 
        on='ProductSubcategoryKey', 
        how='left'
    )[['ProductKey', 'SubcategoryName']]

    # 4. Process Sales
    sales_processed = df_sales[['OrderDate', 'OrderQuantity', 'ProductKey']].merge(
        product_map, on='ProductKey', how='left'
    ).groupby(['OrderDate', 'SubcategoryName'])['OrderQuantity'].sum().reset_index()

    # 5. Process Returns
    returns_processed = df_returns[['ReturnDate', 'ReturnQuantity', 'ProductKey']].merge(
        product_map, on='ProductKey', how='left'
    ).groupby(['ReturnDate', 'SubcategoryName'])['ReturnQuantity'].sum().reset_index()

    # 6. Merge datasets
    final_df = pd.merge(
        sales_processed, 
        returns_processed, 
        left_on=['OrderDate', 'SubcategoryName'], 
        right_on=['ReturnDate', 'SubcategoryName'], 
        how='left'
    )

    # 7. Cleaning and Formatting
    final_df['ReturnQuantity'] = final_df['ReturnQuantity'].fillna(0)
    final_df['OrderDate'] = pd.to_datetime(final_df['OrderDate'], dayfirst=True)
    final_df['ReturnDate'] = pd.to_datetime(final_df['ReturnDate'], dayfirst=True)

    # 8. Apply Data Cut-off
    final_df = final_df[final_df['OrderDate'] >= '2016-08-01']

    # Final selection
    final_df = final_df[[
        'ReturnDate', 
        'ReturnQuantity', 
        'SubcategoryName', 
        'OrderDate', 
        'OrderQuantity'
    ]]

    return final_df, subcategories_df