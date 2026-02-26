import pandas as pd
import os

def load_and_merge_data():
    # Gets the directory where gathering_data.py is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Navigates from src/models/association_rule/ up to data/cleaned/
    # Adjust the number of '../' based on your folder depth
    base_path = os.path.join(current_dir, '../../../data/cleaned/')
    
    sales = pd.read_csv(os.path.join(base_path, 'Cleaned_Sales.csv'))[['OrderNumber', 'ProductKey']]
    products = pd.read_csv(os.path.join(base_path, 'Cleaned_Products.csv'))[['ProductKey', 'ProductSubcategoryKey']]
    subcats = pd.read_csv(os.path.join(base_path, 'Cleaned_Product_Subcategories.csv'))[['ProductSubcategoryKey', 'SubcategoryName']]

    df = sales.merge(products, on='ProductKey', how='left') \
              .merge(subcats, on='ProductSubcategoryKey', how='left')

    df = df[['OrderNumber', 'SubcategoryName']].dropna().reset_index(drop=True)
    
    return df