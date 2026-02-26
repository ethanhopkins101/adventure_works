# src/model/price_elasticity/gathering_data.py
import pandas as pd
import os

def get_cleaned_data():
    """
    Pulls the price elasticity dataset from the absolute source path.
    
    Returns:
        pd.DataFrame: The loaded price_elasticity dataframe.
    """
    # Define absolute source path relative to this script's location
    # Moving up from src/model/price_elasticity to root, then to data/cleaned
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/cleaned/price_elasticity.csv'))
    
    df = pd.read_csv(path)
    
    return df