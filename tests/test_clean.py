import pandas as pd
from src.data.clean import clean_customers

def test_cleaning():
    # Create dummy data with intentional errors
    data = {
        'FirstName': ['John', 'Jane', 'Invalid123', None],
        'LastName': ['Doe', 'Smith', 'Doe', None],
        'Prefix': [None, 'MRS.', 'MR.', 'MR.'],
        'MaritalStatus': ['S', 'M', 'S', 'S'],
        'Gender': [None, 'F', 'M', 'M'],
        'BirthDate': ['1980-01-01', '1850-01-01', '1990-01-01', '1990-01-01'], # 1850 is >100 years
        'EducationLevel': ['Bachelors', 'High School', 'Bachelors', 'Bachelors'],
        'Occupation': ['Professional', 'Clerical', 'Professional', 'Professional'],
        'HomeOwner': ['Y', 'N', 'Y', 'Y'],
        'AnnualIncome': ['$50,000', '$60,000', '$70,000', '$70,000'],
        'EmailAddress': ['john@test.com', 'jane@test.com', 'bad_email', 'valid@test.com'],
        'TotalChildren': [2, 1, 20, 0], # 20 is >15
        'ExtraCol': [1, 2, 3, 4] # Total 13 columns needed
    }
    
    df_test = pd.DataFrame(data)
    
    # Run cleaning
    cleaned_df = clean_customers(df_test)
    
    # Verification checks
    print(f"Rows before: {len(df_test)} | Rows after: {len(cleaned_df)}")
    print(f"Gender 'U' count: {(cleaned_df['Gender'] == 'U').sum()}")
    print(f"Max Age Check: {cleaned_df['BirthDate'].min()}")
    
    return cleaned_df

from src.data.clean import clean_products

def test_product_cleaning():
    # 1. Create dummy product data with intentional errors
    product_data = {
        'ProductKey': [1, 2, 3, 4, 5, 6, 7],
        'ProductSKU': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
        'ProductName': ['Silver Bike', 'Red Bike', 'Blue Helmet', 'Black Jersey', 'Silver Lock', 'Yellow Bell', 'Grey Pump'],
        'ModelName': ['M1', 'M1', 'M1', 'M1', 'M1', 'M1', 'M1'],
        'ProductDescription': ['D']*7,
        'ProductColor': [None, 'Red', None, 'Black', None, 'Yellow', 'Grey'], # 3 Nones to test imputation
        'ProductSize': ['L', 'XL', '62', 'M', 'S', '44', '52'],
        'ProductStyle': ['U', 'M', '0', 'W', 'U', 'M', '0'],
        'ProductCost': [100.0, 100.0, 101.0, 100.0, 100.0, 99.0, 100.0], # Similar costs
        'ProductPrice': [150.0, 150.0, 151.0, 150.0, 150.0, 149.0, 150.0], # Similar prices
        'ProductSubcategoryKey': [1, 2, 3, 4, 5, 6, 7]
    }
    
    df_prod_test = pd.DataFrame(product_data)
    
    # 2. Run the cleaning function
    cleaned_products = clean_products(df_prod_test)
    
    # 3. Validation results
    print("\n--- Product Cleaning Test Results ---")
    print(f"Rows before: {len(df_prod_test)} | Rows after: {len(cleaned_products)}")
    
    if not cleaned_products.empty:
        print("\n--- Cleaned Product Data ---")
        print(cleaned_products[['ProductName', 'ProductColor', 'ProductSize', 'ProductPrice']].to_string(index=False))
        
    return cleaned_products

from src.data.clean import clean_calendar

def test_calendar_cleaning():
    data = {'Date': ['2023-02-13', '01-2023-02', '6-25-2021', None]}
    df_cal = pd.DataFrame(data)
    
    cleaned_cal = clean_calendar(df_cal)
    
    print("\n--- Cleaned Calendar Data ---")
    print(cleaned_cal)
    return cleaned_cal


from src.data.clean import clean_subcategories

def test_subcategory_cleaning():
    data = {
        'SubcategoryKey': [1, 2, 3, 4, 5],
        'SubcategoryName': [' Mountain Bikes ', 'Road Bikes', 'Bikes123', None, 'Wheels'],
        'CategoryKey': [1, 1, 1, 1, 2]
    }
    df_sub = pd.DataFrame(data)
    
    # This should trigger the Try-Catch because there are 3 columns (Pass)
    cleaned_sub = clean_subcategories(df_sub)
    
    print("\n--- Cleaned Subcategories ---")
    print(cleaned_sub.to_string(index=False))
    return cleaned_sub

from src.data.clean import clean_categories

def test_category_cleaning():
    data = {
        'CategoryKey': [1, 2, 3, 4],
        'CategoryName': [' Bikes ', 'Components', 'Clothing1', None]
    }
    df_cat = pd.DataFrame(data)
    
    cleaned_cat = clean_categories(df_cat)
    
    print("\n--- Cleaned Categories ---")
    print(cleaned_cat.to_string(index=False))
    return cleaned_cat

from src.data.clean import clean_territories

def test_territory_cleaning():
    data = {
        'SalesTerritoryKey': [1, 2, 3, 4],
        'Region': [' Northwest ', 'Central', None, None],
        'Country': ['United States', 'United States', None, 'France'],
        'Continent': ['North America', 'North America', 'Unknown', 'Europe']
    }
    df_terr = pd.DataFrame(data)
    
    cleaned_terr = clean_territories(df_terr)
    
    print("\n--- Cleaned Territories ---")
    print(cleaned_terr.to_string(index=False))
    return cleaned_terr

from src.data.clean import clean_sales

def test_sales_cleaning():
    data = {
        'OrderDate': ['2023-01-01', '02-01-2023', '2023-01-01', '2023-01-05'],
        'StockDate': ['2022-12-01', '2022-12-01', '2022-12-01', '2022-12-01'],
        'OrderNumber': [' SO123 ', 'SO456', 'SO123', 'SO789'],
        'ProductKey': [101, 102, 101, 103],
        'CustomerKey': [500, 501, 500, 502],
        'TerritoryKey': [1, 1, 1, 1],
        'OrderLineItem': [1, 1, 1, 1],
        'OrderQuantity': [1, 2, 1, 132] # 132 will likely be the 99% outlier in this small set
    }
    df_sales = pd.DataFrame(data)
    
    cleaned_sales = clean_sales(df_sales)
    
    print("\n--- Cleaned Sales ---")
    print(cleaned_sales)
    return cleaned_sales

if __name__ == "__main__":
    print("RUNNING ALL DATA CLEANING TESTS...")
    
    # Run individual tests
    # test_cleaning()
    # test_product_cleaning()
    # test_calendar_cleaning()
    #test_subcategory_cleaning()
    #test_category_cleaning()
    #test_territory_cleaning()
    test_sales_cleaning()