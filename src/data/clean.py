import pandas as pd
import numpy as np
import re
import gender_guesser.detector as gender

# Initialize gender detector globally to avoid re-instantiation
detector = gender.Detector()

def clean_customers(df):
    """
    Cleans the Adventure Works Customer table following 12 specific steps.
    """
    df_clean = df.copy()

    # 1. Column Count Validation
    try:
        if len(df_clean.columns) != 13:
            raise ValueError(f"Expected 13 columns, found {len(df_clean.columns)}")
    except ValueError as e:
        print(f"Column Validation Error: {e}")

    # 2. Row Filtering (NaNs, Duplicates, Missing Names)
    # thresh: Keep only rows with at least 50% non-NA values
    limit = len(df_clean.columns) // 2
    df_clean = df_clean.dropna(thresh=limit)
    df_clean = df_clean.drop_duplicates()
    df_clean = df_clean.dropna(subset=['FirstName', 'LastName'], how='all')

    # 4. Strip whitespace from all string cells
    df_clean = df_clean.map(lambda x: x.strip() if isinstance(x, str) else x)

    # 3. Categorical & DateTime Transformation
    cat_cols = ['Prefix', 'MaritalStatus', 'Gender', 'EducationLevel', 'Occupation', 'HomeOwner']
    for col in cat_cols:
        df_clean[col] = df_clean[col].astype('category')
    
    df_clean['BirthDate'] = pd.to_datetime(df_clean['BirthDate'])

    # 5. Age Validation (Proxy date: 01-06-2017)
    try:
        proxy_date = pd.to_datetime('2017-06-01')
        age = (proxy_date - df_clean['BirthDate']).dt.days / 365.25
        if (age > 100).any():
            print("Warning: Customers older than 100 years detected. Filtering...")
            df_clean = df_clean[age <= 100]
    except Exception as e:
        print(f"Age Validation Error: {e}")

    # 6. Gender & Prefix Imputation
    def impute_logic(row):
        g = row['Gender']
        p = row['Prefix']
        
        if pd.isna(g):
            guess = detector.get_gender(str(row['FirstName']))
            if 'female' in guess: g = 'F'
            elif 'male' in guess: g = 'M'
        
        if pd.isna(p):
            if g == 'M': p = 'MR.'
            elif g == 'F':
                p = 'MRS.' if row['MaritalStatus'] == 'M' else 'MS.'
        return pd.Series([g, p])

    df_clean[['Gender', 'Prefix']] = df_clean.apply(impute_logic, axis=1)
    df_clean['Gender'] = df_clean['Gender'].fillna('U').astype('category')
    df_clean['Prefix'] = df_clean['Prefix'].fillna(df_clean['Prefix'].mode()[0]).astype('category')

    # 7. Domain Constraints Validation
    valid_values = {
        'Prefix': ['MR.', 'MS.', 'MRS.'],
        'MaritalStatus': ['M', 'S'],
        'Gender': ['M', 'F', 'U'],
        'EducationLevel': ['Bachelors', 'Partial College', 'High School', 'Partial High School', 'Graduate Degree'],
        'Occupation': ['Professional', 'Management', 'Skilled Manual', 'Clerical', 'Manual'],
        'HomeOwner': ['Y', 'N']
    }
    try:
        for col, valid_list in valid_values.items():
            invalid = df_clean[~df_clean[col].isin(valid_list)]
            if not invalid.empty:
                print(f"Warning: Invalid values found in {col}. Removing {len(invalid)} rows.")
                df_clean = df_clean[df_clean[col].isin(valid_list)]
    except Exception as e:
        print(f"Domain Validation Error: {e}")

# 8. AnnualIncome to Float
    try:
        # Convert to string first to ensure .str accessor works, 
        # then strip symbols, then convert to numeric
        df_clean['AnnualIncome'] = pd.to_numeric(
            df_clean['AnnualIncome'].astype(str)
            .str.replace(r'[\$,]', '', regex=True), 
            errors='coerce'
        )
        # Drop rows where income couldn't be converted
        df_clean = df_clean.dropna(subset=['AnnualIncome'])
    except Exception as e:
        print(f"AnnualIncome Conversion Error: {e}")
    # 9. Email Format Validation (Regex)
    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    df_clean = df_clean[df_clean['EmailAddress'].str.contains(email_regex, na=False, regex=True)]

    # 10. String-only Names
    name_regex = r'^[A-Za-z\s\-]+$'
    df_clean = df_clean[df_clean['FirstName'].str.contains(name_regex, na=False) & 
                        df_clean['LastName'].str.contains(name_regex, na=False)]

    # 11. TotalChildren Constraint
    df_clean = df_clean[df_clean['TotalChildren'] <= 15]

    # 12. Annual Income 99th Percentile Filter
    try:
        # Final check: ensure the column is float64
        df_clean['AnnualIncome'] = df_clean['AnnualIncome'].astype(float)
        
        lower_bound = df_clean['AnnualIncome'].quantile(0.005)
        upper_bound = df_clean['AnnualIncome'].quantile(0.995)
        
        df_clean = df_clean[(df_clean['AnnualIncome'] >= lower_bound) & 
                            (df_clean['AnnualIncome'] <= upper_bound)]
    except Exception as e:
        print(f"Quantile Calculation Error: {e}")
    return df_clean


def clean_products(df):
    """
    Cleans the Adventure Works Product table following 7 specific steps.
    """
    df_clean = df.copy()

    # 1. Column Count Validation
    try:
        if len(df_clean.columns) != 11:
            raise ValueError(f"Expected 11 columns, found {len(df_clean.columns)}")
    except ValueError as e:
        print(f"Column Validation Error: {e}")

    # 2. Row Filtering (NaNs, Duplicates, Missing Subcategory)
    limit = len(df_clean.columns) // 2
    df_clean = df_clean.dropna(thresh=limit)
    df_clean = df_clean.drop_duplicates()
    df_clean = df_clean.dropna(subset=['ProductSubcategoryKey'])

    # 4. Strip whitespace from all string cells
    df_clean = df_clean.map(lambda x: x.strip() if isinstance(x, str) else x)

    # 6. Impute Colors from ProductName
    colors_list = ['Silver/Black', 'Red', 'Black', 'White', 'Blue', 'Multi', 'Silver', 'Yellow', 'Grey']
    mode_color = df_clean['ProductColor'].mode()[0]

    def find_color(row):
        if pd.isna(row['ProductColor']) or str(row['ProductColor']).lower() == 'nan':
            for color in colors_list:
                if color.lower() in str(row['ProductName']).lower():
                    return color
            return mode_color
        return row['ProductColor']

    df_clean['ProductColor'] = df_clean.apply(find_color, axis=1)

    # 3. Categorical Transformation
    cat_cols = ['ProductColor', 'ProductSize', 'ProductStyle']
    for col in cat_cols:
        df_clean[col] = df_clean[col].astype('category')

    # 5. Price & Cost Validation (Numeric + 99% Distribution)
    for col in ['ProductPrice', 'ProductCost']:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        df_clean = df_clean.dropna(subset=[col])
        
        # 99% Distribution Filter
        lower_bound = df_clean[col].quantile(0.005)
        upper_bound = df_clean[col].quantile(0.995)
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

    # 7. Domain Constraints Validation
    valid_values = {
        'ProductColor': colors_list,
        'ProductSize': ['0', 'M', 'L', 'S', 'XL', '62', '44', '48', '52', '56', '58', '60', '42', '46', '38', '40', '70', '50', '54'],
        'ProductStyle': ['0', 'U', 'W', 'M']
    }
    
    try:
        for col, valid_list in valid_values.items():
            # Convert both to string for consistent comparison
            df_clean = df_clean[df_clean[col].astype(str).isin(valid_list)]
    except Exception as e:
        print(f"Domain Validation Error: {e}")

    return df_clean


def clean_calendar(df):
    df_clean = df.copy()

    # 1. Drop NaNs
    df_clean = df_clean.dropna(subset=['Date'])

    # 2. Strict & Flexible Parsing
    try:
        df_clean['Date'] = pd.to_datetime(
            df_clean['Date'], 
            dayfirst=True, 
            format='mixed', 
            errors='coerce'
        )

        # --- TEMPORAL GUARDRAIL ---
        # Filters out any dates that shouldn't exist in the historical dataset
        proxy_max_date = pd.Timestamp('2017-06-30')
        df_clean = df_clean[df_clean['Date'] <= proxy_max_date]

        # Drop rows that failed to parse (NaT)
        df_clean = df_clean.dropna(subset=['Date'])

        # 3. Sort by Date
        df_clean = df_clean.sort_values(by='Date')
        
        # 4. Final Formatting to 'DD-MM-YYYY' string
        df_clean['Date'] = df_clean['Date'].dt.strftime('%d-%m-%Y')
        
    except Exception as e:
        print(f"Calendar Logic Error: {e}")

    return df_clean


def clean_subcategories(df):
    """
    Cleans Product Subcategories: validates schema, removes noise, 
    and enforces string integrity.
    """
    df_clean = df.copy()

    # 1. Column Count Validation
    try:
        if len(df_clean.columns) != 3:
            raise ValueError(f"Expected 3 columns, but found {len(df_clean.columns)}")
    except ValueError as e:
        print(f"Schema Validation Error: {e}")
        # In a real pipeline, you might return None or raise the error here

    # 2. Drop Duplicates and NaNs
    df_clean = df_clean.drop_duplicates()
    df_clean = df_clean.dropna(subset=['SubcategoryName'])

    # 3 & 5. String Cleaning: Strip spaces and remove names with numbers
    try:
        # Strip spaces first
        df_clean['SubcategoryName'] = df_clean['SubcategoryName'].astype(str).str.strip()
        
        # Keep only rows where SubcategoryName contains NO digits (\d)
        df_clean = df_clean[~df_clean['SubcategoryName'].str.contains(r'\d', na=False)]
    except Exception as e:
        print(f"String Transformation Error: {e}")

    # 4. Transform into Categories
    df_clean['SubcategoryName'] = df_clean['SubcategoryName'].astype('category')

    return df_clean


def clean_categories(df):
    """
    Cleans Product Categories: enforces 2-column schema and string purity.
    """
    df_clean = df.copy()

    # 1. Column Count Validation
    try:
        if len(df_clean.columns) != 2:
            raise ValueError(f"Expected 2 columns, but found {len(df_clean.columns)}")
    except ValueError as e:
        print(f"Category Schema Error: {e}")

    # 2. Drop Duplicates and NaNs
    df_clean = df_clean.drop_duplicates()
    df_clean = df_clean.dropna(subset=['CategoryName'])

    # 3 & 5. String Cleaning: Strip and Remove Numbers
    try:
        # Strip whitespace
        df_clean['CategoryName'] = df_clean['CategoryName'].astype(str).str.strip()
        
        # Keep only rows without digits
        df_clean = df_clean[~df_clean['CategoryName'].str.contains(r'\d', na=False)]
    except Exception as e:
        print(f"Category Name Error: {e}")

    # 4. Transform to categories
    df_clean['CategoryName'] = df_clean['CategoryName'].astype('category')

    return df_clean


def clean_territories(df):
    """
    Cleans Territory data: validates schema, handles spatial NaNs, 
    and sanitizes categorical regions.
    """
    df_clean = df.copy()

    # 1. Column Count Validation
    try:
        if len(df_clean.columns) != 4:
            raise ValueError(f"Expected 4 columns, but found {len(df_clean.columns)}")
    except ValueError as e:
        print(f"Territory Schema Error: {e}")

    # 2. Drop duplicates and rows missing BOTH Country and Region
    df_clean = df_clean.drop_duplicates()
    df_clean = df_clean.dropna(subset=['Country', 'Region'], how='all')

    # 3 & 4. Strip spaces and Transform into Categories
    geo_cols = ['Region', 'Country', 'Continent']
    try:
        for col in geo_cols:
            if col in df_clean.columns:
                # Strip spaces first while it's still a string
                df_clean[col] = df_clean[col].astype(str).str.strip()
                # Then transform to category
                df_clean[col] = df_clean[col].astype('category')
    except Exception as e:
        print(f"Territory Transformation Error: {e}")

    return df_clean


def clean_sales(df):
    """
    Cleans Sales: Aggregates duplicates, removes future-dated rows, 
    and filters outliers.
    """
    # Record initial count for validation
    initial_row_count = len(df)
    df_clean = df.copy()

    # 1. Date Parsing (Apply to both OrderDate and StockDate)
    # format='mixed' allows pandas to switch between DMY and MDY automatically
    df_clean['OrderDate'] = pd.to_datetime(df_clean['OrderDate'], dayfirst=True, format='mixed', errors='coerce')
    df_clean['StockDate'] = pd.to_datetime(df_clean['StockDate'], dayfirst=True, format='mixed', errors='coerce')
    
    # --- TEMPORAL GUARDRAIL ---
    proxy_max_date = pd.Timestamp('2017-06-30') 
    df_clean = df_clean[df_clean['OrderDate'] <= proxy_max_date]
    
    # 2. Drop NaNs and Strip Whitespace
    critical_cols = ['OrderDate', 'OrderNumber', 'ProductKey', 'CustomerKey', 'OrderQuantity']
    df_clean = df_clean.dropna(subset=critical_cols)
    df_clean['OrderNumber'] = df_clean['OrderNumber'].astype(str).str.strip()

    # 3. Aggregated De-duplication
    group_cols = ['OrderDate', 'OrderNumber', 'ProductKey', 'CustomerKey', 'TerritoryKey']
    df_clean = df_clean.groupby(group_cols, as_index=False).agg({
        'OrderQuantity': 'sum',
        'StockDate': 'first',      
        'OrderLineItem': 'first'
    })

    # 4. Outlier Detection
    if not df_clean.empty:
        q_high = df_clean['OrderQuantity'].quantile(0.99)
        df_clean = df_clean[df_clean['OrderQuantity'] <= q_high]

    # 5. Final Formatting (Format both dates back to DD-MM-YYYY strings)
    df_clean = df_clean.sort_values(by='OrderDate')
    df_clean['OrderQuantity'] = df_clean['OrderQuantity'].astype(int)
    
    # Apply format to both columns
    df_clean['OrderDate'] = df_clean['OrderDate'].dt.strftime('%d-%m-%Y')
    df_clean['StockDate'] = df_clean['StockDate'].dt.strftime('%d-%m-%Y')

    # --- FINAL INTEGRITY CHECK ---
    if len(df_clean) > initial_row_count:
        print(f"🚨 CRITICAL ERROR: Row count increased from {initial_row_count} to {len(df_clean)}.")
    
    return df_clean
def clean_returns(df):
    """
    Cleans Returns data: validates schema, handles outliers, 
    and synchronizes date formats.
    """
    df_clean = df.copy()

    # 1. Column Count Validation
    try:
        if len(df_clean.columns) != 4:
            raise ValueError(f"Expected 4 columns, but found {len(df_clean.columns)}")
    except ValueError as e:
        print(f"Returns Schema Error: {e}")

    # 2. Drop duplicates and critical NaNs
    critical_cols = ['ReturnQuantity', 'ReturnDate', 'ProductKey']
    df_clean = df_clean.drop_duplicates()
    df_clean = df_clean.dropna(subset=critical_cols)

    # 3. Ensure 'ReturnQuantity' is int
    df_clean['ReturnQuantity'] = df_clean['ReturnQuantity'].astype(int)

    # 4. Outlier Detection: 99% Distribution Filter
    if not df_clean.empty:
        q_high = df_clean['ReturnQuantity'].quantile(0.99)
        df_clean = df_clean[df_clean['ReturnQuantity'] <= q_high]

    # 5. Transform 'ReturnDate' to 'day-month-year'
    try:
        # Flexible parsing as established
        df_clean['ReturnDate'] = pd.to_datetime(
            df_clean['ReturnDate'], 
            dayfirst=True, 
            format='mixed', 
            errors='coerce'
        )
        # Remove any rows that failed parsing
        df_clean = df_clean.dropna(subset=['ReturnDate'])
        # Final string formatting
        df_clean['ReturnDate'] = df_clean['ReturnDate'].dt.strftime('%d-%m-%Y')
    except Exception as e:
        print(f"Returns Date Error: {e}")

    return df_clean


def clean_price_elasticity(df):
    """
    Cleans the price elasticity raw data by removing noise and ensuring 
    correct data types for regression analysis.
    """
    # Remove missing values and duplicates
    df = df.dropna().drop_duplicates()
    
    # Ensure numeric types for modeling
    df['ProductPrice'] = pd.to_numeric(df['ProductPrice'], errors='coerce')
    df['OrderQuantity'] = pd.to_numeric(df['OrderQuantity'], errors='coerce')
    
    return df.dropna()
