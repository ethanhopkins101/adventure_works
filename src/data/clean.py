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
        # 'format="mixed"' tells pandas to guess the format for EACH row 
        # 'dayfirst=True' prioritizes the European/International style
        df_clean['Date'] = pd.to_datetime(
            df_clean['Date'], 
            dayfirst=True, 
            format='mixed', 
            errors='coerce'
        )

        # Drop rows that are truly garbage (e.g. '01-2023-02' which is non-standard)
        df_clean = df_clean.dropna(subset=['Date'])

        # 3. Set DateTimeIndex
        df_clean.set_index('Date', inplace=True)
        
        # 4. Final Formatting to 'DD-MM-YYYY'
        df_clean.index = df_clean.index.strftime('%d-%m-%Y')
        
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
    Cleans Sales data: handles outliers, enforces schema, 
    and validates relational keys.
    """
    df_clean = df.copy()

    # 1. Column Count Validation
    try:
        if len(df_clean.columns) != 8:
            raise ValueError(f"Expected 8 columns, but found {len(df_clean.columns)}")
    except ValueError as e:
        print(f"Sales Schema Error: {e}")

    # 2. Drop duplicates and NaNs in critical columns
    critical_cols = ['OrderDate', 'OrderNumber', 'ProductKey', 'CustomerKey', 'OrderQuantity']
    df_clean = df_clean.drop_duplicates()
    df_clean = df_clean.dropna(subset=critical_cols)

    # 3. Ensure 'OrderQuantity' is int
    df_clean['OrderQuantity'] = df_clean['OrderQuantity'].astype(int)

    # 4. Outlier Detection: 99% Distribution Filter
    q_high = df_clean['OrderQuantity'].quantile(0.99)
    df_clean = df_clean[df_clean['OrderQuantity'] <= q_high]

    # 5. Transform Dates to 'day-month-year'
    date_cols = ['OrderDate', 'StockDate']
    for col in date_cols:
        try:
            # Flexible parsing like we did for Calendar
            df_clean[col] = pd.to_datetime(df_clean[col], dayfirst=True, format='mixed', errors='coerce')
            df_clean = df_clean.dropna(subset=[col])
            df_clean[col] = df_clean[col].dt.strftime('%d-%m-%Y')
        except Exception as e:
            print(f"Sales Date Error in {col}: {e}")

    # 6. Strip spaces from 'OrderNumber'
    df_clean['OrderNumber'] = df_clean['OrderNumber'].astype(str).str.strip()

    return df_clean