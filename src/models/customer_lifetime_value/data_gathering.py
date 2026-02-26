import pandas as pd

def get_clv_prepared_data(sales_path, products_path):
    """
    Extracts sales and product data to calculate profit per transaction.
    Returns a cleaned DataFrame for CLV modeling.
    """
    # 1. Load the cleaned datasets
    df_sales = pd.read_csv(sales_path)
    df_products = pd.read_csv(products_path)

    # 2. Extract specific columns and join tables
    df_clv_raw = df_sales[['OrderDate', 'CustomerKey', 'OrderQuantity', 'ProductKey']].merge(
        df_products[['ProductKey', 'ProductCost', 'ProductPrice']], 
        on='ProductKey', 
        how='left'
    )

    # 3. Obtain Profit (Monetary Value)
    df_clv_raw['Profit'] = (df_clv_raw['ProductPrice'] - df_clv_raw['ProductCost']) * df_clv_raw['OrderQuantity']

    # 4. Final selection and cleaning
    df_final = df_clv_raw[['OrderDate', 'CustomerKey', 'Profit']].dropna()
    
    return df_final

def get_purchase_probability_data(sales_path, products_path, subcats_path):
    """
    Calculates the purchase probability for the top 3 subcategories per customer.
    Returns a panel-format DataFrame.
    """
    # 1. Load datasets
    sales = pd.read_csv(sales_path, usecols=['ProductKey', 'CustomerKey'])
    products = pd.read_csv(products_path, usecols=['ProductKey', 'ProductSubcategoryKey'])
    subcats = pd.read_csv(subcats_path, usecols=['ProductSubcategoryKey', 'SubcategoryName'])

    # 2. Multi-stage Merge
    df = (sales.merge(products, on='ProductKey', how='left')
               .merge(subcats, on='ProductSubcategoryKey', how='left'))

    # 3. Frequency calculations
    customer_item_counts = df.groupby(['CustomerKey', 'SubcategoryName']).size().reset_index(name='item_count')
    customer_totals = df.groupby('CustomerKey').size().reset_index(name='total_purchases')

    # 4. Probability calculation
    purchase_probability = customer_item_counts.merge(customer_totals, on='CustomerKey')
    purchase_probability['probability of purchase'] = (
        purchase_probability['item_count'] / purchase_probability['total_purchases']
    )

    # 5. Filter for top 3 (Panel Format)
    purchase_probability = (
        purchase_probability.sort_values(['CustomerKey', 'item_count'], ascending=[True, False])
        .groupby('CustomerKey')
        .head(3)
        .reset_index(drop=True)
    )

    # 6. Cleanup
    final_cols = ['CustomerKey', 'SubcategoryName', 'probability of purchase']
    return purchase_probability[final_cols].dropna()