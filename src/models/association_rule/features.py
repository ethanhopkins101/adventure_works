import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

def create_basket(df):
    c = df.groupby('OrderNumber')['SubcategoryName'].apply(list)
    
    te = TransactionEncoder()
    basket_array = te.fit_transform(c)
    
    basket = pd.DataFrame(basket_array, columns=te.columns_)
    
    return basket