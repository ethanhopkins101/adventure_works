# src/model/price_elasticity/train.py
import pandas as pd
import os
import joblib
from pygam import ExpectileGAM, GAM, s, l, f
from sklearn.preprocessing import LabelEncoder

def train_and_save_models(df):
    df['event'] = df['event'].astype(str).str.strip()
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../models/price_elasticity/'))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # --- FIX: Fit Encoder on ALL events first ---
    global_le = LabelEncoder()
    global_le.fit(df['event']) 

    # 1. Baseline Model Training (No Promo)
    baseline_data = df.query('event == "No Promo"').copy()
    for product in baseline_data['CategoryName'].unique():
        product_data = baseline_data[baseline_data['CategoryName'] == product]
        X = product_data[['ProductPrice']].values
        y = product_data['OrderQuantity'].values
        gam_baseline = ExpectileGAM(s(0), expectile=0.5).fit(X, y)
        joblib.dump(gam_baseline, os.path.join(model_dir, f'baseline_{product}.pkl'))

    # 2. Promo Model Training (Includes all events in the mapping)
    promo_data = df.dropna(subset=['ProductPrice', 'OrderQuantity', 'event']).copy()
    
    for cat in promo_data['CategoryName'].unique():
        cat_data = promo_data[promo_data['CategoryName'] == cat].copy()
        if len(cat_data) < 5 or cat_data['ProductPrice'].nunique() <= 1:
            continue

        # Use the global_le which now KNOWS what 'No Promo' is
        X = pd.DataFrame({
            'price': cat_data['ProductPrice'],
            'event': global_le.transform(cat_data['event']) 
        }).values 
        y = cat_data['OrderQuantity'].values
        
        try:
            gam_promo = GAM(l(0) + f(1)).fit(X, y)
            joblib.dump(gam_promo, os.path.join(model_dir, f'promo_{cat}.pkl'))
            joblib.dump(global_le, os.path.join(model_dir, f'le_{cat}.pkl'))
        except Exception as e:
            print(f"Error modeling {cat}: {e}")

    return True