import pandas as pd
import joblib
import os

def predict_significant_rules(rules_df=None, min_conf=0.3, min_lift=1.0):
    # 1. Logic to use provided df OR load from absolute path
    if rules_df is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Anchor to project root and target the saved pickle
        model_path = os.path.normpath(os.path.join(current_dir, '../../../models/association_rule/rules_model.pkl'))
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Run train.py first.")
            
        rules_df = joblib.load(model_path)
    
    # 2. Filtering for high Confidence and Lift
    significant_rules = rules_df[
        (rules_df['confidence'] > min_conf) & 
        (rules_df['lift'] > min_lift)
    ].copy()

    # 3. Remove redundant 'swapped' rules
    significant_rules['subset'] = significant_rules.apply(
        lambda x: frozenset(x['antecedents'] | x['consequents']), axis=1
    )
    significant_rules = significant_rules.drop_duplicates(subset=['subset'])

    # 4. Final cleanup
    significant_rules = significant_rules.drop(columns=['subset'])
    
    return significant_rules