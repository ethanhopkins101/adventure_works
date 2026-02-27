import pandas as pd
import joblib
import os

def format_set(itemset):
    """Converts frozenset to string if one item, or list of strings if more."""
    items = list(itemset)
    return items[0] if len(items) == 1 else items

def predict_significant_rules(rules_df=None, min_conf=0.3, min_lift=1.0):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    if rules_df is None:
        model_path = os.path.normpath(os.path.join(current_dir, '../../../models/association_rule/rules_model.pkl'))
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}.")
        rules_df = joblib.load(model_path)
    
    # 1. Filter
    significant_rules = rules_df[
        (rules_df['confidence'] > min_conf) & (rules_df['lift'] > min_lift)
    ].copy()

    # 2. Deduplicate
    significant_rules['subset'] = significant_rules.apply(
        lambda x: frozenset(x['antecedents'] | x['consequents']), axis=1
    )
    significant_rules = significant_rules.drop_duplicates(subset=['subset']).drop(columns=['subset'])

    # 3. Format Sets: frozenset -> string/list
    significant_rules['antecedents'] = significant_rules['antecedents'].apply(format_set)
    significant_rules['consequents'] = significant_rules['consequents'].apply(format_set)

    # 4. Round numerical values
    significant_rules = significant_rules.round(2)

    # 5. Save to CSV
    output_dir = os.path.normpath(os.path.join(current_dir, '../../../data/models/association_rule/'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    csv_path = os.path.join(output_dir, 'significant_rules.csv')
    significant_rules.to_csv(csv_path, index=False)
    
    return significant_rules