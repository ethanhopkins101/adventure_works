import pandas as pd
import joblib
import os
from mlxtend.frequent_patterns import apriori, association_rules

def train_association_model(basket):
    # 1. Generate frequent itemsets
    fi = apriori(basket, min_support=0.01, use_colnames=True)
    
    # 2. Generate association rules
    ar = association_rules(fi, metric='lift', min_threshold=1)
    
    # 3. Clean and sort the rules
    ar = ar.iloc[:, :7]
    ar = ar.sort_values('lift', ascending=False)
    
    # 4. Save the model using Absolute Path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up 3 levels to adventure_works root, then into models/association_rule
    output_dir = os.path.normpath(os.path.join(current_dir, '../../../models/association_rule'))
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    model_path = os.path.join(output_dir, 'rules_model.pkl')
    joblib.dump(ar, model_path)
    
    return ar