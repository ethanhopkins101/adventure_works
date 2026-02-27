import pandas as pd
import json
import os

def sync_subcategory_encoding(current_df):
    """
    Ensures persistent encoding for subcategories. 
    Saves mapping to ../../json_files/encoder.json
    """
    # 1. Path setup
    base_dir = os.path.dirname(__file__)
    json_dir = os.path.abspath(os.path.join(base_dir, '../../json_files/encoder'))
    mapping_path = os.path.join(json_dir, 'encoder.json')
    subcat_path = os.path.join(base_dir, '../../data/cleaned/Cleaned_Product_Subcategories.csv')

    # Create directory if it doesn't exist
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)

    # 2. Load existing mapping
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
    else:
        mapping = {}

    # 3. Get all subcategories
    # Load master list and combine with current data to catch any new Shopify items
    master_subcats = pd.read_csv(subcat_path)['SubcategoryName'].unique()
    current_subcats = current_df['SubcategoryName'].unique()
    all_seen_items = set(master_subcats) | set(current_subcats)

    # 4. Update mapping
    updated = False
    next_id = max(mapping.values()) + 1 if mapping else 0

    # Sort to ensure deterministic assignment for new batches
    for item in sorted(all_seen_items):
        if item not in mapping:
            mapping[item] = int(next_id)
            next_id += 1
            updated = True

    # 5. Save if updated
    if updated:
        with open(mapping_path, 'w') as f:
            json.dump(mapping, f, indent=4)

    # 6. Apply mapping
    current_df['SubcategoryEncoded'] = current_df['SubcategoryName'].map(mapping)
    
    return current_df, mapping