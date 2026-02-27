"""
WEEKLY RETRAINING & MODEL ROTATION SCRIPT
=========================================
Frequency: Run once a week (via GitHub Actions).

STORY:
To prevent model drift and ensure recommendations are based on the latest 
AdventureWorks data, this script automates the transition between training cycles.

LOGIC:
1. ARCHIVE: It visits every folder in /src/models/. 
2. ROTATE: Existing backups in 'previous_models/' are shifted (prev_1 -> prev_2, etc.)
   to maintain a chronological history.
3. CLEAR: Current trained models (.pkl, .joblib, etc.) are renamed with '_prev_1' 
   and moved to the 'previous_models/' subfolder.
4. RETRAIN: With the main model directories empty, it triggers run_main_pipeline().
   The pipelines, finding no existing models, are forced to retrain from scratch 
   using the most recent cleaned data.
"""

import os
import shutil
import re
from main import run_main_pipeline

def rotate_and_retrain():
    # 1. Configuration
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_root = os.path.join(base_dir, 'models')
    
    # Extensions to move (add any other model formats you use)
    model_extensions = ('.pkl', '.pth', '.h5', '.joblib')

    print("🔄 Starting Weekly Model Rotation...")

    # 2. Iterate through each model folder
    for model_name in os.listdir(models_root):
        model_path = os.path.join(models_root, model_name)
        
        if os.path.isdir(model_path):
            backup_dir = os.path.join(model_path, 'previous_models')
            os.makedirs(backup_dir, exist_ok=True)

            # --- Step A: Shift existing backups (prev_1 -> prev_2) ---
            existing_backups = sorted(os.listdir(backup_dir), reverse=True)
            for f in existing_backups:
                # Match pattern: name_prev_N.ext
                match = re.search(r'_prev_(\d+)', f)
                if match:
                    current_num = int(match.group(1))
                    new_name = f.replace(f'_prev_{current_num}', f'_prev_{current_num + 1}')
                    os.rename(os.path.join(backup_dir, f), os.path.join(backup_dir, new_name))

            # --- Step B: Move current models to backup as prev_1 ---
            for file in os.listdir(model_path):
                if file.endswith(model_extensions):
                    src_file = os.path.join(model_path, file)
                    
                    # Construct new name: filename_prev_1.ext
                    name_part, ext_part = os.path.splitext(file)
                    dst_file = os.path.join(backup_dir, f"{name_part}_prev_1{ext_part}")
                    
                    # Move (this automatically deletes the original from the model folder)
                    shutil.move(src_file, dst_file)
                    print(f"📦 Backed up: {model_name}/{file} -> previous_models/")

    # 3. Trigger Retraining
    print("\n🚀 Model folders cleared. Triggering main.py for retraining...")
    run_main_pipeline()

if __name__ == "__main__":
    rotate_and_retrain()