import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

# Absolute path configurations
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, '../../../models/bayesian_mmm/mmm_orbit_model.pkl')
IMAGE_DIR = os.path.join(BASE_DIR, '../../../pictures/bayesian_mmm/')
JSON_DIR = os.path.join(BASE_DIR, '../../../json_files/bayesian_mmm/')

def load_trained_model():
    """Helper to load the pickled Orbit model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please run train.py first.")
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

def get_roi_analysis(df_orig, df_adstocked):
    """
    Objective 1: Calculates Profit ROI, returns the analysis table, 
    and saves it to the specified absolute path.
    """
    model = load_trained_model()
    channels = ['tv_s', 'ooh_s', 'print_s', 'facebook_s', 'search_s']
    
    # Extract Weights
    coef_df = model.get_regression_coefs()
    weights = coef_df[coef_df['regressor'].isin(channels)].set_index('regressor')['coefficient']
    
    # Calculation Logic
    total_orig_spend = df_orig[channels].sum()
    adstocked_contribution = (df_adstocked[channels] * weights).sum()
    
    analysis_table = pd.DataFrame({
        'Total_Orig_Spend': total_orig_spend,
        'Adstocked_Profit': adstocked_contribution, 
        'New_Coefficient': weights,
        'New_ROI': adstocked_contribution / total_orig_spend
    })
    
    analysis_df = analysis_table.sort_values(by='New_ROI', ascending=False).round(2)

    # Absolute Path Logic
    DATA_PATH = os.path.join(BASE_DIR, '../../../data/models/bayesian_mmm/')
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        
    # Save the CSV
    analysis_df.to_csv(os.path.join(DATA_PATH, 'roi_analysis.csv'))
    
    return analysis_df


def generate_waterfall_chart(df_adstocked):
    """
    Objective 2: Generates and saves the contribution waterfall chart.
    """
    model = load_trained_model()
    channels = ['tv_s', 'ooh_s', 'print_s', 'facebook_s', 'search_s']
    
    # Get Weights and Predictions
    coef_df = model.get_regression_coefs()
    weights = coef_df[coef_df['regressor'].isin(channels)].set_index('regressor')['coefficient']
    predicted_df = model.predict(df=df_adstocked)
    
    # Contribution Math
    channel_contrib_matrix = df_adstocked[channels] * weights.values
    total_channel_contrib = channel_contrib_matrix.sum().sum()
    total_predicted_profit = predicted_df['prediction'].sum()
    baseline_organic = total_predicted_profit - total_channel_contrib
    
    labels = ['Baseline (Organic)'] + channels
    values = [baseline_organic] + list(channel_contrib_matrix.sum().values)
    
    # Plotting
    plt.figure(figsize=(14, 7))
    cumulative = 0
    colors = plt.cm.get_cmap('tab10', len(labels))

    for i, (label, val) in enumerate(zip(labels, values)):
        plt.bar(label, val, bottom=cumulative, color=colors(i), edgecolor='black', alpha=0.8)
        label_pos = cumulative + val + (max(values) * 0.02)
        plt.text(i, label_pos, f'${val:,.0f}', ha='center', va='bottom', fontweight='bold')
        if i > 0:
            plt.plot([i-1, i], [cumulative, cumulative], color='gray', linestyle='--', linewidth=1)
        cumulative += val

    plt.title("Profit Contribution Breakdown (Adstocked Model)", fontsize=16, pad=20)
    plt.ylabel("Total Profit ($)", fontsize=12)
    plt.ylim(0, cumulative * 1.15)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save Image
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
    plt.savefig(os.path.join(IMAGE_DIR, 'profit_waterfall.png'))
    plt.close()

def run_budget_simulations(budgets=[15000, 20000]):
    """
    Objective 3: Runs strategy simulations and saves to JSON.
    """
    # Using the ROIs derived from your adstocked model
    # Order: facebook_s, tv_s, ooh_s, print_s, search_s
    rois = [1.77, 0.90, 0.54, 0.21, 0.04]
    channels = ['facebook_s', 'tv_s', 'ooh_s', 'print_s', 'search_s']
    
    simulations = {}
    
    for total_budget in budgets:
        # Optimization Logic
        fb_cap = 0.55 if total_budget <= 15000 else 0.50
        tv_cap = 0.30 if total_budget <= 15000 else 0.35
        
        fb_spend = total_budget * fb_cap
        other_min = total_budget * 0.05
        # Remaining goes to TV
        tv_spend = total_budget - fb_spend - (other_min * 3)
        
        allocations = [fb_spend, tv_spend, other_min, other_min, other_min]
        
        df_strat = pd.DataFrame({
            'Channel': channels,
            'Budget': allocations,
            'ROI': rois
        })
        df_strat['Expected_Profit'] = df_strat['Budget'] * df_strat['ROI']
        
        # Convert to dictionary for JSON
        simulations[f"budget_{total_budget}"] = df_strat.to_dict(orient='records')

    # Save JSON
    if not os.path.exists(JSON_DIR):
        os.makedirs(JSON_DIR)
    with open(os.path.join(JSON_DIR, 'budget_simulations.json'), 'w') as f:
        json.dump(simulations, f, indent=4)
        
    return simulations