# src/model/price_elasticity/predictions.py
import pandas as pd
import os
import joblib
from plotnine import *
import matplotlib.pyplot as plt

def generate_optimization_plots(all_gam_results, best_profit_df):
    """
    Generates and saves Price vs. Profit optimization curves.
    """
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../pictures/price_elasticity'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plot = (
        ggplot(all_gam_results, aes(x='ProductPrice', y='profit_pred_0.5', color='CategoryName'))
        + geom_ribbon(aes(ymax='profit_pred_0.975', ymin='profit_pred_0.025'), 
                      fill="#d3d3d3", alpha=0.5, color=None, show_legend=False)
        + geom_line(size=1)
        + geom_point(data=best_profit_df, color="red", size=3)
        + facet_wrap('~CategoryName', scales='free')
        + labs(
            title="Category Price vs. Profit Optimization",
            subtitle="Optimal Price (Red) vs. 95% Confidence Interval",
            x="Price ($)",
            y="Predicted Profit ($)"
        )
        + theme_minimal()
        + theme(figure_size=(10, 7))
    )
    
    plot_path = os.path.join(output_dir, 'profit_optimization.png')
    plot.save(plot_path, width=10, height=7, dpi=300)
    return plot_path

def generate_performance_csv(df, models_path):
    """
    Runs stored models to generate the final comparison table.
    """
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/models/price_elasticity'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Baseline Reference (Locked to No Promo Median)
    baseline = df[df['event'] == 'No Promo'].groupby('CategoryName').agg({
        'ProductPrice': 'median',
        'OrderQuantity': 'median',
        'profit': 'median'
    }).reset_index()
    baseline.columns = ['Item Name', 'Orig Price', 'Orig Qty', 'Orig Profit']

    all_results = []
    unique_items = df['CategoryName'].unique()

    # 2. Prediction Loop using saved models
    for item in unique_items:
        # Load specific model and encoder for the item
        promo_model_path = os.path.join(models_path, f'promo_{item}.pkl')
        le_path = os.path.join(models_path, f'le_{item}.pkl')
        
        if not os.path.exists(promo_model_path):
            continue
            
        gam = joblib.load(promo_model_path)
        le = joblib.load(le_path)
        
        item_data = df[df['CategoryName'] == item].copy()
        event_summary = item_data.groupby('event').agg({'ProductPrice': 'median'}).reset_index()
        
        X_pred = pd.DataFrame({
            'price': event_summary['ProductPrice'],
            'event_enc': le.transform(event_summary['event'])
        })
        
        # Calculate Expected Metrics
        event_summary['Exp Qty'] = gam.predict(X_pred)
        event_summary['Exp Revenue'] = event_summary['ProductPrice'] * event_summary['Exp Qty']
        
        unit_cost = (item_data['ProductPrice'] - (item_data['profit'] / item_data['OrderQuantity'])).median()
        event_summary['Exp Profit'] = (event_summary['ProductPrice'] - unit_cost) * event_summary['Exp Qty']
        event_summary['Item Name'] = item
        
        all_results.append(event_summary)

    # 3. Final Formatting
    final_table = pd.merge(pd.concat(all_results), baseline, on='Item Name', how='left')
    final_table = final_table.rename(columns={'ProductPrice': 'Adjusted Price', 'event': 'Event'})
    
    final_table = final_table[[
        'Item Name', 'Orig Price', 'Orig Qty', 'Orig Profit',
        'Adjusted Price', 'Exp Qty', 'Exp Revenue', 'Exp Profit', 'Event'
    ]]
    
    final_table['is_promo'] = final_table['Event'] != 'No Promo'
    final_table = final_table.sort_values(['Item Name', 'is_promo', 'Event']).drop(columns=['is_promo'])
    
    csv_path = os.path.join(output_dir, 'final_performance_table.csv')
    final_table.to_csv(csv_path, index=False)
    return csv_path