import os
import pickle
from orbit.models import DLT

def train_and_save_model(df):
    """
    Initializes the DLT model, fits it using profit as the response,
    and saves the model object to the absolute path.
    """
    # 1. Initialize Orbit DLT Model
    # Using 'profit' as the response and excluding 'trend' from regressors 
    # as DLT handles the trend component internally.
    mmm_orbit = DLT(
        response_col='profit',
        date_col='date',
        regressor_col=['tv_s', 'ooh_s', 'print_s', 'facebook_s', 'search_s'],
        seasonality=12, 
        seed=888,
        estimator='stan-map',
        n_bootstrap_draws=1000,
        prediction_percentiles=[5, 95]
    )

    # 2. Fit the model
    mmm_orbit.fit(df=df)

    # 3. Define the absolute path and ensure directory exists
    # Path relative to src/models/bayesian_mmm/train.py
    base_path = os.path.dirname(__file__)
    save_path = os.path.join(base_path, '../../../models/bayesian_mmm/')
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 4. Save the model using pickle
    model_filename = os.path.join(save_path, 'mmm_orbit_model.pkl')
    with open(model_filename, 'wb') as f:
        pickle.dump(mmm_orbit, f)
    
    return model_filename