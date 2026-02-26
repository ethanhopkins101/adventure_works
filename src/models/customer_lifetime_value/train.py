import os
import joblib
import pandas as pd
import lifetimes as lf
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def train_lifetimes_models(df, folder_path='../../../models/customer_lifetime_value'):
    """
    Fits BG/NBD and Gamma-Gamma models and saves them to disk.
    """
    # 1. Fit BG/NBD
    bgf = lf.BetaGeoFitter(penalizer_coef=0.1)
    bgf.fit(df['frequency'], df['recency'], df['T'])

    # 2. Fit Gamma-Gamma
    ggf = lf.GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(df['frequency'], df['monetary_value'])

    abs_path = os.path.abspath(folder_path)
    if not os.path.exists(abs_path):
        os.makedirs(abs_path)
        
    bgf.save_model(os.path.join(abs_path, 'bgf_model.pkl'))
    ggf.save_model(os.path.join(abs_path, 'ggf_model.pkl'))
    print(f"✅ Lifetimes models saved at: {abs_path}")
    return bgf, ggf
def train_kmeans_clustering(df, folder_path='../../../models/customer_lifetime_value'):
    """
    Standardizes data, fits K-Means with 4 clusters, and saves both scaler and model.
    """
    features = ['frequency', 'recency', 'monetary_value']
    
    # 1. Scale
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features])

    # 2. Cluster
    kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(scaled_features)

    abs_path = os.path.abspath(folder_path)
    if not os.path.exists(abs_path):
        os.makedirs(abs_path)
        
    joblib.dump(scaler, os.path.join(abs_path, 'scaler.pkl'))
    joblib.dump(kmeans, os.path.join(abs_path, 'kmeans_model.pkl'))
    print(f"✅ K-Means assets saved at: {abs_path}")
    return scaler, kmeans