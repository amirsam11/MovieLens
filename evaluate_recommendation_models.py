# evaluate_recommendation_models.py

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load test data
try:
    test_data = pd.read_csv('test_data.csv')
    actual_ratings = test_data['rating'].values  # True ratings from the test set
    print("Test data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: File not found. Please ensure 'test_data.csv' exists. {e}")
    exit()
except Exception as e:
    print(f"An error occurred while loading the test data: {e}")
    exit()

# Function to calculate MAE and RMSE
def calculate_metrics(actual_ratings, predictions):
    """
    Calculate MAE and RMSE for the given predictions and actual ratings.
    
    Parameters:
        actual_ratings (list or np.array): List of actual ratings.
        predictions (list or np.array): List of predicted ratings.
    
    Returns:
        tuple: MAE and RMSE values.
    """
    if len(actual_ratings) != len(predictions):
        print("Error: Length of actual ratings and predictions do not match.")
        return None, None
    
    mae = mean_absolute_error(actual_ratings, predictions)
    rmse = mean_squared_error(actual_ratings, predictions, squared=False)
    return mae, rmse

# Evaluate Co-Clustering Model
def evaluate_coclustering():
    """
    Evaluate the Co-Clustering model by loading its predictions and calculating MAE and RMSE.
    """
    try:
        coclustering_predictions = pd.read_csv('coclustering_predictions.csv')['prediction'].values
        mae, rmse = calculate_metrics(actual_ratings, coclustering_predictions)
        if mae is not None and rmse is not None:
            print(f"Co-Clustering Model - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    except FileNotFoundError:
        print("Error: 'coclustering_predictions.csv' not found. Please ensure it exists.")
    except Exception as e:
        print(f"An error occurred while evaluating Co-Clustering model: {e}")

# Evaluate KNN Model
def evaluate_knn():
    """
    Evaluate the KNN model by loading its predictions and calculating MAE and RMSE.
    """
    try:
        knn_predictions = pd.read_csv('knn_predictions.csv')['prediction'].values
        mae, rmse = calculate_metrics(actual_ratings, knn_predictions)
        if mae is not None and rmse is not None:
            print(f"KNN Model - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    except FileNotFoundError:
        print("Error: 'knn_predictions.csv' not found. Please ensure it exists.")
    except Exception as e:
        print(f"An error occurred while evaluating KNN model: {e}")

# Evaluate ExtKNNCF Model
def evaluate_ext_knn_cf():
    """
    Evaluate the ExtKNNCF model by loading its predictions and calculating MAE and RMSE.
    """
    try:
        ext_knn_cf_predictions = pd.read_csv('ext_knn_cf_predictions.csv')['prediction'].values
        mae, rmse = calculate_metrics(actual_ratings, ext_knn_cf_predictions)
        if mae is not None and rmse is not None:
            print(f"ExtKNNCF Model - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    except FileNotFoundError:
        print("Error: 'ext_knn_cf_predictions.csv' not found. Please ensure it exists.")
    except Exception as e:
        print(f"An error occurred while evaluating ExtKNNCF model: {e}")

if __name__ == "__main__":
    print("Starting evaluation of recommendation models...")
    
    # Evaluate Co-Clustering Model
    evaluate_coclustering()
    
    # Evaluate KNN Model
    evaluate_knn()
    
    # Evaluate ExtKNNCF Model
    evaluate_ext_knn_cf()
    
    print("Evaluation completed.")
