# evaluate_models.py

from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np

def calculate_metrics(actual_ratings, predictions):
    """
    Calculate MAE and RMSE metrics.
    
    Parameters:
        actual_ratings (list): List of actual ratings.
        predictions (list): List of predicted ratings.
    
    Returns:
        tuple: MAE and RMSE values.
    """
    if len(actual_ratings) != len(predictions):
        raise ValueError("Length of actual ratings and predictions must be the same.")
    
    mae = mean_absolute_error(actual_ratings, predictions)
    
    # Calculate RMSE manually if 'squared=False' is not supported
    mse = mean_squared_error(actual_ratings, predictions)
    rmse = np.sqrt(mse)  # Compute the square root of MSE to get RMSE
    
    return mae, rmse

def evaluate_model(model_name):
    """
    Evaluate a specific model by loading its predictions and calculating MAE and RMSE.
    
    Parameters:
        model_name (str): Name of the model (e.g., 'coclustering', 'knn', 'ext_knn_cf').
    
    Returns:
        dict: Dictionary containing MAE and RMSE for the model.
    """
    try:
        # Load test data
        test_data = pd.read_csv('test_data.csv')
        actual_ratings = test_data['rating'].values
        
        # Load predictions for the specified model
        predictions_file = f"{model_name}_predictions.csv"
        predictions = pd.read_csv(predictions_file)['prediction'].values
        
        # Calculate metrics
        mae, rmse = calculate_metrics(actual_ratings, predictions)
        print(f"{model_name.capitalize()} Model - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        return {model_name: {'MAE': mae, 'RMSE': rmse}}
    except FileNotFoundError as e:
        print(f"Error: File not found. {e}")
        return {}
    except Exception as e:
        print(f"An error occurred while evaluating {model_name} model: {e}")
        return {}

if __name__ == "__main__":
    # List of models to evaluate
    models = ['coclustering', 'knn', 'ext_knn_cf']
    results = {}

    # Evaluate each model
    for model in models:
        model_results = evaluate_model(model)
        results.update(model_results)

    # Print final results
    print("\nFinal Evaluation Results:")
    for model, metrics in results.items():
        print(f"{model.capitalize()} - MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}")
