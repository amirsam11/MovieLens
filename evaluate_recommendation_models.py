# evaluate_recommendation_models.py

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd

def calculate_ndcg(actual, predicted, k=5):
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG) for top-k recommendations.
    
    Parameters:
        actual (list): List of actual ratings.
        predicted (list): List of predicted ratings.
        k (int): Number of top recommendations to consider.
    
    Returns:
        float: NDCG score.
    """
    dcg = 0
    idcg = 0
    sorted_actual = sorted(actual, reverse=True)[:k]
    
    for i, (a, p) in enumerate(zip(actual, predicted)):
        if i < k:
            dcg += (2 ** a - 1) / np.log2(i + 2)
            idcg += (2 ** sorted_actual[i] - 1) / np.log2(i + 2)
    
    return dcg / idcg if idcg != 0 else 0

def evaluate_model(model_name, actual_ratings, predictions):
    """
    Evaluate a model using MAE, RMSE, and NDCG metrics.
    
    Parameters:
        model_name (str): Name of the model.
        actual_ratings (list): List of actual ratings.
        predictions (list): List of predicted ratings.
    
    Returns:
        dict: Evaluation results for the model.
    """
    mae = mean_absolute_error(actual_ratings, predictions)
    mse = mean_squared_error(actual_ratings, predictions)
    rmse = np.sqrt(mse)
    ndcg = calculate_ndcg(actual_ratings, predictions)
    
    print(f"{model_name.capitalize()} Model - MAE: {mae:.4f}, RMSE: {rmse:.4f}, NDCG: {ndcg:.4f}")
    return {model_name: {'MAE': mae, 'RMSE': rmse, 'NDCG': ndcg}}

if __name__ == "__main__":
    models = ['coclustering', 'knn', 'ext_knn_cf']
    results = {}

    for model in models:
        try:
            # Load test data
            test_data = pd.read_csv('test_data.csv')
            actual_ratings = test_data['rating'].values
            
            # Load predictions
            predictions_file = f"{model}_predictions.csv"
            predictions = pd.read_csv(predictions_file)['prediction'].values
            
            # Evaluate the model
            model_results = evaluate_model(model, actual_ratings, predictions)
            results.update(model_results)
        except FileNotFoundError as e:
            print(f"Error: File not found for {model}. {e}")
        except Exception as e:
            print(f"An error occurred while evaluating {model}: {e}")

    # Print final results
    print("\nFinal Evaluation Results:")
    for model, metrics in results.items():
        print(f"{model.capitalize()} - MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}, NDCG: {metrics['NDCG']:.4f}")
