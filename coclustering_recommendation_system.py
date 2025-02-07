# coclustering_recommendation_system.py

from sklearn.cluster import SpectralCoclustering
import pandas as pd
import numpy as np

def create_user_item_matrix(data):
    """
    Create a user-item matrix from the dataset.
    
    Parameters:
        data (pd.DataFrame): Input dataset with columns ['user_id', 'item_id', 'rating'].
    
    Returns:
        pd.DataFrame: User-item matrix filled with ratings, missing values replaced with 0.
    """
    try:
        # Create a pivot table to form the user-item matrix
        user_item_matrix = pd.pivot_table(data, values='rating', index='user_id', columns='item_id')
        
        # Fill missing values with 0 (unrated items)
        user_item_matrix = user_item_matrix.fillna(0)
        
        print("User-item matrix created successfully.")
        return user_item_matrix
    except Exception as e:
        print(f"An error occurred while creating the user-item matrix: {e}")
        return None

def train_coclustering_model(user_item_matrix, n_clusters=5, random_state=42):
    """
    Train a Co-Clustering model using SpectralCoclustering.
    
    Parameters:
        user_item_matrix (pd.DataFrame): User-item matrix.
        n_clusters (int): Number of clusters for users and items.
        random_state (int): Seed for reproducibility.
    
    Returns:
        SpectralCoclustering: Trained Co-Clustering model.
    """
    if user_item_matrix is None:
        return None
    
    try:
        # Initialize the SpectralCoclustering model
        model_co = SpectralCoclustering(n_clusters=n_clusters, random_state=random_state)
        
        # Fit the model to the user-item matrix
        model_co.fit(user_item_matrix.values)
        
        print("Co-Clustering model trained successfully.")
        return model_co
    except Exception as e:
        print(f"An error occurred while training the Co-Clustering model: {e}")
        return None

def get_clusters(model_co, user_item_matrix):
    """
    Extract user and item clusters from the trained Co-Clustering model.
    
    Parameters:
        model_co (SpectralCoclustering): Trained Co-Clustering model.
        user_item_matrix (pd.DataFrame): User-item matrix.
    
    Returns:
        tuple: User clusters and item clusters as NumPy arrays.
    """
    if model_co is None or user_item_matrix is None:
        return None, None
    
    try:
        # Get user clusters (row labels) and item clusters (column labels)
        user_cluster = model_co.row_labels_
        item_cluster = model_co.column_labels_
        
        print("Clusters extracted successfully.")
        return user_cluster, item_cluster
    except Exception as e:
        print(f"An error occurred while extracting clusters: {e}")
        return None, None
def generate_coclustering_predictions(model_co, user_item_matrix, test_data):
    """
    Generate predictions for the test set using the Co-Clustering model.
    
    Parameters:
        model_co (SpectralCoclustering): Trained Co-Clustering model.
        user_item_matrix (pd.DataFrame): User-item matrix.
        test_data (pd.DataFrame): Test dataset with columns ['user_id', 'item_id'].
    
    Returns:
        list: Predicted ratings for the test set.
    """
    try:
        # Map test user-item pairs to row and column indices
        user_indices = []
        item_indices = []
        unknown_pairs = []

        for _, row in test_data.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']

            if user_id in user_item_matrix.index and item_id in user_item_matrix.columns:
                user_indices.append(user_item_matrix.index.get_loc(user_id))
                item_indices.append(user_item_matrix.columns.get_loc(item_id))
            else:
                unknown_pairs.append((user_id, item_id))

        # Extract average ratings for co-clusters
        row_means = user_item_matrix.values.mean(axis=1)
        col_means = user_item_matrix.values.mean(axis=0)
        overall_mean = user_item_matrix.values.mean()

        predictions = []
        for u, i in zip(user_indices, item_indices):
            cluster_u = model_co.row_labels_[u]
            cluster_i = model_co.column_labels_[i]

            # Calculate cluster mean
            cluster_values = user_item_matrix.values[model_co.row_labels_ == cluster_u, :][:, model_co.column_labels_ == cluster_i]
            cluster_mean = cluster_values.mean() if cluster_values.size > 0 else overall_mean

            predictions.append(cluster_mean)

        # Handle unknown user-item pairs
        for user_id, item_id in unknown_pairs:
            predictions.append(overall_mean)  # Assign global mean as default prediction

        print("Co-Clustering predictions generated successfully.")
        return predictions
    except Exception as e:
        print(f"An error occurred while generating Co-Clustering predictions: {e}")
        return []
if __name__ == "__main__":
    try:
        # Step 1: Load the preprocessed training data
        train_data = pd.read_csv('train_data.csv')
        print("Training data loaded successfully.")
        
        # Step 2: Create the user-item matrix
        user_item_matrix = create_user_item_matrix(train_data)
        if user_item_matrix is None:
            print("Error: Failed to create the user-item matrix.")
            exit()

        # Step 3: Train the Co-Clustering model
        model_co = train_coclustering_model(user_item_matrix, n_clusters=5)
        if model_co is None:
            print("Error: Failed to train the Co-Clustering model.")
            exit()

        # Step 4: Generate clusters
        user_cluster, item_cluster = get_clusters(model_co, user_item_matrix)
        if user_cluster is None or item_cluster is None:
            print("Error: Failed to generate user or item clusters.")
            exit()

        # Print cluster results
        print(f"User clusters: {user_cluster}")
        print(f"Item clusters: {item_cluster}")

        # Step 5: Load test data
        test_data = pd.read_csv('test_data.csv')
        print("Test data loaded successfully.")

        # Step 6: Generate predictions for the test set
        coclustering_predictions = generate_coclustering_predictions(model_co, user_item_matrix, test_data[['user_id', 'item_id']])
        if coclustering_predictions is None:
            print("Error: Failed to generate Co-Clustering predictions.")
            exit()

        # Step 7: Save predictions to a file
        pd.DataFrame({'prediction': coclustering_predictions}).to_csv('coclustering_predictions.csv', index=False)
        print("Co-Clustering predictions saved to 'coclustering_predictions.csv'.")

    except FileNotFoundError as fnf_error:
        print(f"Error: File not found. Please ensure all required files exist. {fnf_error}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
