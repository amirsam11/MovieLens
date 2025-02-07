# knn_recommendation_system.py

from sklearn.neighbors import NearestNeighbors
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
        print(f"Error while creating the user-item matrix: {e}")
        return None


def train_knn_model(user_item_matrix, metric='cosine', algorithm='brute'):
    """
    Train a KNN model using the cosine similarity metric.
    
    Parameters:
        user_item_matrix (pd.DataFrame): User-item matrix.
        metric (str): Distance metric for KNN (default is 'cosine').
        algorithm (str): Algorithm used for KNN (default is 'brute').
    
    Returns:
        NearestNeighbors: Trained KNN model.
    """
    if user_item_matrix is None:
        print("Error: User-item matrix is not available.")
        return None
    
    try:
        # Initialize and fit the KNN model
        model_knn = NearestNeighbors(metric=metric, algorithm=algorithm)
        model_knn.fit(user_item_matrix.values)
        print("KNN model trained successfully.")
        return model_knn
    except Exception as e:
        print(f"Error while training the KNN model: {e}")
        return None


def find_similar_users(model_knn, user_item_matrix, user_id, n_neighbors=5):
    """
    Find the top N similar users for a given user based on cosine similarity.
    
    Parameters:
        model_knn (NearestNeighbors): Trained KNN model.
        user_item_matrix (pd.DataFrame): User-item matrix.
        user_id (int): ID of the target user.
        n_neighbors (int): Number of similar users to find.
    
    Returns:
        list: Indices of the top N similar users.
    """
    if model_knn is None or user_item_matrix is None:
        print("Error: KNN model or user-item matrix is not available.")
        return None
    
    try:
        # Get the row corresponding to the target user
        user_vector = user_item_matrix.loc[user_id].values.reshape(1, -1)
        
        # Find the top N similar users
        distances, indices = model_knn.kneighbors(user_vector, n_neighbors=n_neighbors)
        
        # Extract the user IDs of the similar users
        similar_user_ids = [user_item_matrix.index[i] for i in indices.flatten()]
        print(f"Top {n_neighbors} similar users for user {user_id}: {similar_user_ids}")
        return similar_user_ids
    except KeyError:
        print(f"Error: User ID {user_id} not found in the user-item matrix.")
        return None
    except Exception as e:
        print(f"Error while finding similar users: {e}")
        return None


def generate_knn_predictions(model_knn, user_item_matrix, test_data):
    """
    Generate predictions for the test set using the KNN model.
    
    Parameters:
        model_knn (NearestNeighbors): Trained KNN model.
        user_item_matrix (pd.DataFrame): User-item matrix.
        test_data (pd.DataFrame): Test dataset with columns ['user_id', 'item_id'].
    
    Returns:
        list: Predicted ratings for the test set.
    """
    if model_knn is None or user_item_matrix is None:
        print("Error: KNN model or user-item matrix is not available.")
        return []
    
    predictions = []
    try:
        for _, row in test_data.iterrows():
            user_id, item_id = row['user_id'], row['item_id']
            
            # Check if user and item exist in the matrix
            if user_id not in user_item_matrix.index or item_id not in user_item_matrix.columns:
                predictions.append(0)  # Default prediction for unknown user/item pairs
                continue
            
            # Get the corresponding row vector
            user_vector = user_item_matrix.loc[user_id].values.reshape(1, -1)
            item_index = user_item_matrix.columns.get_loc(item_id)
            
            # Find similar users
            _, indices = model_knn.kneighbors(user_vector)
            similar_users = indices.flatten()
            
            # Predict rating as the average of similar users' ratings for the item
            similar_ratings = user_item_matrix.iloc[similar_users, item_index]
            prediction = similar_ratings.mean() if len(similar_ratings) > 0 else 0
            predictions.append(prediction)
        
        print("KNN predictions generated successfully.")
        return predictions
    except Exception as e:
        print(f"Error while generating KNN predictions: {e}")
        return []


if __name__ == "__main__":
    try:
        # Step 1: Load the preprocessed training data
        train_data = pd.read_csv('train_data.csv')
        print("Training data loaded successfully.")
        
        # Step 2: Create the user-item matrix
        user_item_matrix = create_user_item_matrix(train_data)
        if user_item_matrix is None:
            print("Failed to create the user-item matrix. Exiting...")
            exit()
        
        # Step 3: Train the KNN model
        model_knn = train_knn_model(user_item_matrix, metric='cosine', algorithm='brute')
        if model_knn is None:
            print("Failed to train the KNN model. Exiting...")
            exit()
        
        # Step 4: Generate recommendations for a specific user (e.g., user_id = 1)
        user_id = 1
        similar_users = find_similar_users(model_knn, user_item_matrix, user_id, n_neighbors=5)
        if similar_users is None:
            print(f"Failed to find similar users for user {user_id}.")
        
        # Step 5: Generate predictions for the test set
        test_data = pd.read_csv('test_data.csv')
        print("Test data loaded successfully.")
        
        knn_predictions = generate_knn_predictions(model_knn, user_item_matrix, test_data[['user_id', 'item_id']])
        
        # Save predictions to a file
        if knn_predictions:
            pd.DataFrame({'prediction': knn_predictions}).to_csv('knn_predictions.csv', index=False)
            print("KNN predictions saved to 'knn_predictions.csv'.")
        else:
            print("No predictions were generated. Exiting...")
    except FileNotFoundError as fnf_error:
        print(f"File not found: {fnf_error}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")