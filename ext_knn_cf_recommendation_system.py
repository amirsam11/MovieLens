# ext_knn_cf_recommendation_system.py

from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np


def merge_data_with_features(ratings_data, user_features, item_features):
    """
    Merge ratings data with user and item features to create an extended feature dataset.

    Parameters:
        ratings_data (pd.DataFrame): Ratings dataset with columns ['user_id', 'item_id', 'rating'].
        user_features (pd.DataFrame): User features dataset with columns ['user_id', ...].
        item_features (pd.DataFrame): Item features dataset with columns ['item_id', ...].

    Returns:
        pd.DataFrame: Extended dataset with merged features.
    """
    try:
        # Merge ratings with user features on 'user_id'
        merged_data = pd.merge(ratings_data, user_features, on='user_id', how='left')
        
        # Merge the result with item features on 'item_id'
        merged_data = pd.merge(merged_data, item_features, on='item_id', how='left')
        
        # Keep only numeric columns
        merged_data = merged_data.select_dtypes(include=['number'])
        
        print("Data merged with user and item features successfully.")
        return merged_data
    except Exception as e:
        print(f"Error while merging data with features: {e}")
        return None


def create_extended_matrix(data, user_id_col, item_id_col, rating_col, feature_cols):
    """
    Create an extended feature matrix from the merged dataset.

    Parameters:
        data (pd.DataFrame): Merged dataset with ratings and features.
        user_id_col (str): Column name for user IDs.
        item_id_col (str): Column name for item IDs.
        rating_col (str): Column name for ratings.
        feature_cols (list): List of feature column names to include in the matrix.

    Returns:
        pd.DataFrame: Extended feature matrix filled with values, missing values replaced with 0.
    """
    try:
        # Filter the dataset to include only numeric columns
        numeric_data = data.select_dtypes(include=['number'])
        
        # Pivot the data to create a user-item-feature matrix
        pivot_data = numeric_data.pivot_table(
            index=user_id_col, 
            columns=item_id_col, 
            values=feature_cols + [rating_col], 
            aggfunc='mean'
        )
        
        # Fill missing values with 0
        extended_matrix = pivot_data.fillna(0)
        
        print("Extended feature matrix created successfully.")
        return extended_matrix
    except Exception as e:
        print(f"Error while creating the extended feature matrix: {e}")
        return None


def train_ext_knn_model(extended_matrix, metric='cosine', algorithm='brute'):
    """
    Train the ExtKNNCF model using the cosine similarity metric.

    Parameters:
        extended_matrix (pd.DataFrame): Extended feature matrix.
        metric (str): Distance metric for KNN (default is 'cosine').
        algorithm (str): Algorithm used for KNN (default is 'brute').

    Returns:
        NearestNeighbors: Trained KNN model.
    """
    if extended_matrix is None:
        return None
    
    try:
        # Initialize the KNN model with cosine similarity and brute force algorithm
        model_ext_knn = NearestNeighbors(metric=metric, algorithm=algorithm)
        
        # Fit the model to the extended feature matrix
        model_ext_knn.fit(extended_matrix.values)
        
        print("ExtKNNCF model trained successfully.")
        return model_ext_knn
    except Exception as e:
        print(f"Error while training the ExtKNNCF model: {e}")
        return None


def find_similar_users(model_ext_knn, extended_matrix, user_id, n_neighbors=5):
    """
    Find the top N similar users for a given user based on the extended feature matrix.

    Parameters:
        model_ext_knn (NearestNeighbors): Trained ExtKNNCF model.
        extended_matrix (pd.DataFrame): Extended feature matrix.
        user_id (int): ID of the target user.
        n_neighbors (int): Number of similar users to find.

    Returns:
        list: Indices of the top N similar users.
    """
    if model_ext_knn is None or extended_matrix is None:
        return None
    
    try:
        # Get the row corresponding to the target user
        user_vector = extended_matrix.loc[user_id].values.reshape(1, -1)
        
        # Find the top N similar users
        distances, indices = model_ext_knn.kneighbors(user_vector, n_neighbors=n_neighbors)
        
        # Extract the user IDs of the similar users
        similar_user_ids = [extended_matrix.index[i] for i in indices.flatten()]
        
        print(f"Top {n_neighbors} similar users for user {user_id}: {similar_user_ids}")
        return similar_user_ids
    except Exception as e:
        print(f"Error while finding similar users: {e}")
        return None


def generate_ext_knn_cf_predictions(model_ext_knn, extended_matrix, test_data):
    """
    Generate predictions for the test set using the ExtKNNCF model.

    Parameters:
        model_ext_knn (NearestNeighbors): Trained ExtKNNCF model.
        extended_matrix (pd.DataFrame): Extended feature matrix.
        test_data (pd.DataFrame): Test dataset with columns ['user_id', 'item_id'].

    Returns:
        list: Predicted ratings for the test set.
    """
    try:
        predictions = []
        for _, row in test_data.iterrows():
            user_id, item_id = row['user_id'], row['item_id']
            
            # Get the corresponding row vector
            user_vector = extended_matrix.loc[user_id].values.reshape(1, -1)
            item_index = extended_matrix.columns.get_level_values(1).tolist().index(item_id)
            
            # Find similar users
            distances, indices = model_ext_knn.kneighbors(user_vector)
            similar_users = indices.flatten()
            
            # Predict rating as the average of similar users' ratings for the item
            similar_ratings = extended_matrix.iloc[similar_users, item_index]
            prediction = similar_ratings.mean() if len(similar_ratings) > 0 else 0
            predictions.append(prediction)
        
        print("ExtKNNCF predictions generated successfully.")
        return predictions
    except Exception as e:
        print(f"Error while generating ExtKNNCF predictions: {e}")
        return []


if __name__ == "__main__":
    # Step 1: Load the preprocessed datasets
    try:
        ratings_data = pd.read_csv('ratings_data.csv')  # Contains ['user_id', 'item_id', 'rating']
        user_features = pd.read_csv('user_features.csv')  # Contains ['user_id', demographic features]
        item_features = pd.read_csv('item_features.csv')  # Contains ['item_id', genre features]
        test_data = pd.read_csv('test_data.csv')  # Contains ['user_id', 'item_id']
        
        print("Datasets loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: File not found. Please ensure all required files exist. {e}")
        exit()
    except Exception as e:
        print(f"Error while loading the datasets: {e}")
        exit()
    
    # Step 2: Merge data with user and item features
    merged_data = merge_data_with_features(ratings_data, user_features, item_features)
    
    if merged_data is not None:
        # Step 3: Create the extended feature matrix
        feature_cols = [col for col in merged_data.columns if col not in ['user_id', 'item_id', 'rating']]
        extended_matrix = create_extended_matrix(merged_data, 'user_id', 'item_id', 'rating', feature_cols)
        
        if extended_matrix is not None:
            # Step 4: Train the ExtKNNCF model
            model_ext_knn = train_ext_knn_model(extended_matrix, metric='cosine', algorithm='brute')
            
            if model_ext_knn is not None:
                # Step 5: Generate predictions for the test set
                ext_knn_cf_predictions = generate_ext_knn_cf_predictions(
                    model_ext_knn, extended_matrix, test_data[['user_id', 'item_id']]
                )
                
                # Save predictions to a file
                pd.DataFrame({'prediction': ext_knn_cf_predictions}).to_csv('ext_knn_cf_predictions.csv', index=False)
                print("ExtKNNCF predictions saved to 'ext_knn_cf_predictions.csv'.")