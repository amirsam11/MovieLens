# data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """
    Load the MovieLens 100K dataset into a pandas DataFrame.
    
    Parameters:
        file_path (str): Path to the dataset file (u.data).
    
    Returns:
        pd.DataFrame: Loaded dataset with specified column names.
    """
    # Define column names for the dataset
    columns = ['user_id', 'item_id', 'rating', 'timestamp']
    
    try:
        # Load the dataset using pandas
        data = pd.read_csv(file_path, sep='\t', names=columns)
        print("Dataset loaded successfully.")
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Please check the file path.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
        return None

def preprocess_data(data):
    """
    Preprocess the dataset by adding a 'recommended' column.
    
    Parameters:
        data (pd.DataFrame): Input dataset.
    
    Returns:
        pd.DataFrame: Preprocessed dataset with the 'recommended' column added.
    """
    if data is None:
        return None
    
    try:
        # Add a new column 'recommended': 1 if rating >= 3, else 0
        data['recommended'] = data['rating'].apply(lambda x: 1 if x >= 3 else 0)
        print("Preprocessing complete: 'recommended' column added.")
        return data
    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
        return None

def split_data(data, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing subsets.
    
    Parameters:
        data (pd.DataFrame): Input dataset.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed for reproducibility.
    
    Returns:
        tuple: A tuple containing the training and testing datasets.
    """
    if data is None:
        return None, None
    
    try:
        # Split the data into training and testing sets
        train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
        print(f"Data split into training ({len(train_data)} samples) and testing ({len(test_data)} samples) sets.")
        return train_data, test_data
    except Exception as e:
        print(f"An error occurred while splitting the data: {e}")
        return None, None

def save_data(train_data, test_data, train_file='train_data.csv', test_file='test_data.csv'):
    """
    Save the training and testing datasets to CSV files.
    
    Parameters:
        train_data (pd.DataFrame): Training dataset.
        test_data (pd.DataFrame): Testing dataset.
        train_file (str): Filename for the training dataset.
        test_file (str): Filename for the testing dataset.
    """
    if train_data is None or test_data is None:
        print("Error: Data not available for saving.")
        return
    
    try:
        # Save the training and testing datasets to CSV files
        train_data.to_csv(train_file, index=False)
        test_data.to_csv(test_file, index=False)
        print(f"Training data saved to {train_file}.")
        print(f"Testing data saved to {test_file}.")
    except Exception as e:
        print(f"An error occurred while saving the data: {e}")

if __name__ == "__main__":
    # Define the file path for the MovieLens 100K dataset
    DATASET_FILE_PATH = 'ml-100k/u.data'
    
    # Step 1: Load the dataset
    raw_data = load_data(DATASET_FILE_PATH)
    
    if raw_data is not None:
        # Step 2: Preprocess the dataset
        preprocessed_data = preprocess_data(raw_data)
        
        if preprocessed_data is not None:
            # Step 3: Split the dataset into training and testing sets
            train_set, test_set = split_data(preprocessed_data)
            
            if train_set is not None and test_set is not None:
                # Step 4: Save the training and testing datasets to CSV files
                save_data(train_set, test_set)
