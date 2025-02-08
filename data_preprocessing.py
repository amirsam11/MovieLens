# data_preprocessing.py

import pandas as pd

def preprocess_data(input_file, output_train, output_test):
    """
    Preprocess the MovieLens dataset and split it into training and testing sets.

    Parameters:
        input_file (str): Path to the raw dataset file (e.g., 'ml-100k/u.data').
        output_train (str): Path to save the training dataset.
        output_test (str): Path to save the testing dataset.
    """
    try:
        # Define column names
        columns = ['user_id', 'item_id', 'rating', 'timestamp']
        
        # Load the dataset
        data = pd.read_csv(input_file, sep='\t', names=columns)
        print("Dataset loaded successfully.")
        
        # Add a 'recommended' column: 1 if rating >= 3, else 0
        data['recommended'] = data['rating'].apply(lambda x: 1 if x >= 3 else 0)
        print("Added 'recommended' column.")
        
        # Split the data into training and testing sets (80% train, 20% test)
        train_data = data.sample(frac=0.8, random_state=42)
        test_data = data.drop(train_data.index)
        print("Data split into training and testing sets.")
        
        # Save the datasets
        train_data.to_csv(output_train, index=False)
        test_data.to_csv(output_test, index=False)
        print(f"Training data saved to {output_train}.")
        print(f"Testing data saved to {output_test}.")

    except FileNotFoundError:
        print(f"Error: File not found at {input_file}. Please check the file path.")
    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")

if __name__ == "__main__":
    # Define file paths
    raw_data_path = 'ml-100k/u.data'
    train_data_path = 'train_data.csv'
    test_data_path = 'test_data.csv'
    
    # Preprocess and split the data
    preprocess_data(raw_data_path, train_data_path, test_data_path)
