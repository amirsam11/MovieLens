import pandas as pd

# Generate ratings_data.csv
columns = ['user_id', 'item_id', 'rating', 'timestamp']
ratings_data = pd.read_csv('ml-100k/u.data', sep='\t', names=columns)
ratings_data.to_csv('ratings_data.csv', index=False)
print("Generated 'ratings_data.csv'.")

# Generate user_features.csv
columns = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
user_data = pd.read_csv('ml-100k/u.user', sep='|', names=columns)

# Encode categorical features
user_data['gender'] = user_data['gender'].map({'M': 0, 'F': 1})
user_data = pd.get_dummies(user_data, columns=['occupation'])
user_data.to_csv('user_features.csv', index=False)
print("Generated 'user_features.csv'.")

# Generate item_features.csv
columns = ['item_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
           'unknown', 'Action', 'Adventure', 'Animation', 'Childrens', 'Comedy',
           'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film_Noir', 'Horror',
           'Musical', 'Mystery', 'Romance', 'Sci_Fi', 'Thriller', 'War', 'Western']
item_data = pd.read_csv('ml-100k/u.item', sep='|', names=columns, encoding='latin-1')

# Keep only relevant columns (item_id and genre features)
item_features = item_data[['item_id', 'unknown', 'Action', 'Adventure', 'Animation', 'Childrens',
                           'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                           'Film_Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
                           'Sci_Fi', 'Thriller', 'War', 'Western']]
item_features.to_csv('item_features.csv', index=False)
print("Generated 'item_features.csv'.")
