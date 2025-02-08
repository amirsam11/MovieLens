import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity

class ExtKNNCF:
    def __init__(self, K_min=5, K_max=20, cognitive_weight=0.5, cognitive_threshold=0.7):
        self.K_min = K_min
        self.K_max = K_max
        self.cognitive_weight = cognitive_weight
        self.cognitive_threshold = cognitive_threshold

    def fit(self, ratings, user_cognition):
        self.ratings = ratings
        self.user_cognition = user_cognition

    def predict(self, test_data):
        predictions = []
        for _, row in test_data.iterrows():
            user_id, item_id = row['user_id'], row['item_id']
            user_ratings = self.ratings.loc[user_id].values.reshape(1, -1)
            user_cog = self.user_cognition.loc[user_id].values.reshape(1, -1)

            # Find neighbors
            distances = cdist(user_ratings, self.ratings.values, metric='euclidean')
            neighbors = np.argsort(distances, axis=1)[:, :self.K_max]

            # Compute final distances
            final_distances = []
            for neighbor in neighbors[0]:
                neighbor_id = self.ratings.index[neighbor]
                neighbor_cog = self.user_cognition.loc[neighbor_id].values.reshape(1, -1)

                # Cognitive similarity
                cog_sim = cosine_similarity(user_cog, neighbor_cog)[0][0]

                # Final distance
                if cog_sim < self.cognitive_threshold:
                    final_distance = distances[0][neighbor]
                else:
                    final_distance = (1 - self.cognitive_weight) * distances[0][neighbor] + \
                                     self.cognitive_weight * (1 - cog_sim)

                final_distances.append((neighbor_id, final_distance))

            # Sort neighbors by final distance
            final_distances.sort(key=lambda x: x[1])
            top_k = [n[0] for n in final_distances[:self.K_min]]

            # Predict rating
            numerator = sum(self.ratings.loc[n, item_id] for n in top_k if not np.isnan(self.ratings.loc[n, item_id]))
            denominator = len(top_k)
            predicted_rating = numerator / denominator if denominator > 0 else 0
            predictions.append(predicted_rating)

        return predictions

# Load data
ratings = pd.read_csv('ratings.csv')  # MovieLens dataset
user_cognition = pd.read_csv('user_cognition.csv')  # User cognition data

# Preprocess data
ratings = ratings.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
train_data, test_data = train_test_split(ratings.stack().reset_index(), test_size=0.2, random_state=42)

# Initialize and train ExtKNNCF
model = ExtKNNCF(K_min=5, K_max=20, cognitive_weight=0.5, cognitive_threshold=0.7)
model.fit(ratings, user_cognition)

# Generate predictions
test_data = test_data[['user_id', 'item_id']]
predictions = model.predict(test_data)

# Evaluate performance
actual_ratings = [ratings.loc[row['user_id'], row['item_id']] for _, row in test_data.iterrows()]
mae = mean_absolute_error(actual_ratings, predictions)
rmse = np.sqrt(mean_squared_error(actual_ratings, predictions))
print(f"MAE: {mae}, RMSE: {rmse}")
