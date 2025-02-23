# movie-recom
movie recommendation system
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse

# Load movie ratings dataset
ratings_dict = {
    "userId": [1, 1, 1, 2, 2, 3, 3, 4, 4, 5],
    "movieId": [101, 102, 103, 101, 104, 102, 103, 101, 105, 106],
    "rating": [5, 3, 4, 4, 5, 2, 3, 5, 4, 2]
}
df = pd.DataFrame(ratings_dict)

# Define a Reader object to specify rating scale
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)

# Split data into training and test sets
trainset, testset = train_test_split(data, test_size=0.2)

# Use SVD (Singular Value Decomposition) algorithm
model = SVD()
model.fit(trainset)

# Evaluate model performance
predictions = model.test(testset)
rmse(predictions)

# Function to get top N movie recommendations for a user
def get_movie_recommendations(user_id, movie_ids, model, n=5):
    predictions = [model.predict(user_id, movie_id) for movie_id in movie_ids]
    predictions.sort(key=lambda x: x.est, reverse=True)
    return [(pred.iid, pred.est) for pred in predictions[:n]]

# Get movie recommendations for a specific user
all_movie_ids = set(df['movieId'].unique())
watched_movie_ids = set(df[df['userId'] == 1]['movieId'])
new_movie_ids = all_movie_ids - watched_movie_ids  # Unwatched movies
recommendations = get_movie_recommendations(1, new_movie_ids, model, n=3)

# Print recommendations
print("Recommended Movies for User 1:")
for movie_id, rating in recommendations:
    print(f"Movie ID: {movie_id}, Predicted Rating: {rating:.2f}")
