
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


# =====================================
# 1. LOAD DATA
# =====================================

ratings = pd.read_csv("data/processed_ratings.csv")

train, test = train_test_split(
    ratings,
    test_size=0.2,
    random_state=42
)

print("Training Data:", train.shape)
print("Testing Data:", test.shape)


# =====================================
# 2. CREATE USER-MOVIE MATRIX
# =====================================

user_movie_matrix = train.pivot(
    index="userId",
    columns="movieId",
    values="rating"
)

user_ids = user_movie_matrix.index
movie_ids = user_movie_matrix.columns

# Fill missing ratings with zero
R = user_movie_matrix.fillna(0).values.astype(np.float32)

print("Matrix Shape:", R.shape)


# =====================================
# 3. USER SIMILARITY
# =====================================

user_similarity = cosine_similarity(
    csr_matrix(R),
    dense_output=False
)

print("Similarity calculated!")


# =====================================
# 4. PREDICT RATINGS
# =====================================

# Normalize similarity by sum of absolute similarities
similarity_sum = np.asarray(
    abs(user_similarity).sum(axis=1)
).ravel()

# Weighted rating sums
weighted_ratings = user_similarity @ R

predicted_matrix = np.divide(
    weighted_ratings,
    similarity_sum[:, None],
    out=np.zeros_like(weighted_ratings),
    where=similarity_sum[:, None] != 0
)

# Use global mean for missing predictions
global_mean = train["rating"].mean()

predicted_matrix[predicted_matrix == 0] = global_mean

predicted_matrix = np.clip(
    predicted_matrix,
    0.5,
    5.0
)


# =====================================
# 5. TEST PREDICTIONS
# =====================================

user_to_index = {
    user_id: idx
    for idx, user_id in enumerate(user_ids)
}

movie_to_index = {
    movie_id: idx
    for idx, movie_id in enumerate(movie_ids)
}

valid_test = test[
    test["userId"].isin(user_to_index)
    & test["movieId"].isin(movie_to_index)
].copy()

user_indices = valid_test["userId"].map(user_to_index).to_numpy()
movie_indices = valid_test["movieId"].map(movie_to_index).to_numpy()

valid_test["predicted_rating"] = predicted_matrix[
    user_indices,
    movie_indices
]

# Evaluate only held-out ratings
rmse = np.sqrt(
    mean_squared_error(
        valid_test["rating"],
        valid_test["predicted_rating"]
    )
)

mae = mean_absolute_error(
    valid_test["rating"],
    valid_test["predicted_rating"]
)


# =====================================
# 6. PRECISION@10 AND RECALL@10
# =====================================

def precision_recall_at_k(k=10, threshold=4.0):

    precisions = []
    recalls = []

    for user_id, group in valid_test.groupby("userId"):

        user_idx = user_to_index[user_id]

        relevant_movies = set(
            group.loc[
                group["rating"] >= threshold,
                "movieId"
            ]
        )

        if not relevant_movies:
            continue

        # Scores for all movies
        scores = predicted_matrix[user_idx].copy()

        # Exclude movies already present in training data
        watched = R[user_idx] > 0
        scores[watched] = -np.inf

        # Top K recommendations
        top_indices = np.argpartition(
            scores,
            -k
        )[-k:]

        top_indices = top_indices[
            np.argsort(scores[top_indices])[::-1]
        ]

        recommended_movies = {
            movie_ids[idx] for idx in top_indices
            if scores[idx] != -np.inf
        }

        hits = len(recommended_movies & relevant_movies)

        precisions.append(hits / k)
        recalls.append(hits / len(relevant_movies))

    return np.mean(precisions), np.mean(recalls)


precision, recall = precision_recall_at_k(10)


# =====================================
# 7. RESULTS
# =====================================

print("\n========== MODEL EVALUATION ==========")

print(f"RMSE:         {rmse:.4f}")
print(f"MAE:          {mae:.4f}")
print(f"Precision@10: {precision:.4f}")
print(f"Recall@10:    {recall:.4f}")

print("\nEvaluation completed successfully!")