import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity


# =====================================
# 1. LOAD DATA
# =====================================

ratings = pd.read_csv("data/processed_ratings.csv")

print("Ratings Dataset:", ratings.shape)


# =====================================
# 2. CREATE USER-MOVIE MATRIX
# =====================================

user_movie_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
)

print("\nUser-Movie Matrix:")
print(user_movie_matrix.shape)


# =====================================
# 3. HANDLE MISSING VALUES
# =====================================

user_movie_matrix_filled = user_movie_matrix.fillna(0)


# =====================================
# 4. CALCULATE USER SIMILARITY
# =====================================

user_similarity = cosine_similarity(
    user_movie_matrix_filled
)

user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.index
)

print("\nSimilarity Matrix:")
print(user_similarity_df.shape)


# =====================================
# 5. RECOMMENDATION FUNCTION
# =====================================

def recommend_movies(user_id, top_n=10):

    if user_id not in user_movie_matrix.index:
        return "User not found!"

    # Similar users
    similar_users = user_similarity_df[user_id].sort_values(
        ascending=False
    )

    # Exclude the same user
    similar_users = similar_users.drop(user_id)

    # Select top 10 similar users
    similar_users = similar_users.head(10)

    # Movies already watched
    watched_movies = user_movie_matrix.loc[user_id].dropna().index

    # Candidate movies
    candidate_movies = user_movie_matrix.columns.difference(
        watched_movies
    )

    recommendations = []

    for movie_id in candidate_movies:

        weighted_sum = 0
        similarity_sum = 0

        for other_user, similarity in similar_users.items():

            rating = user_movie_matrix.loc[other_user, movie_id]

            if not pd.isna(rating) and similarity > 0:

                weighted_sum += similarity * rating
                similarity_sum += similarity

        if similarity_sum > 0:

            predicted_rating = weighted_sum / similarity_sum

            recommendations.append(
                (movie_id, predicted_rating)
            )

    recommendations.sort(
        key=lambda x: x[1],
        reverse=True
    )

    top_movies = recommendations[:top_n]

    result = pd.DataFrame(
        top_movies,
        columns=["movieId", "predicted_rating"]
    )

    # Add movie titles
    movies = pd.read_csv("data/processed_movies.csv")

    result = result.merge(
        movies[["movieId", "title", "genres"]],
        on="movieId",
        how="left"
    )

    return result


# =====================================
# 6. TEST
# =====================================

print("\n========== RECOMMENDATIONS ==========")

print(recommend_movies(user_id=1, top_n=10))