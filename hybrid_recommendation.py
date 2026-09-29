import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =====================================
# 1. LOAD DATA
# =====================================

movies = pd.read_csv("data/processed_movies.csv")
ratings = pd.read_csv("data/processed_ratings.csv")


# =====================================
# 2. CONTENT-BASED MODEL
# =====================================

movies["genres"] = movies["genres"].fillna("")

movies["genres"] = movies["genres"].str.replace(
    "|", " ", regex=False
)

movies["title_clean"] = movies["title_clean"].fillna("")

movies["features"] = (
    movies["title_clean"] + " " + movies["genres"]
)

tfidf = TfidfVectorizer(stop_words="english")

tfidf_matrix = tfidf.fit_transform(movies["features"])

content_similarity = cosine_similarity(
    tfidf_matrix,
    tfidf_matrix
)

movie_indices = pd.Series(
    movies.index,
    index=movies["title"]
).drop_duplicates()


# =====================================
# 3. COLLABORATIVE MODEL
# =====================================

user_movie_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
)

user_similarity = cosine_similarity(
    user_movie_matrix.fillna(0)
)

user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.index
)


# =====================================
# 4. HYBRID RECOMMENDATION
# =====================================

def hybrid_recommend(movie_title, user_id, top_n=10, alpha=0.5):

    if movie_title not in movie_indices:
        return "Movie not found!"

    if user_id not in user_movie_matrix.index:
        return "User not found!"

    movie_idx = movie_indices[movie_title]

    # Content similarity scores
    content_scores = content_similarity[movie_idx]

    # Collaborative similar users
    similar_users = user_similarity_df[user_id].drop(
        user_id
    ).sort_values(ascending=False).head(10)

    watched_movies = user_movie_matrix.loc[user_id].dropna().index

    recommendations = []

    for idx, movie in movies.iterrows():

        movie_id = movie["movieId"]

        # Don't recommend the input movie
        if idx == movie_idx:
            continue

        # Don't recommend already watched movies
        if movie_id in watched_movies:
            continue

        # Content score
        content_score = content_scores[idx]

        # Collaborative score
        weighted_sum = 0
        similarity_sum = 0

        if movie_id in user_movie_matrix.columns:

            for other_user, similarity in similar_users.items():

                rating = user_movie_matrix.loc[
                    other_user, movie_id
                ]

                if not pd.isna(rating) and similarity > 0:

                    weighted_sum += similarity * rating
                    similarity_sum += similarity

        if similarity_sum > 0:
            collaborative_score = weighted_sum / similarity_sum / 5
        else:
            collaborative_score = 0

        # Hybrid score
        hybrid_score = (
            alpha * content_score
            + (1 - alpha) * collaborative_score
        )

        recommendations.append({
            "movieId": movie_id,
            "title": movie["title"],
            "genres": movie["genres"],
            "content_score": content_score,
            "collaborative_score": collaborative_score,
            "hybrid_score": hybrid_score
        })

    result = pd.DataFrame(recommendations)

    result = result.sort_values(
        by="hybrid_score",
        ascending=False
    )

    return result.head(top_n)


# =====================================
# 5. TEST
# =====================================

result = hybrid_recommend(
    movie_title="Toy Story (1995)",
    user_id=1,
    top_n=10
)

print("\n========== HYBRID RECOMMENDATIONS ==========")

print(result)