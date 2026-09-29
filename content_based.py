import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =====================================
# 1. LOAD DATA
# =====================================

movies = pd.read_csv("data/processed_movies.csv")

print("Movies Loaded:", movies.shape)


# =====================================
# 2. SELECT FEATURES
# =====================================

movies["genres"] = movies["genres"].fillna("")

movies["genres"] = movies["genres"].str.replace(
    "|", " ", regex=False
)

movies["title_clean"] = movies["title_clean"].fillna("")

movies["features"] = (
    movies["title_clean"] + " " + movies["genres"]
)


# =====================================
# 3. TF-IDF VECTORIZATION
# =====================================

tfidf = TfidfVectorizer(stop_words="english")

tfidf_matrix = tfidf.fit_transform(movies["features"])

print("TF-IDF Matrix Shape:", tfidf_matrix.shape)


# =====================================
# 4. COSINE SIMILARITY
# =====================================

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

print("Similarity Matrix Shape:", cosine_sim.shape)


# =====================================
# 5. MOVIE INDEX
# =====================================

indices = pd.Series(
    movies.index,
    index=movies["title"]
).drop_duplicates()


# =====================================
# 6. RECOMMENDATION FUNCTION
# =====================================

def recommend_movies(movie_title, top_n=10):

    if movie_title not in indices:
        return "Movie not found!"

    idx = indices[movie_title]

    similarity_scores = list(
        enumerate(cosine_sim[idx])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    similarity_scores = similarity_scores[1:top_n + 1]

    movie_indices = [
        i[0] for i in similarity_scores
    ]

    recommendations = movies.iloc[movie_indices][
        ["movieId", "title", "genres"]
    ]

    return recommendations


# =====================================
# 7. TEST
# =====================================

print("\n========== RECOMMENDATIONS ==========")

result = recommend_movies("Toy Story (1995)", 10)

print(result)