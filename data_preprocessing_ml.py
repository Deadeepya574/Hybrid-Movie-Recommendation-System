import pandas as pd
import numpy as np
import re

# =====================================
# 1. LOAD DATA
# =====================================

movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")

print("Original Movies:", movies.shape)
print("Original Ratings:", ratings.shape)


# =====================================
# 2. HANDLE MISSING VALUES
# =====================================

movies.dropna(subset=["movieId", "title"], inplace=True)

ratings.dropna(
    subset=["userId", "movieId", "rating"],
    inplace=True
)


# =====================================
# 3. REMOVE DUPLICATES
# =====================================

movies.drop_duplicates(subset=["movieId"], inplace=True)

ratings.drop_duplicates(
    subset=["userId", "movieId"],
    keep="last",
    inplace=True
)


# =====================================
# 4. CLEAN MOVIE TITLES
# =====================================

def clean_title(title):
    title = re.sub(r"\(\d{4}\)", "", title)
    title = title.strip()
    return title

movies["title_clean"] = movies["title"].apply(clean_title)


# =====================================
# 5. PROCESS GENRES
# =====================================

movies["genres"] = movies["genres"].fillna("")

movies["genres_list"] = movies["genres"].apply(
    lambda x: x.split("|") if x else []
)


# =====================================
# 6. FILTER LOW-ACTIVITY USERS
# =====================================

user_counts = ratings["userId"].value_counts()

active_users = user_counts[user_counts >= 5].index

ratings = ratings[
    ratings["userId"].isin(active_users)
]


# =====================================
# 7. FILTER LOW-RATING MOVIES
# =====================================

movie_counts = ratings["movieId"].value_counts()

popular_movies = movie_counts[movie_counts >= 5].index

ratings = ratings[
    ratings["movieId"].isin(popular_movies)
]


# =====================================
# 8. MERGE DATA
# =====================================

merged_data = ratings.merge(
    movies,
    on="movieId",
    how="inner"
)


# =====================================
# 9. SAVE PROCESSED DATA
# =====================================

movies.to_csv("data/processed_movies.csv", index=False)

ratings.to_csv("data/processed_ratings.csv", index=False)

merged_data.to_csv("data/merged_data.csv", index=False)


# =====================================
# 10. FINAL SUMMARY
# =====================================

print("\n========== PREPROCESSING COMPLETE ==========")

print("Processed Movies:", movies.shape)
print("Processed Ratings:", ratings.shape)
print("Merged Dataset:", merged_data.shape)

print("\nMissing Values:")
print(merged_data.isnull().sum())

print("\nSaved successfully!")