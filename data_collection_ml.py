import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. LOAD DATA
# ==========================================

movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")
tags = pd.read_csv("data/tags.csv")
links = pd.read_csv("data/links.csv")


# ==========================================
# 2. BASIC INFORMATION
# ==========================================

print("\n========== MOVIES ==========")
print(movies.head())
print("Shape:", movies.shape)

print("\n========== RATINGS ==========")
print(ratings.head())
print("Shape:", ratings.shape)

print("\n========== TAGS ==========")
print(tags.head())
print("Shape:", tags.shape)

print("\n========== LINKS ==========")
print(links.head())
print("Shape:", links.shape)


# ==========================================
# 3. NUMBER OF MOVIES AND USERS
# ==========================================

print("\n========== DATASET STATISTICS ==========")

print("Number of movies:", movies["movieId"].nunique())
print("Number of users:", ratings["userId"].nunique())
print("Number of ratings:", len(ratings))


# ==========================================
# 4. MISSING VALUES
# ==========================================

print("\n========== MISSING VALUES ==========")

print("\nMovies:")
print(movies.isnull().sum())

print("\nRatings:")
print(ratings.isnull().sum())

print("\nTags:")
print(tags.isnull().sum())

print("\nLinks:")
print(links.isnull().sum())


# ==========================================
# 5. DUPLICATES
# ==========================================

print("\n========== DUPLICATES ==========")

print("Duplicate movies:", movies.duplicated().sum())
print("Duplicate ratings:", ratings.duplicated().sum())
print("Duplicate tags:", tags.duplicated().sum())
print("Duplicate links:", links.duplicated().sum())


# ==========================================
# 6. RATING DISTRIBUTION
# ==========================================

print("\n========== RATING DISTRIBUTION ==========")

print(ratings["rating"].value_counts().sort_index())

plt.figure(figsize=(8, 5))

sns.countplot(
    x="rating",
    data=ratings
)

plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Number of Ratings")

plt.show()


# ==========================================
# 7. MOST RATED MOVIES
# ==========================================

movie_rating_count = (
    ratings.groupby("movieId")
    .size()
    .sort_values(ascending=False)
)

print("\n========== MOST RATED MOVIES ==========")

most_rated = movie_rating_count.head(10)

print(most_rated)


# ==========================================
# 8. MOST ACTIVE USERS
# ==========================================

user_rating_count = (
    ratings.groupby("userId")
    .size()
    .sort_values(ascending=False)
)

print("\n========== MOST ACTIVE USERS ==========")

print(user_rating_count.head(10))


# ==========================================
# 9. MOVIE + RATING INFORMATION
# ==========================================

movie_rating_data = ratings.merge(
    movies,
    on="movieId"
)

print("\n========== MERGED DATA ==========")

print(movie_rating_data.head())

print("Merged shape:", movie_rating_data.shape)


# ==========================================
# 10. SAVE RAW COPIES
# ==========================================

movies.to_csv("data/movies_raw.csv", index=False)
ratings.to_csv("data/ratings_raw.csv", index=False)

print("\nData collection completed successfully!")