import webbrowser
from threading import Timer

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =====================================
# 1. INITIALIZE FLASK
# =====================================

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return send_from_directory("frontend", "index.html")


@app.route("/<path:filename>")
def frontend_files(filename):
    return send_from_directory("frontend", filename)

# =====================================
# 2. LOAD DATA
# =====================================

movies = pd.read_csv("data/processed_movies.csv")
ratings = pd.read_csv("data/processed_ratings.csv")

movies["genres"] = movies["genres"].fillna("")

movies["title_clean"] = movies["title_clean"].fillna("")

movies["features"] = (
    movies["title_clean"] + " " + movies["genres"].str.replace(
        "|", " ", regex=False
    )
)


# =====================================
# 3. CONTENT-BASED MODEL
# =====================================

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
# 4. COLLABORATIVE MODEL
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
# 6. SEARCH MOVIES API
# =====================================

@app.route("/search", methods=["GET"])
def search_movies():

    query = request.args.get("q", "").strip()

    if not query:
        return jsonify({"error": "Search query required"}), 400

    result = movies[
        movies["title"].str.contains(
            query,
            case=False,
            na=False
        )
    ].head(20)

    return jsonify(
        result[["movieId", "title", "genres"]].to_dict(
            orient="records"
        )
    )


# =====================================
# 7. MOVIE DETAILS API
# =====================================

@app.route("/movie/<int:movie_id>", methods=["GET"])
def movie_details(movie_id):

    result = movies[movies["movieId"] == movie_id]

    if result.empty:
        return jsonify({"error": "Movie not found"}), 404

    return jsonify(
        result[["movieId", "title", "genres"]].iloc[0].to_dict()
    )


# =====================================
# 8. HYBRID RECOMMENDATION API
# =====================================

@app.route("/recommend", methods=["GET"])
def recommend():

    movie_title = request.args.get("movie")
    user_id = request.args.get("user_id", type=int)
    top_n = request.args.get("top_n", default=10, type=int)

    if not movie_title or user_id is None:
        return jsonify({
            "error": "Provide movie and user_id"
        }), 400

    if movie_title not in movie_indices:
        return jsonify({
            "error": "Movie not found"
        }), 404

    if user_id not in user_movie_matrix.index:
        return jsonify({
            "error": "User not found"
        }), 404

    if not 1 <= top_n <= 50:
        return jsonify({
            "error": "top_n must be between 1 and 50"
        }), 400

    movie_idx = movie_indices[movie_title]

    content_scores = content_similarity[movie_idx]

    similar_users = (
        user_similarity_df[user_id]
        .drop(user_id)
        .sort_values(ascending=False)
        .head(10)
    )

    watched_movies = set(
        user_movie_matrix.loc[user_id].dropna().index
    )

    recommendations = []

    for idx, movie in movies.iterrows():

        movie_id = movie["movieId"]

        if idx == movie_idx:
            continue

        if movie_id in watched_movies:
            continue

        content_score = float(content_scores[idx])

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
            collaborative_score = (
                weighted_sum / similarity_sum / 5
            )
        else:
            collaborative_score = 0

        hybrid_score = (
            0.5 * content_score
            + 0.5 * collaborative_score
        )

        recommendations.append({
            "movieId": int(movie_id),
            "title": movie["title"],
            "genres": movie["genres"],
            "content_score": round(content_score, 4),
            "collaborative_score": round(
                collaborative_score, 4
            ),
            "hybrid_score": round(hybrid_score, 4)
        })

    recommendations.sort(
        key=lambda x: x["hybrid_score"],
        reverse=True
    )

    return jsonify({
        "status": "success",
        "input_movie": movie_title,
        "user_id": user_id,
        "recommendations": recommendations[:top_n]
    })


# =====================================
# 9. RUN SERVER
# =====================================

if __name__ == "__main__":

    Timer(
        1.5,
        lambda: webbrowser.open("http://127.0.0.1:5000/")
    ).start()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )