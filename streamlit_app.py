
import os
import re
import requests
import pandas as pd
import streamlit as st

from hybrid_recommendation import hybrid_recommend

st.set_page_config(
    page_title="CineMatch | Discover Movies",
    page_icon="🎬",
    layout="wide"
)

TMDB_API_KEY="6c441138aea6fce6808058e44978b18f"
        
# -------------------- STYLING --------------------
st.markdown("""
<style>
.stApp {
    background: #090d18;
    color: #f5f6fc;
}
.block-container {
    max-width: 1450px;
    padding-top: 1.5rem;
}
.hero {
    padding: 34px;
    border-radius: 22px;
    background: linear-gradient(120deg, #201b42, #111a30 65%, #42213f);
    border: 1px solid #373452;
    margin-bottom: 28px;
}
.hero-label {
    color: #f9bb72;
    font-size: 12px;
    letter-spacing: 3px;
    font-weight: 700;
}
.hero h1 {
    color: white;
    font-size: 42px;
    margin: 10px 0;
}
.hero p {
    color: #c3c8dd;
    font-size: 16px;
}
.movie-title {
    font-size: 17px;
    font-weight: 700;
    color: #ffffff;
    min-height: 48px;
}
.movie-meta {
    color: #a9b1ca;
    font-size: 13px;
}
.score {
    color: #69e5b5;
    font-weight: 700;
    font-size: 20px;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #12192a;
    border: 1px solid #28334c;
    border-radius: 15px;
    padding: 12px;
}
.stButton button[kind="primary"] {
    border-radius: 10px;
    min-height: 44px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)


# -------------------- TMDB POSTERS --------------------


@st.cache_data(ttl=86400, show_spinner=False)
def get_movie_poster(title, api_key):
    if not api_key:
        return None, None, None

    match = re.search(r"\((\d{4})\)\s*$", title)
    year = match.group(1) if match else None
    clean_title = re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip()

    try:
        for search_year in (year, None):
            params = {
                "api_key": api_key,
                "query": clean_title,
                "include_adult": "false",
            }

            if search_year:
                params["year"] = search_year

            response = requests.get(
                "https://api.themoviedb.org/3/search/movie",
                params=params,
                timeout=15,
            )
            response.raise_for_status()
            candidates = response.json().get("results", [])

            # Prefer the correct release year and a usable poster.
            with_poster = [
                m for m in candidates
                if m.get("poster_path")
            ]

            selected = next(
                (
                    m for m in with_poster
                    if year
                    and m.get("release_date", "")[:4] == year
                ),
                with_poster[0] if with_poster else None,
            )

            if selected:
                poster_url = (
                    "https://image.tmdb.org/t/p/w500"
                    + selected["poster_path"]
                )
                release_date = selected.get("release_date", "")
                actual_year = release_date[:4] if release_date else year

                return (
                    poster_url,
                    actual_year,
                    selected.get("overview", ""),
                )

        return None, year, None

    except requests.RequestException as error:
        print(f"TMDB error for {title}: {error}")
        return None, year, None

# -------------------- HEADER --------------------
st.markdown("""
<div class="hero">
    <div class="hero-label">CINEMATCH · PERSONALIZED DISCOVERY</div>
    <h1>🎬 Find your next favourite.</h1>
    <p>Discover films picked for you using hybrid recommendation intelligence.</p>
</div>
""", unsafe_allow_html=True)


# -------------------- SEARCH --------------------
st.subheader("Find your next movie")

with st.container(border=True):
    movie_title = st.text_input(
        "Movie title",
        placeholder="Try Avatar or Toy Story (1995)"
    )

    col1, col2 = st.columns(2)

    with col1:
        user_id = st.number_input(
            "User ID",
            min_value=1,
            value=1,
            step=1
        )

    with col2:
        top_n = st.select_slider(
            "Movies to recommend",
            options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
             11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            value=8
        )

    alpha = st.slider(
        "Content-based vs collaborative filtering",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1
    )

    st.caption(
        f"Content-based: {alpha:.0%}  ·  "
        f"Collaborative: {1-alpha:.0%}"
    )

    search_clicked = st.button(
        "✨ Discover Movies",
        type="primary",
        use_container_width=True
    )


# -------------------- RESULTS --------------------
if search_clicked:
    if not movie_title.strip():
        st.warning("Enter a movie title first.")
    else:
        try:
            with st.spinner("Finding your next favourite..."):
                results = hybrid_recommend(
                    movie_title.strip(),
                    int(user_id),
                    top_n=int(top_n),
                    alpha=float(alpha)
                )

            if isinstance(results, pd.DataFrame) and not results.empty:
                st.markdown("## ✨ Recommended for you")
                st.caption(
                    f"{len(results)} movies selected by your recommendation model."
                )

                if not TMDB_API_KEY:
                    st.info(
                        "Posters are disabled. Set the TMDB_API_KEY "
                        "environment variable to show movie posters."
                    )

                for start in range(0, len(results), 4):
                    batch = results.iloc[start:start + 4]
                    columns = st.columns(4)

                    for col, (_, movie) in zip(columns, batch.iterrows()):
                        title = str(movie["title"])
                        genres = str(movie.get("genres", ""))
                        score = float(movie["hybrid_score"])

                        poster, year, overview = get_movie_poster(
                            title, TMDB_API_KEY
                        )

                        with col:
                            with st.container(border=True):
                                if poster:
                                    st.image(poster, use_container_width=True)
                                else:
                                    st.markdown(
                                        "<div style='height:250px;display:flex;"
                                        "align-items:center;justify-content:center;"
                                        "background:#202941;border-radius:10px;"
                                        "font-size:48px'>🎞️</div>",
                                        unsafe_allow_html=True
                                    )

                                st.markdown(
                                    f"<div class='movie-title'>"
                                    f"{title}</div>",
                                    unsafe_allow_html=True
                                )

                                if year:
                                    st.markdown(
                                        f"<div class='movie-meta'>{year}</div>",
                                        unsafe_allow_html=True
                                    )

                                st.caption(genres.replace("|", " · "))

                                st.markdown(
                                    f"<div class='score'>{score:.1%} "
                                    f"<span style='font-size:12px;color:#a9b1ca'>"
                                    f"ranking score</span></div>",
                                    unsafe_allow_html=True
                                )

                                with st.expander("More details"):
                                    st.write(
                                        overview or
                                        "No synopsis available from TMDB."
                                    )
                                    st.write(
                                        f"Content score: "
                                        f"{float(movie['content_score']):.3f}"
                                    )
                                    st.write(
                                        f"Collaborative score: "
                                        f"{float(movie['collaborative_score']):.3f}"
                                    )

                st.caption(
                    "Movie metadata and poster images provided by TMDB. "
                    "This product uses the TMDB API but is not endorsed "
                    "or certified by TMDB."
                )

            elif isinstance(results, pd.DataFrame):
                st.info("No recommendations found. Try another movie or user.")
            else:
                st.warning(str(results))

        except Exception as error:
            st.error(f"Could not generate recommendations: {error}")

st.divider()
st.caption("CineMatch · A hybrid movie recommendation project")
