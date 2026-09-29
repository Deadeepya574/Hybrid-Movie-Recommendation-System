const API_URL = "";

let selectedMovie = null;

// Search movies
document.getElementById("searchBtn").addEventListener("click", searchMovies);

async function searchMovies() {

    const query = document.getElementById("movieSearch").value.trim();
    const resultsDiv = document.getElementById("searchResults");

    if (!query) {
        resultsDiv.textContent = "Please enter a movie name.";
        return;
    }

    resultsDiv.textContent = "Searching...";

    try {

        const response = await fetch(
            `${API_URL}/search?q=${encodeURIComponent(query)}`
        );

        if (!response.ok) {
            throw new Error("Search request failed");
        }

        const movies = await response.json();

        resultsDiv.replaceChildren();

        if (movies.length === 0) {
            resultsDiv.textContent = "No movies found.";
            return;
        }

        movies.forEach(movie => {

            const button = document.createElement("button");

            button.className = "search-item";
            button.textContent = `${movie.title} | ${movie.genres}`;

            button.addEventListener("click", () => {

                selectedMovie = movie.title;

                document.getElementById("selectedMovie").textContent =
                    `Selected Movie: ${movie.title}`;

                resultsDiv.replaceChildren();

            });

            resultsDiv.appendChild(button);

        });

    } catch (error) {

        resultsDiv.textContent =
            "Could not connect to Flask. Make sure app.py is running.";

        console.error(error);
    }
}


// Get recommendations
document.getElementById("recommendBtn")
    .addEventListener("click", getRecommendations);

async function getRecommendations() {

    const userId = document.getElementById("userId").value;
    const resultsDiv = document.getElementById("recommendations");

    if (!selectedMovie) {
        resultsDiv.textContent = "Please search and select a movie first.";
        return;
    }

    if (!userId || Number(userId) < 1) {
        resultsDiv.textContent = "Please enter a valid User ID.";
        return;
    }

    resultsDiv.textContent = "Generating recommendations...";

    try {

        const url =
            `${API_URL}/recommend?movie=${encodeURIComponent(selectedMovie)}` +
            `&user_id=${encodeURIComponent(userId)}&top_n=10`;

        const response = await fetch(url);

        if (!response.ok) {
            throw new Error("Recommendation request failed");
        }

        const data = await response.json();

        resultsDiv.replaceChildren();

        if (!data.recommendations || data.recommendations.length === 0) {
            resultsDiv.textContent = "No recommendations found.";
            return;
        }

        data.recommendations.forEach(movie => {

            const card = document.createElement("div");
            card.className = "movie-card";

            const title = document.createElement("h3");
            title.textContent = movie.title;

            const genres = document.createElement("p");
            genres.textContent = `Genres: ${movie.genres}`;

            const score = document.createElement("p");
            score.textContent =
                `Hybrid Score: ${Number(movie.hybrid_score).toFixed(3)}`;

            card.append(title, genres, score);

            resultsDiv.appendChild(card);

        });

    } catch (error) {

        resultsDiv.textContent =
            "Recommendation failed. Check your Flask backend.";

        console.error(error);
    }
}