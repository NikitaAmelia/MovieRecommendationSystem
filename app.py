import streamlit as st
import pickle
import requests
import os

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Movie Recommendation System",
    layout="centered"
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
        color: white;
    }

    section[data-testid="stSidebar"] {
        background-color: #161b22;
    }

    h1, h2, h3, h4, h5, h6, p, label, span {
        color: white;
    }

    }
    div[data-baseweb="select"] input {
    color: white !important;
    caret-color: white;
    background-color: #161b22 !important;
    }

    /* Value yang sudah dipilih */
    div[data-baseweb="select"] span {
    color: white !important;
    }
    
    /* Placeholder */
    div[data-baseweb="select"] input::placeholder {
    color: #9ca3af !important;
    }
    
    /* Dropdown menu */
    div[data-baseweb="menu"] {
    background-color: #161b22 !important;
    }
    
    /* Item dropdown */
    div[data-baseweb="option"] {
    color: white !important;
    }
    
    /* Hover item */
    div[data-baseweb="option"]:hover {
    background-color: #21262d !important;
    }

    /* BUTTON */
    div.stButton > button {
        background-color: #e50914;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6em 1.2em;
        font-weight: bold;
    }

    div.stButton > button:hover {
        background-color: #b20710;
        color: white;
    }

    div.stButton > button:active {
        background-color: #e50914;
        color: white;
    }

    div.stButton > button:focus {
        background-color: #e50914;
        color: white;
        outline: none;
        box-shadow: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# API KEY
# =========================
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

if not TMDB_API_KEY:
    st.error("TMDB API Key belum diset. Gunakan environment variable TMDB_API_KEY.")
    st.stop()

# =========================
# LOAD DATA
# =========================
movies = pickle.load(open("movies.pkl", "rb"))

if os.path.exists("similarity.pkl"):
    similarity = pickle.load(open("similarity.pkl", "rb"))
else:
    st.warning("Similarity model not found. Generating similarity matrix...")
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["overview"].fillna(""))
    similarity = cosine_similarity(tfidf_matrix)


# =========================
# FUNCTIONS
# =========================
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
    data = requests.get(url).json()
    poster_path = data.get("poster_path")
    return f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None


def recommend(movie_title):
    index = movies[movies["title"] == movie_title].index[0]
    distances = sorted(
        list(enumerate(similarity[index])),
        key=lambda x: x[1],
        reverse=True
    )

    names, posters = [], []

    for i in distances[1:6]:
        movie_id = movies.iloc[i[0]].id
        names.append(movies.iloc[i[0]].title)
        posters.append(fetch_poster(movie_id))

    return names, posters

# =========================
# UI
# =========================
st.title("🎬 Movie Recommendation System")
st.write("Get movie recommendations based on content similarity")

selected_movie = st.selectbox(
    "Select a movie you like:",
    movies["title"].values
)

if st.button("Recommend"):
    names, posters = recommend(selected_movie)

    st.subheader("🎥 Recommended Movies")
    cols = st.columns(5)

    for col, name, poster in zip(cols, names, posters):
        with col:
            if poster:
                st.image(poster, use_container_width=True)
            st.caption(name)
