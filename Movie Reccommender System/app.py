import streamlit as st
import pickle
import pandas as pd
import requests
import ssl
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Enable SSL certificate verification
ssl._create_default_https_context = ssl._create_default_https_context


def create_session():
    """Creates a session with retry logic for requests."""
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session


@st.cache_data  # Cache the function to avoid repeated API calls
def fetch_poster(movie_id):
    session = create_session()
    try:
        response = session.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=fee194d79574cdec6a82fdd3ef89a90b",
            timeout=10,
            verify=True  # SSL certificate verification enabled
        )
        response.raise_for_status()
        data = response.json()
        poster_url = "https://image.tmdb.org/t/p/w500/" + data.get('poster_path', "")
        if not poster_url.endswith('.jpg'):  # If no valid poster URL, use the placeholder
            poster_url = "https://via.placeholder.com/500x750?text=No+Image+Available"
        return poster_url
    except requests.exceptions.RequestException as e:
        print(f"Error fetching poster: {e}")
        return "https://via.placeholder.com/500x750?text=No+Image+Available"


def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_movies_posters = []
    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id

        recommended_movies.append(movies.iloc[i[0]].title)
        # fetch poster from API
        recommended_movies_posters.append(fetch_poster(movie_id))  # Fixed the call to fetch poster using movie_id
    return recommended_movies, recommended_movies_posters


# Load preprocessed data
movie_dict = pickle.load(open('movie_dict.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))
movies = pd.DataFrame(movie_dict)

# Streamlit app interface
st.title('🎬 Movie Recommender System')

# Custom CSS Styling
st.markdown(
    """
    <style>
        body {
            background-color: #121212;
            color: #ffffff;
            font-family: 'Arial', sans-serif;
        }

        .stImage {
            display: flex;
            justify-content: center;
            gap: 20px;
            flex-wrap: nowrap;
            overflow-x: auto;
            padding: 10px;
            white-space: nowrap;
        }

        .movie-card {
            display: inline-block;
            text-align: center;
            width: 200px;
        }

        .movie-card img {
            width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(255, 255, 255, 0.2);
        }

        h1 {
            text-align: center;
            color: #ffcc00;
            text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.7);
        }

        select, button {
            font-size: 16px;
            padding: 10px;
        }

        button {
            background-color: #ffcc00;
            color: black;
            font-weight: bold;
            border-radius: 5px;
            border: none;
            cursor: pointer;
        }

        button:hover {
            background-color: #ff9900;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Selectbox for movie selection
selected_movie_name = st.selectbox(
    'Select a movie:', movies['title'].values
)

if st.button('Recommend'):
    names, posters = recommend(selected_movie_name)

    if len(names) == 5 and len(posters) == 5:
        cols = st.columns(5)  # Create 5 equal columns
        for col, name, poster in zip(cols, names, posters):
            with col:
                st.image(poster, caption=name, use_container_width=True)  # Updated here
    else:
        st.error("Error: Less than 5 recommendations available.")

