# Movie Recommender System

## Overview
This project is a **Movie Recommender System** built using **Python, Pandas, Streamlit, and Scikit-learn**. It suggests similar movies based on content similarity using **Natural Language Processing (NLP) techniques**.

## Features
- **Content-Based Recommendation**: Suggests movies similar to the selected movie.
- **Efficient Data Preprocessing**: Cleans and processes the dataset to extract relevant information.
- **Bag-of-Words & Cosine Similarity**: Uses NLP to compare movie tags.
- **Interactive UI**: Built with Streamlit for a smooth user experience.
- **Movie Posters**: Fetches posters using the TMDB API for a visually appealing interface.

## Dataset
The recommender system is trained on the **TMDB 5000 Movies Dataset**, which contains:
- Movie details (title, overview, genres, cast, crew, etc.)
- Movie IDs used to fetch posters via the TMDB API.

###  Install Dependencies
Ensure you have Python installed, then run:
```sh
pip install -r requirements.txt
```

### 3. Download Dataset
Place the **tmdb_5000_movies.csv** and **tmdb_5000_credits.csv** inside the project directory.

### 4. Run the Application
```sh
streamlit run app.py
```

## How It Works
1. **Preprocessing**:
   - Merges two datasets (movies & credits).
   - Extracts essential columns and removes duplicates.
   - Converts categorical data (genres, cast, crew) into a structured format.
   - Creates **tags** for each movie by combining key features.

2. **Feature Engineering & Vectorization**:
   - Applies **stemming** to normalize text.
   - Uses **CountVectorizer** to extract important words.
   - Computes **cosine similarity** between movie tags to find similar movies.

3. **Recommendation & UI**:
   - User selects a movie from the dropdown.
   - The system fetches the top 5 most similar movies.
   - Displays posters fetched dynamically using TMDB API.

## Technologies Used
- **Python** (Pandas, NumPy, Scikit-learn, NLTK, Requests)
- **Streamlit** (For UI & interaction)
- **TMDB API** (For fetching movie posters)

## Future Improvements
- Add **Hybrid Filtering** (Collaborative + Content-Based)
- Improve **Text Preprocessing** with advanced NLP techniques
- Deploy the application on **Heroku** or **Streamlit Cloud**

## License
This project is open-source and available under the **MIT License**.

---
**Author:** Simranjit Kaur  
GitHub: [Simranjit15kaur](https://github.com/Simranjit15kaur)  


