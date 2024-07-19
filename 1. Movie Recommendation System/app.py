import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz, process

# Load your pre-trained model components (TfidfVectorizer and similarity matrix)
# Example: Load TfidfVectorizer and similarity matrix
def load_model():
    # Load your TfidfVectorizer and other necessary components here
    # Replace with your actual loading mechanism
    tfidf_vectorizer = TfidfVectorizer()
    # Load or compute your similarity matrix here
    
    similarity_matrix = np.load("D:/X/AI/Internships/ProgrammingTech/1_Movie Recommendation/similarity_matrix.npy")  # Example loading from file
    
    return tfidf_vectorizer, similarity_matrix

# Function to get similar movies
def get_similar_movies(movie_title, similarity_matrix, movies_df, top_n=5):
    # Check if the movie title exists in the DataFrame
    if movie_title not in movies_df['title'].values:
        # Perform fuzzy matching to find the closest match
        matches = process.extract(movie_title, movies_df['title'].values, limit=5, scorer=fuzz.token_sort_ratio)
        closest_match = matches[0][0]  # Select the closest match
        st.warning(f"Movies similar to '{closest_match}'")
        movie_title = closest_match  # Use the closest match
    
    # Retrieve the index of the movie title
    movie_index = movies_df[movies_df['title'] == movie_title].index[0]
    
    # Calculate similarity scores and find similar movies
    similarity_scores = list(enumerate(similarity_matrix[movie_index]))
    similar_movies_indices = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    similar_movies = [movies_df.iloc[i[0]]['title'] for i in similar_movies_indices]
    
    return similar_movies

# Load data (if needed) - This could be a placeholder
def load_data():
    # Example loading data into a DataFrame
    df = pd.read_pickle("D:/X/AI/Internships/ProgrammingTech/1_Movie Recommendation/dataframe.pkl")
    return df

# Main function to run the Streamlit app
def main():
    st.title('Movie Recommendation System')

    # Load your model components
    tfidf_vectorizer, similarity_matrix = load_model()

    # Load your data into a DataFrame
    df = load_data()

    # Input for movie title
    input_movie = st.text_input('Enter a movie title:', 'Toy Story (1995)')

    if st.button('Recommend'):
        # Get similar movies based on input
        similar_movies = get_similar_movies(input_movie, similarity_matrix, df)
        
        # Display results
        st.subheader(f'Movies similar to "{input_movie}":')
        for idx, movie in enumerate(similar_movies):
            st.write(f"{idx + 1}. {movie}")
    
if __name__ == '__main__':
    main()
