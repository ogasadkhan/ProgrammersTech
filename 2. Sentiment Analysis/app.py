import streamlit as st
import joblib
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK resources (you only need to do this once)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load the trained RandomForest model
with open('D:/X/AI/Internships/ProgrammingTech/2_Sentiment Analysis/rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

# Load the TF-IDF vectorizer (replace 'vectorizer.pkl' with your saved vectorizer file)
with open('D:/X/AI/Internships/ProgrammingTech/2_Sentiment Analysis/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Initialize NLTK's WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

# Function for NLP preprocessing
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)

    # Lowercasing
    tokens = [token.lower() for token in tokens]

    # Removing punctuation
    tokens = [token for token in tokens if token not in string.punctuation]

    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatization
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join tokens into a single string
    processed_text = ' '.join(tokens)

    return processed_text

# Streamlit app
def main():
    st.title("Sentiment Analysis")
    # st.write("Enter a sentence to predict its sentiment category.")

    # Input text box for user input
    user_input = st.text_input("Enter a sentence:")


    # Button to trigger sentiment analysis
    if st.button("Predict"):
        if user_input:
            # Preprocess the user input
            processed_sentence = preprocess_text(user_input)

            # Transform the preprocessed sentence using the loaded vectorizer
            sentence_tfidf = vectorizer.transform([processed_sentence])

            # Predict sentiment label for the sentence
            predicted_sentiment = rf_model.predict(sentence_tfidf)

            # Mapping dictionary
            sentiment_mapping_reverse = {
                -1: 'Negative',
                0: 'Neutral',
                1: 'Positive',
                2: 'Irrelevant'
            }

            # Print the predicted sentiment category
            predicted_category = sentiment_mapping_reverse.get(predicted_sentiment[0], 'Unknown')
            st.write(predicted_category)

# Run the Streamlit app
if __name__ == "__main__":
    main()
