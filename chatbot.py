import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import random
import os
import time
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_model_and_components(model_path, tokenizer_path, label_encoder_path):
    # Load the trained model
    model = load_model(model_path)

    # Load the tokenizer
    with open(tokenizer_path) as file:
        tokenizer_data = json.load(file)
        tokenizer = tokenizer_from_json(tokenizer_data)  # Pass the dictionary directly
    
    # Load the label encoder
    with open(label_encoder_path) as file:
        classes = json.load(file)
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.array(classes)

    return model, tokenizer, label_encoder

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Lemmatize tokens
    return " ".join(tokens)

def predict_intent(text, model, tokenizer, label_encoder, max_len=20):
    preprocessed_text = preprocess_text(text)
    # Tokenize and pad the preprocessed text
    sequence = tokenizer.texts_to_sequences([preprocessed_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    
    # Predict the category
    prediction = model.predict(padded_sequence)
    intent_index = np.argmax(prediction)
    intent = label_encoder.inverse_transform([intent_index])[0]
    return intent

def responses(intent, user_input, corpus_file):
    print("Extracted Intent:", intent)  # Debug print
    print("User Input:", user_input)  # Debug print
    
    if intent in corpus_file:
        responses_list = corpus_file[intent]
        if len(responses_list) > 1:
            # Calculate TF-IDF vectors for user input and intent responses
            tfidf_vectorizer = TfidfVectorizer()
            all_responses = [user_input] + responses_list
            tfidf_matrix = tfidf_vectorizer.fit_transform(all_responses)

            # Calculate cosine similarity between user input and responses
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            # Sort responses based on similarity (higher is better)
            sorted_indices = np.argsort(similarities)[::-1]

            # Choose the most similar response
            most_similar_response = responses_list[sorted_indices[0]]
            return most_similar_response
        else:
            return responses_list[0]
    else:
        return "Sorry, I don't have information on that topic."


def main():
    model_path = 'intent_model.h5'
    tokenizer_path = 'tokenizer.json'
    label_encoder_path = 'label_encoder.json'
    intents_json_path = 'intents.json'

    # Load model and components
    model, tokenizer, label_encoder = load_model_and_components(model_path, tokenizer_path, label_encoder_path)

    # Load intents data
    with open(intents_json_path) as file:
        intents_data = json.load(file)

    # Convert intents data to a dictionary for easy access
    corpus_file = {}
    for entry in intents_data:
        corpus_file[entry['INSTRUCTION']] = entry['RESPONSE']

    print("WeekndBot: Hello! I am WeekndBot. Please enter your name for the best experience, or enter 'bye' to exit.")
    user_input = input("You: ")

    if user_input.lower() == 'bye':
        print("WeekndBot: Goodbye!")
        return

    userName = user_input.lower() + '.txt'
    if os.path.exists(userName):
        print(f"Welcome back, {user_input}!")
    else:
        with open(userName, 'w') as f:
            f.write(f"User: {user_input}\n")

    print("WeekndBot: What would you like to know about The Weeknd? Enter 'bye' to exit.")

    while True:
        user_input = input("You: ")

        if user_input.lower() == 'bye':
            print("WeekndBot: Goodbye!")
            break

        intent = predict_intent(user_input, model, tokenizer, label_encoder)
        bot_response = responses(intent, user_input, corpus_file)
        print("WeekndBot:", bot_response)

if __name__ == "__main__":
    main()
