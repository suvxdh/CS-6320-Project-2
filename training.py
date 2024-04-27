import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import GRU
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Function to load and clean text data
def load_and_clean_data(filename):
    with open(filename) as file:
        data = json.load(file)
    instructions = [item['Context'] for item in data]
    responses = [item['Response'] for item in data]
    
    # Clean the instructions
    instructions = [clean_text(text) for text in instructions]
    return instructions, responses

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove non-letters
    words = text.lower().split()  # Convert to lower case, split into words
    stops = set(stopwords.words("english"))  # Load the list of stopwords
    meaningful_words = [w for w in words if not w in stops]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()  # Lemmatize words
    lemmatized_words = [lemmatizer.lemmatize(w) for w in meaningful_words]
    return " ".join(lemmatized_words)

# Function to prepare the text data
def prepare_text_data(texts, num_words, max_len):
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len)
    return tokenizer, padded_sequences

def build_model(input_dim, output_dim):
    model = Sequential([
        Embedding(input_dim, 50),
        GRU(256),  
        Dense(256, activation='relu'),
        Dropout(0.3),  
        Dense(output_dim, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Main execution flow
if __name__ == "__main__":
    instructions, responses = load_and_clean_data('combined_dataset.json')
    num_words = 10000
    max_len = 20

    # Prepare the data
    tokenizer, padded_instructions = prepare_text_data(instructions, num_words, max_len)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(responses)
    categorical_labels = to_categorical(labels)

    # Build and train the model
    model = build_model(num_words + 1, len(set(labels)))
    model.fit(padded_instructions, categorical_labels, epochs=125, batch_size=256)

    # Save the model and tokenizer
    model.save('intent_model.h5')
    with open('tokenizer.json', 'w') as f:
        json.dump(tokenizer.to_json(), f)
    with open('label_encoder.json', 'w') as f:
        json.dump(label_encoder.classes_.tolist(), f)

    print("Model and tokenizer saved successfully.")