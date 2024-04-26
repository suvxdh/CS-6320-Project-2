import json
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
import matplotlib.pyplot as plt

# Load JSON data
with open('intents.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Prepare lists for training data
instructions = [item['INSTRUCTION'] for item in data]
responses = [item['RESPONSE'] for item in data]

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(instructions + responses)
vocab_size = len(tokenizer.word_index) + 1

# Convert text to sequences of integers
seq_instructions = tokenizer.texts_to_sequences(instructions)
seq_responses = tokenizer.texts_to_sequences(responses)

# Pad sequences for uniform input size
max_seq_length = max(max(len(seq) for seq in seq_instructions), max(len(seq) for seq in seq_responses))
seq_instructions = pad_sequences(seq_instructions, maxlen=max_seq_length, padding='post')
seq_responses = pad_sequences(seq_responses, maxlen=max_seq_length, padding='post')

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(seq_instructions, seq_responses, test_size=0.2, random_state=42)

# Define the model with adjusted parameters
model = Sequential()
model.add(Embedding(vocab_size, 200))  # Increased embedding dimension to 200
model.add(LSTM(256, return_sequences=True))  # Increased LSTM units to 256
model.add(Dropout(0.4))  # Increased dropout rate to 0.4
model.add(LSTM(256, return_sequences=True))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the updated model with reshaped target tensors
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15, batch_size=64) 

# Save the model architecture to JSON
model_json = model.to_json()
with open("model_architecture.json", "w") as json_file:
    json_file.write(model_json)

# Save the trained model weights
model.save_weights("model_weights.h5")

# Plotting the updated learning curve
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model accuracy during training')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()
