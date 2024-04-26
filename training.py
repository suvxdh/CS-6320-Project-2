import json
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, TimeDistributed
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load intents from JSON file
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Extract patterns and responses from intents
patterns = []
responses = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        responses.append(random.choice(intent['responses']))

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(patterns)
vocab_size = len(tokenizer.word_index) + 1

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(patterns)
max_sequence_len = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_len, padding='post')
print("is the max sequence length size: ", max_sequence_len)
# Convert responses to sequences
response_sequences = tokenizer.texts_to_sequences(responses)
padded_response_sequences = pad_sequences(response_sequences, maxlen=max_sequence_len, padding='post')

# Convert sequences to one-hot encoding
responses_array = np.zeros((len(padded_response_sequences), max_sequence_len, vocab_size), dtype=np.int8)
for i, seq in enumerate(padded_response_sequences):
    for j, token_index in enumerate(seq):
        responses_array[i, j, token_index] = 1

# Model architecture with TimeDistributed layer
model = Sequential([
    Embedding(vocab_size, 128),
    LSTM(128, return_sequences=True),
    TimeDistributed(Dense(vocab_size, activation='softmax'))  # Apply Dense layer to each time step
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training with history to plot learning curve
history = model.fit(padded_sequences, responses_array, epochs=450, validation_split=0.2)

# Save the model
model.save('chatbot_model.h5')

# Plotting the learning curve
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model accuracy during training')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# Printing the final training and validation accuracy
final_train_accuracy = history.history['accuracy'][-1]
final_val_accuracy = history.history['val_accuracy'][-1]
print("Final Training Accuracy: {:.2f}%".format(final_train_accuracy * 100))
print("Final Validation Accuracy: {:.2f}%".format(final_val_accuracy * 100))
