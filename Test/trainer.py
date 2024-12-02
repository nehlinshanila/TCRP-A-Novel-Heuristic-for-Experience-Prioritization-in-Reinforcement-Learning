# pip install pandas matplotlib numpy tensorflow datasets transformers scikit-learn rouge
import numpy as np
import pandas as pd
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from datasets import load_dataset
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

tf.debugging.set_log_device_placement(True)
K.clear_session()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# from rouge import Rouge

# 1. Data Preparation

# Load the xlsum dataset from Hugging Face
dataset = load_dataset("csebuetnlp/xlsum", 'bengali')

# Extract the Bengali portion of the dataset
bengali_dataset = dataset["train"]

# Convert the dataset to pandas DataFrames
train_df = pd.DataFrame(bengali_dataset)

# Split data into training and testing sets
train_texts, test_texts, train_summaries, test_summaries = train_test_split(
    train_df['text'], train_df['summary'], test_size=0.2, random_state=42
)

# 2. Text Preprocessing

# Tokenize the text
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(train_texts)
train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

# Get the max length of the train sequence variable
max_length = 100
for ts in train_sequences:
    if max_length < len(ts):
        max_length = len(ts)

# Pad sequences to have the same length
# max_length = 100  # Adjust as needed
train_padded = pad_sequences(train_sequences, maxlen=max_length)
test_padded = pad_sequences(test_sequences, maxlen=max_length)

# Tokenize and pad summaries (similar to above)
tokenizer_summaries = Tokenizer()
tokenizer_summaries.fit_on_texts(train_summaries)
train_summary_sequences = tokenizer_summaries.texts_to_sequences(train_summaries)
test_summary_sequences = tokenizer_summaries.texts_to_sequences(test_summaries)
train_summary_padded = pad_sequences(train_summary_sequences, maxlen=max_length)
test_summary_padded = pad_sequences(test_summary_sequences, maxlen=max_length)

# 3. Model Building

# Define the LSTM model
model = Sequential()
model.add(Embedding(len(tokenizer.word_index) + 1, 64))
model.add(LSTM(64, return_sequences=True))
model.add(Dense(len(tokenizer_summaries.word_index) + 1, activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. Model Training

# Train the model
model.fit(train_padded, train_summary_padded, epochs=10, batch_size=32)  # Adjust epochs and batch size

# 5. Model Evaluation and Prediction

# Evaluate the model
loss, accuracy = model.evaluate(test_padded, test_summary_padded)
print('Loss:', loss)
print('Accuracy:', accuracy)


# Generate summaries for new text (needs improvement - see notes below)
def generate_summary(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length)
    predicted_sequence = np.argmax(model.predict(padded_sequence), axis=-1)
    # Convert predicted sequence back to text
    summary = tokenizer_summaries.sequences_to_texts(predicted_sequence)
    return summary


# Example usage
new_text = "আজকের আবহাওয়া খুব সুন্দর।"
# new_text = "The weather is really good today. I have enjoyed it thoroughly."
summary = generate_summary(new_text)
print(summary)

















# Calculate ROUGE scores
# rouge = Rouge()
# rouge_scores = []
# for i in range(len(test_texts)):
#     reference_summary = test_summaries.iloc[i]  # Access the reference summary
#     generated_summary = generate_summary(test_texts.iloc[i])
#     scores = rouge.get_scores(generated_summary, reference_summary)
#     rouge_scores.append(scores[0])

# # Aggregate ROUGE scores (e.g., average)
# avg_rouge_scores = {
#     'rouge-1': np.mean([score['rouge-1']['f'] for score in rouge_scores]),
#     'rouge-2': np.mean([score['rouge-2']['f'] for score in rouge_scores]),
#     'rouge-l': np.mean([score['rouge-l']['f'] for score in rouge_scores])
# }
# print("Average ROUGE scores:", avg_rouge_scores)
