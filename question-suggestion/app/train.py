import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('./data/data.csv')
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

questions = df['Question'].tolist()
labels = df['Category'].tolist()

label_dict = {label: idx for idx, label in enumerate(set(labels))}
numerical_labels = [label_dict[label] for label in labels]

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(questions)
sequences = tokenizer.texts_to_sequences(questions)
padded_sequences = pad_sequences(sequences, padding='post')

numerical_labels = np.array(numerical_labels)

X_train, X_val, y_train, y_val = train_test_split(padded_sequences, numerical_labels, test_size=0.2, random_state=42)

# Define the model
def create_model(num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=10000, output_dim=64),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

num_classes = len(label_dict)
model = create_model(num_classes)

history = model.fit(X_train, y_train, epochs=500, validation_data=(X_val, y_val))

# Save the model and tokenizer
model.save('./question-suggestion/models/question-suggestion/model_question_suggestion.h5')
import pickle
with open('./question-suggestion/models/question-suggestion/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('./question-suggestion/models/question-suggestion/label_dict.pickle', 'wb') as handle:
    pickle.dump(label_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
