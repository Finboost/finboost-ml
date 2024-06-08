import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load data
df = pd.read_csv('./data/data.csv')
questions = df['Question'].tolist()
labels = df['Category'].tolist()

# Load tokenizer and label dictionary
with open('./models/question-suggestion/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('./models/question-suggestion/label_dict.pickle', 'rb') as handle:
    label_dict = pickle.load(handle)

num_classes = len(label_dict)

# Load the model
model = tf.keras.models.load_model('./models/question-suggestion/model_question_suggestion.h5')

# Function to suggest questions
def suggest_questions(user_input, total_questions=4):
    sequence = tokenizer.texts_to_sequences([user_input])
    padded_sequence = pad_sequences(sequence, maxlen=pad_sequences(tokenizer.texts_to_sequences(questions)).shape[1], padding='post')
    predictions = model.predict(padded_sequence)
    labels = list(label_dict.keys())
    
    # Get the top category with the highest probability
    top_category_idx = np.argmax(predictions[0])
    top_category = labels[top_category_idx]
    top_probability = predictions[0][top_category_idx]

    print("\nTop category with its probability:")
    print(f'{top_category}: {top_probability:.4f}')
    
    # Filter questions from the dataset based on the top category
    suggested_questions = df[df['Category'] == top_category]['Question'].tolist()
    
    # Ensure that the total number of suggestions does not exceed available questions
    num_suggestions = min(total_questions, len(suggested_questions))
    
    # Select random questions from the suggested list, ensuring they belong to the top category
    return np.random.choice(suggested_questions, num_suggestions, replace=False)

# Example usage
user_input = "Apa itu PPh 21 dan bagaimana cara menghitungnya?"
suggested_questions = suggest_questions(user_input)
print("\nSuggested questions:")
for question in suggested_questions:
    print(question)
