from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)

# Load data
df = pd.read_csv('./data/question-suggestion/data.csv')
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
    suggested_questions = np.random.choice(suggested_questions, num_suggestions, replace=False).tolist()  # Konversi ke list
    
    return suggested_questions, top_category, top_probability


@app.route('/suggest', methods=['POST'])
def suggest():
    data = request.json
    user_input = data.get('user_input', '')
    total_questions = data.get('total_questions', 4)
    
    if not user_input:
        return jsonify({"error": "User input is required"}), 400
    
    suggested_questions, top_category, top_probability = suggest_questions(user_input, total_questions)
    
    response = {
        "top_category": top_category,
        "probability": float(top_probability),  # Konversi numpy float32 ke float bawaan
        "suggested_questions": suggested_questions  # Tidak perlu konversi ke list dan panggilan tolist()
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
