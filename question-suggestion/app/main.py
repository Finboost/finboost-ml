import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

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

# Function to suggest questions based on user input and profile data
def suggest_questions(user_input=None, profile_data=None, total_questions=4):
    labels = list(label_dict.keys())
    suggested_questions = []
    top_category = None
    top_probability = None

    # If user_input is valid and not a placeholder
    if user_input and user_input != "-":
        sequence = tokenizer.texts_to_sequences([user_input])
        padded_sequence = pad_sequences(sequence, maxlen=pad_sequences(tokenizer.texts_to_sequences(questions)).shape[1], padding='post')
        predictions = model.predict(padded_sequence)
        top_category_idx = np.argmax(predictions[0])
        top_category = labels[top_category_idx]
        top_probability = predictions[0][top_category_idx]

        # Filter questions based on the top category
        suggested_questions.extend(df[df['Category'] == top_category]['Question'].tolist())
    
    # If profile_data is provided and not empty, prioritize profile data
    if profile_data and any(value not in ["0", "-", ""] for value in profile_data.values()):
        profile_based_questions = []
        for key, value in profile_data.items():
            if value and value not in ["0", "-", ""]:
                profile_questions = df[df['Question'].str.contains(value, case=False, na=False)]['Question'].tolist()
                profile_based_questions.extend(profile_questions)
        
        # If no user_input or user_input is placeholder, use profile_based_questions
        if not user_input or user_input == "-":
            suggested_questions = profile_based_questions
        
        # If user_input was used, combine both suggestions, prioritize user_input based suggestions
        else:
            suggested_questions = list(set(suggested_questions) | set(profile_based_questions))
    
    # If no suggestions found, provide random questions
    if not suggested_questions:
        suggested_questions = df['Question'].tolist()
    
    # Ensure the total number of suggestions does not exceed available questions
    num_suggestions = min(total_questions, len(suggested_questions))
    
    # Select random questions from the suggested list
    suggested_questions = np.random.choice(suggested_questions, num_suggestions, replace=False).tolist()
    
    return suggested_questions, top_category, top_probability

@app.route('/suggest', methods=['POST'])
def suggest():
    data = request.json
    user_input = data.get('user_input', '')
    total_questions = data.get('total_questions', 4)
    profile_data = {
        'incomePerMonth': data.get('incomePerMonth', ''),
        'investments': data.get('investments', ''),
        'totalSavings': data.get('totalSavings', ''),
        'totalDebts': data.get('totalDebts', ''),
        'insurances': data.get('insurances', '')
    }

    # If user_input is missing or a placeholder, use profile data for suggestions
    if not user_input or user_input == "-":
        suggested_questions, top_category, top_probability = suggest_questions(user_input=None, profile_data=profile_data, total_questions=total_questions)
    else:
        suggested_questions, top_category, top_probability = suggest_questions(user_input=user_input, profile_data=profile_data, total_questions=total_questions)

    response = {
        "suggested_questions": suggested_questions,
        "top_category": top_category,
        "probability": float(top_probability) if top_probability is not None else None
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
