import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Blueprint, request, jsonify
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizerFast, TFBertForQuestionAnswering

main = Blueprint('main', __name__)

# Load data
df = pd.read_csv('../data/question-suggestion/data.csv')
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
questions = df['Question'].tolist()
labels = df['Category'].tolist()

# Prepare tokenizer and sequences
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(questions)
sequences = tokenizer.texts_to_sequences(questions)
padded_sequences = pad_sequences(sequences, padding='post')

# Label encoding
label_dict = {label: idx for idx, label in enumerate(set(labels))}
numerical_labels = [label_dict[label] for label in labels]

# Load model
model_name = "Rifky/Indobert-QA"
model = TFBertForQuestionAnswering.from_pretrained(model_name)
model.load_weights('../models/my_model_weights.h5')

# API Endpoint
@main.route('/suggest', methods=['POST'])
def suggest_questions():
    user_input = request.json['question']
    sequence = tokenizer.texts_to_sequences([user_input])
    padded_sequence = pad_sequences(sequence, maxlen=padded_sequences.shape[1], padding='post')
    predictions = model.predict(padded_sequence)
    labels = list(label_dict.keys())

    top_categories_prediction = np.argsort(predictions[0])[-1:][::-1]
    top_categories = [(labels[idx], predictions[0][idx]) for idx in top_categories_prediction]

    suggested_questions = []
    for category, _ in top_categories:
        category_questions = df[df['Category'] == category]['Question'].tolist()
        suggested_questions.extend(category_questions)
    
    return jsonify({'questions': np.random.choice(suggested_questions, 4).tolist()})
