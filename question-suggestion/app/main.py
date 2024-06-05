import numpy as np
import tensorflow as tf
import pandas as pd
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Load the dataset
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

# Define the model
model = tf.keras.models.load_model("./models/question-suggestion/model_question_suggestion.h5")

# Function to predict top questions
def suggest_questions(user_input, total_questions=4):
    sequence = tokenizer.texts_to_sequences([user_input])
    padded_sequence = pad_sequences(sequence, maxlen=padded_sequences.shape[1], padding='post')
    predictions = model.predict(padded_sequence)
    labels = list(label_dict.keys())
    
    # Get top category
    top_category_idx = np.argmax(predictions)
    top_category = labels[top_category_idx]
    
    # Print top category with its probability
    print("Top category and with the probability:")
    print(f'{top_category}: {predictions[0][top_category_idx]:.4f}')
    
    # Filter questions from the dataset based on top category
    suggested_questions_df = df[df['Category'] == top_category]
    
    # Add 'Probability' column to DataFrame
    suggested_questions_df['Probability'] = predictions[0][top_category_idx]
    
    # Sort by probability and get top 4 questions
    suggested_questions_df = suggested_questions_df.sort_values(by='Probability', ascending=False).head(total_questions)
    
    suggested_questions = suggested_questions_df['Question'].tolist()
    
    # Return top category, probability, and suggested questions
    return top_category, float(predictions[0][top_category_idx]), suggested_questions




# Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    user_input = data.get('user_input', '')
    
    top_category, probability, suggested_questions = suggest_questions(user_input)
    
    response = {
        "top_category": top_category,
        "probability": probability,
        "suggested_questions": suggested_questions
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)
