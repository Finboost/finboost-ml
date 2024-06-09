from flask import Flask, request, jsonify
import pandas as pd
from model import gpt_model

app = Flask(__name__)

# Load the dataset (assuming it is necessary for context-based responses)
df = pd.read_csv('./data/generative-ai/final_dataset.csv')

def find_context_for_question(question, dataframe):
    max_matched_words = 0
    best_matched_context = None
    context_found = False

    for _, row in dataframe.iterrows():
        if question.strip().lower() in row['question'].strip().lower():
            best_matched_context = row['context']
            context_found = True
            break

    if not context_found:
        for _, row in dataframe.iterrows():
            dataset_tokens = set(row['question'].strip().lower().split())
            matched_words = len(set(question.strip().lower().split()).intersection(dataset_tokens))
            if matched_words > max_matched_words:
                max_matched_words = matched_words
                best_matched_context = row['context']

    return best_matched_context, context_found

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    question = data.get('question', '')
    
    best_context, context_found = find_context_for_question(question, df)
    if not context_found:
        best_context = question  # Use the question as the context if no match is found
    
    answer = gpt_model.generate_response(best_context)
    
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
