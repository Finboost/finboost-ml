from flask import Flask, request, jsonify
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizerFast, TFBertForQuestionAnswering
import requests

app = Flask(__name__)

# Load the tokenizer and model
model_name = "Rifky/Indobert-QA"
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = TFBertForQuestionAnswering.from_pretrained(model_name)
# model.load_weights('./models/gen-ai/my_model_weights.h5')  # Load the model weights

# URL of the model weights
weights_url = 'https://storage.googleapis.com/finboost-generative-ai-model/my_model_weights.h5'

# Download the model weights
local_weights_file = 'my_model_weights.h5'
response = requests.get(weights_url)

with open(local_weights_file, 'wb') as f:
    f.write(response.content)

# Load the model weights
model.load_weights(local_weights_file)

# Load the dataset
df = pd.read_csv('./data/final_dataset.csv')

def find_best_context(question):
    max_matched_words = 0
    best_matched_context = None

    for _, row in df.iterrows():
        dataset_tokens = set(row['question'].strip().lower().split())
        matched_words = len(set(question.strip().lower().split()).intersection(dataset_tokens))
        if matched_words > max_matched_words:
            max_matched_words = matched_words
            best_matched_context = row['context']
    
    return best_matched_context

def answer_question(question, context):
    inputs = tokenizer(question, context, return_tensors="tf", max_length=384, truncation=True, padding=True)
    outputs = model(inputs)
    answer_start = tf.argmax(outputs.start_logits, axis=1).numpy()[0]
    answer_end = tf.argmax(outputs.end_logits, axis=1).numpy()[0]
    if answer_end < answer_start:
        answer_end = answer_start
    max_answer_length = 100
    answer_tokens = inputs["input_ids"][0][answer_start:answer_end + max_answer_length]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).split("[SEP]")[0]
    return answer

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    question = data.get('question', '')
    best_context = find_best_context(question)
    if not best_context:
        return jsonify({'answer': 'Context not found for the given question.'})
    answer = answer_question(question, best_context)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
