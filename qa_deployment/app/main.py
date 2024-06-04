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
# model.load_weights('./models/gen-ai/my_model_weights.h5')    # Load the model weights

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

def provide_recommendation():
    return "Terima kasih atas pertanyaannya. Saya ingin menjelaskan bahwa peran saya di sini adalah untuk memberikan panduan dan rekomendasi seputar isu keuangan"

def answer_question(question, context):
    if context is None:
        return provide_recommendation()  # Return recommendation if context not found

    inputs = tokenizer(question, context, return_tensors="tf", max_length=384, truncation=True, padding=True)

    outputs = model(inputs)

    # Get the start and end positions of the answer
    answer_start = tf.argmax(outputs.start_logits, axis=1).numpy()[0]
    answer_end = tf.argmax(outputs.end_logits, axis=1).numpy()[0]

    # Ensure the answer_end is greater than or equal to answer_start
    if answer_end < answer_start:
        answer_end = answer_start

    # Extract the answer tokens, extending to a reasonable length if necessary
    max_answer_length = 100
    answer_tokens = inputs["input_ids"][0][answer_start:answer_end + max_answer_length]

    # Convert tokens to text, skipping special tokens
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    # Strip out any trailing special tokens (like [SEP])
    answer = answer.split("[SEP]")[0]

    return answer

def find_context_for_question(question, dataframe):
    max_matched_words = 0
    best_matched_context = None
    context_found = False

    # Iterate through the rows of the dataframe
    for _, row in dataframe.iterrows():
        if question.strip().lower() in row['question'].strip().lower():
            best_matched_context = row['context']
            context_found = True
            break  # Break the loop if a matching context is found

    # If no exact match is found, find the best matching context based on the number of common words
    if not context_found:
        for _, row in dataframe.iterrows():
            dataset_tokens = set(row['question'].strip().lower().split())
            matched_words = len(set(question.strip().lower().split()).intersection(dataset_tokens))
            if matched_words > max_matched_words:
                max_matched_words = matched_words
                best_matched_context = row['context']

    return best_matched_context, context_found

def provide_recommendation_for_question(question):
    # Define the common words to ignore
    common_words = {
        "apa", "kenapa", "mengapa", "apa itu", "bagaimana", "siapa",
        "di mana", "kapan", "yang", "adalah", "untuk", "dengan",
        "ke", "dari", "atau", "dan", "jika", "jika", "maka",
        "seperti", "oleh", "agar", "sehingga", "karena", "namun",
        "jadi", "tidak", "adalah", "bahwa", "itu", "dalam",
        "oleh", "pada", "untuk", "dengan", "tanpa", "saat",
        "akan", "sudah", "belum", "pernah", "apakah", "apabila",
        "bagaimanakah", "sebagaimana", "adakah", "bilamana", "mengapakah",
        "kapankah", "dimanakah", "siapakah", "apa sajakah", "berapa"
    }

    # Function to filter out common words from a list of tokens
    def filter_common_words(tokens):
        return [word for word in tokens if word not in common_words]

    max_matched_words = 0
    best_matched_context = None

    # Tokenize and filter the question
    question_tokens = set(filter_common_words(question.strip().lower().split()))

    # Iterate through the rows of the dataset
    for _, row in df.iterrows():
        dataset_tokens = set(filter_common_words(row['question'].strip().lower().split()))
        matched_words = len(question_tokens.intersection(dataset_tokens))
        if matched_words > max_matched_words:
            max_matched_words = matched_words
            best_matched_context = row['context']

    if best_matched_context:
        return best_matched_context
    else:
        return provide_recommendation()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    question = data.get('question', '')
    best_context, context_found = find_context_for_question(question, df)
    if not context_found:
        answer = provide_recommendation_for_question(question)
    else:
        answer = answer_question(question, best_context)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
