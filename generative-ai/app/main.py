# python -m app.app
from flask import Flask, request, jsonify, render_template
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
import tensorflow as tf
# from .financial_keywords import FINANCIAL_KEYWORDS as keywords
# from .banned_keywords import BANNED_WORDS as banneds
from . import financial_keywords as keywords
from . import banned_keywords as banneds
import pandas as pd


FINANCIAL_KEYWORDS = keywords.FINANCIAL_KEYWORDS
BANNED_WORDS = banneds.BANNED_WORDS


app = Flask(__name__)

# Load model and tokenizer
model_name = "./models/gen-ai/"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = TFGPT2LMHeadModel.from_pretrained(model_name)

df = pd.read_csv('./data/finansial-dataset-v2.csv')
prompt_response_dict = dict(zip(df['prompt'], df['response']))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get('prompt', '').lower()

    # Handle specific user inputs
    if prompt in ['hello', 'terima kasih', 'makasih', 'hi', 'p', 'oke', 'ok', 'okay', 'thanks', 'hai', 'mksh', 'hallo', 'pe', 'siapa kamu']:
        response = "Terima kasih telah menghubungi saya sebagai assisten finansial anda. Ada yang bisa saya bantu?"
        is_expert = False
    else:
        response, is_expert = generate_response(prompt)
    
    return jsonify({"response": response, "isExpert": is_expert})

def generate_response(prompt):
    # Check for banned words
    if any(banned_word in prompt.lower() for banned_word in BANNED_WORDS):
        return "Mohon maaf, Bahasa yang digunakan tidak pantas dan tidak diperbolehkan. Jika Anda memerlukan bantuan terkait masalah finansial, dengan senang hati saya siap membantu.", False

    if prompt in prompt_response_dict:
        return prompt_response_dict[prompt]

    is_finance_related = any(keyword in prompt.lower() for keyword in FINANCIAL_KEYWORDS)

    if not is_finance_related:
        return "Terima kasih atas pertanyaannya. Saya ingin menjelaskan bahwa peran saya di sini adalah untuk memberikan panduan dan rekomendasi seputar masalah finansial. Silakan ajukan pertanyaan terkait finansial anda", False

    # Ensure the prompt ends with appropriate punctuation
    if not prompt.endswith(('.', '?', '!', ':')):
        prompt += '.'

    inputs = tokenizer(prompt, return_tensors="tf", padding=True, truncation=True)
    prompt_output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=100,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(prompt_output[0], skip_special_tokens=True)

    # Ensure the response does not include the prompt
    response = response.replace(prompt, '').strip()

    # Ensure the response ends with a complete sentence
    if not response.endswith('.'):
        response = response.rsplit('.', 1)[0] + '.'

    THRESHOLD_LENGTH = 10
    is_expert = len(prompt.split()) > THRESHOLD_LENGTH
    if is_expert:
        response += "\n\nJika Anda memerlukan penjelasan lebih lanjut atau bantuan dari seorang ahli, Anda dapat menggunakan fitur konsultasi dengan pakar finansial kami, silahkan cek list expert di menu kami."

    return response, is_expert

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
