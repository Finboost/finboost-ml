from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
from . import financial_keywords  # Assuming financial_keywords.py contains a list of keywords

app = Flask(__name__)

# model_path = "saved_model"
model_path = "Bahasalab/Bahasa-4b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

THRESHOLD_LENGTH = 15
FINANCIAL_KEYWORDS = financial_keywords.KEYWORDS

def contains_financial_keywords(prompt):
    for keyword in FINANCIAL_KEYWORDS:
        if keyword.lower() in prompt.lower():
            return True
    return False

def trim_response_to_last_complete_sentence(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return ' '.join(sentences[:-1]) + (sentences[-1] if sentences[-1][-1] in '.!?' else '')

@app.route('/generate', methods=['POST'])
def predict():
    data = request.json
    prompt = data.get('prompt', '')
    
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400

    if not contains_financial_keywords(prompt):
        return jsonify({'response': 'Terima kasih atas pertanyaannya. Saya ingin menjelaskan bahwa peran saya di sini adalah untuk memberikan panduan dan rekomendasi seputar isu keuangan. Silakan ajukan pertanyaan terkait keuangan.'}), 200

    messages = [
        # {"role": "system", "content": "Kamu adalah asisten dari Finboost yang membantu seputar isu keuangan"},
        {"role": "system", "content": "Kamu adalah asisten dari Finboost yang membantu seputar isu keuangan. Kamu tidak akan menjawab pertanyaan lain selain yang terkait dengan keuangan."},
        {"role": "user", "content": prompt}
    ]

    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(device)
        generated_ids = model.generate(
            input_ids=inputs.input_ids, 
            attention_mask=inputs.attention_mask, 
            max_new_tokens=256,  # Increase token limit
            eos_token_id=tokenizer.eos_token_id,  # Ensure the model knows when to stop
            pad_token_id=tokenizer.pad_token_id
        )
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Ensure response ends with a complete sentence.
        assistant_response = response.split('assistant\n', 1)[-1].strip()
        assistant_response = trim_response_to_last_complete_sentence(assistant_response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # Determine if the prompt length exceeds the threshold
    is_expert = len(prompt.split()) > THRESHOLD_LENGTH

    if is_expert:
        assistant_response += "\n\nJika Anda memerlukan penjelasan lebih lanjut atau bantuan dari seorang ahli, Anda dapat menggunakan fitur konsultasi dengan pakar keuangan kami, silahkan cek list expert di menu kami."

    return jsonify({'response': assistant_response, 'isExpert': is_expert})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
