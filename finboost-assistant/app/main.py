from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def create_app():
    app = Flask(__name__)

    # Load the model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "Bahasalab/Bahasa-4b-chat-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    @app.route('/generate', methods=['POST'])
    def generate_response():
        data = request.json
        prompt = data.get("prompt", "")

        if not any(finance_keyword in prompt.lower() for finance_keyword in ["keuangan", "finansial", "investasi", "saham", "bank", "ekonomi", "asuransi"]):
            return jsonify({"response": "Maaf, saya hanya bisa menjawab pertanyaan seputar keuangan."})

        messages = [
            {"role": "system", "content": "Kamu adalah asisten dari Finboost yang hanya membantu seputar masalah keuangan saja."},
            {"role": "user", "content": prompt}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        generated_ids = model.generate(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=512,
            eos_token_id=tokenizer.eos_token_id
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return jsonify({"response": response})

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host='0.0.0.0', port=8080)
