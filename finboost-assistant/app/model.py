import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logging.basicConfig(filename='error.log', level=logging.ERROR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "Bahasalab/Bahasa-4b-chat-v2"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
except Exception as e:
    logging.error("Error loading model: %s", e)
    raise e

financial_keywords = ["uang", "keuangan", "finansial", "investasi", "tabungan", "hutang", "bank", "asuransi", "pajak", "kredit", "anggaran"]

def generate_response(user_input):
    system_message = {"role": "system", "content": "Kamu adalah asisten dari Finboost yang membantu seputar masalah keuangan."}
    user_message = {"role": "user", "content": user_input}
    
    if not any(keyword in user_input.lower() for keyword in financial_keywords):
        return "Saya hanya bisa membantu pertanyaan seputar masalah keuangan."
    
    messages = [system_message, user_message]
    
    try:
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
        return response
    except Exception as e:
        logging.error("Error generating response: %s", e)
        return "Terjadi kesalahan dalam memproses permintaan Anda."
