from flask import Blueprint, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

bp = Blueprint("main", __name__)

@bp.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get("prompt", "")
    
    # Load the model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "Bahasalab/Bahasa-4b-chat-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    # Your generation logic here
    response = "Generated response"
    
    return jsonify({"response": response})
