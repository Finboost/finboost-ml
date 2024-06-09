from flask import Blueprint, request, jsonify
from .model import generate_response

bp = Blueprint('main', __name__)

@bp.route('/generate', methods=['POST'])
def generate():
    data = request.json
    user_input = data.get('prompt', '')
    
    if not user_input:
        return jsonify({"error": "Prompt is required"}), 400
    
    response = generate_response(user_input)
    return jsonify({"response": response})
