import requests

def get_suggestions(user_input, total_questions=4):
    url = 'http://localhost:8080/suggest'  # Ganti dengan URL aplikasi yang dideploy jika perlu
    headers = {'Content-Type': 'application/json'}
    data = {
        'user_input': user_input,
        'total_questions': total_questions
    }
    
    response = requests.post(url, json=data, headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        return result
    else:
        print(f"Error: {response.status_code}")
        return None

if __name__ == '__main__':
    user_input = "Bagaimana cara mengenali proyek cryptocurrency yang menjanjikan?"  # Contoh input pengguna
    suggestions = get_suggestions(user_input)

    if suggestions:
        print("\nTop categories with their probabilities:")
        print(f"{suggestions['top_category']}: {suggestions['probability']:.4f}")
        
        print("\nSuggested questions:")
        for question in suggestions['suggested_questions']:
            print(question)
