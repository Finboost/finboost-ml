import requests

def predict(prompt):
    url = 'http://localhost:8080/generate'
    payload = {
        'prompt': prompt
    }
    response = requests.post(url, json=payload)
    return response.json()

if __name__ == '__main__':
    prompt = "Apa itu ROI?"
    result = predict(prompt)
    print("Prompt:", prompt)
    print("Response:", result['response'])
