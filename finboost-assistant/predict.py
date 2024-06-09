import requests

def test_prompt(prompt):
    url = "http://localhost:5000/generate"
    payload = {"prompt": prompt}
    headers = {"Content-Type": "application/json"}
    
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        response_text = response.json()["response"]
        print("Response:", response_text)
    else:
        print("Error:", response.json())

if __name__ == "__main__":
    prompt = "Bagaimana cara mengatur keuangan pribadi?"
    test_prompt(prompt)
