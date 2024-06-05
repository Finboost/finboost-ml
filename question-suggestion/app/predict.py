import requests

# URL of the Flask application (use HTTP)
url = "http://127.0.0.1:8081/predict"

# Sample question and context
data = {
    "user_input": "Bagaimana cara memilih obligasi yang baik"
}

try:
    # Send POST request to the Flask app
    response = requests.post(url, json=data)
    
    # Print the response from the server
    print("Response from server:")
    response.raise_for_status()  # Raise an error for bad status codes
    print(response.json())
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")
