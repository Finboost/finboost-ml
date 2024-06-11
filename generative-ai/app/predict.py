import requests

# URL of the Flask application (use HTTP)
url = "http://127.0.0.1:8080/generate"

# Sample question and context
data = {
    "prompt": "apa itu finansial"
}

# Send POST request to the Flask app
response = requests.post(url, json=data)

# Print the response from the server
print("Response from server:")
print(response.json())
