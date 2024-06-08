import requests
import time
import hashlib

cache = {}

def get_cached_response(prompt):
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    return cache.get(prompt_hash)

def set_cached_response(prompt, response):
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    cache[prompt_hash] = response

def test_prompt(prompt):
    cached_response = get_cached_response(prompt)
    if cached_response:
        print("Response (from cache):", cached_response)
        return
    
    url = "https://YOUR_CLOUD_RUN_URL"
    payload = {"prompt": prompt}
    headers = {"Content-Type": "application/json"}
    
    start_time = time.time()
    response = requests.post(url, json=payload, headers=headers)
    end_time = time.time()
    
    if response.status_code == 200:
        response_text = response.json()["response"]
        set_cached_response(prompt, response_text)
        print("Response:", response_text)
        print("Time taken:", end_time - start_time, "seconds")
    else:
        print("Error:", response.json())

if __name__ == "__main__":
    prompt = "Bagaimana cara mengatur keuangan pribadi?"
    test_prompt(prompt)
    # Test with the same prompt to see the caching effect
    test_prompt(prompt)
