import requests

def get_suggestions(user_input, total_questions=4, profile_data=None):
    url = 'http://localhost:8080/suggest'  # Replace with the deployed application URL if necessary
    headers = {'Content-Type': 'application/json'}
    data = {
        'user_input': user_input,
        'total_questions': total_questions
    }
    
    # Include profile data in the request payload if provided
    if profile_data:
        data.update(profile_data)
    
    response = requests.post(url, json=data, headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        return result
    else:
        print(f"Error: {response.status_code}")
        return None

if __name__ == '__main__':
    user_input = "-"  # Example user input
    profile_data = {
        "income": "0",
        "investment_type": "Reksadana",
        "savings": "0",
        "debt": "0",
        "insurance_type": "-"
    }

    suggestions = get_suggestions(user_input, profile_data=profile_data)

    if suggestions:
        if suggestions["top_category"]:
            print("\nTop category with its probability:")
            print(f"{suggestions['top_category']}: {suggestions['probability']:.4f}")
        
        print("\nSuggested questions:")
        for question in suggestions['suggested_questions']:
            print(question)
