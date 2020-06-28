import requests

response = requests.post('https://us-central1-mindful-279120.cloudfunctions.net/advanced-analysis', json={'text': ['Fuck you']})
print(response.json())