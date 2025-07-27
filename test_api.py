import json
import requests

# 👇 Replace this with your live Railway URL
url = "https://titanic-api-production.up.railway.app/predict"

# Load input data from JSON file
with open("app/input.json") as f:
    data = json.load(f)

# Send POST request to the API
response = requests.post(url, json=data)

# Handle response
print(f"Status code: {response.status_code}")

try:
    print("Response:")
    print(response.json())
except json.JSONDecodeError:
    print("Response was not JSON. Raw response:")
    print(response.text)
