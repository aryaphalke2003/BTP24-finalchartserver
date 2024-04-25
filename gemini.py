import requests
import base64
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
API_KEY = os.getenv('GOOGLE_API_KEY')

# Image file path
image_path = 'PMC7465646___4.jpg'

# Load image data
with open(image_path, 'rb') as file:
    image_data = file.read()

# Encode image data to base64
encoded_image = base64.b64encode(image_data).decode('utf-8')

# Google Gemini API endpoint
url = 'https://vision.googleapis.com/v1/images:annotate?key=' + API_KEY

# Request payload
payload = {
  "requests": [
    {
      "image": {
        "content": encoded_image
      },
      "features": [
        {
          "type": "DOCUMENT_TEXT_DETECTION"
        }
      ]
    }
  ]
}

# Make POST request to the API
response = requests.post(url, json=payload)

# Check if request was successful
if response.status_code == 200:
    # Print the response
    print(response.json())
else:
    print("Error:", response.text)
