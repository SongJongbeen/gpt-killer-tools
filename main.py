import requests
import os
import json
from dotenv import load_dotenv
from GPTZero.model import GPT2PPL

load_dotenv()

class GPTKillerTools:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.api_key = None
        if model_name == 'detecting-ai':
            self.api_key = os.getenv("DETECTING_AI_KEY")
        elif model_name == 'zerogpt':
            self.api_key = os.getenv("ZEROGPT_API_KEY")
        elif model_name == 'gptzero':
            self.api_key = os.getenv("GPTZERO_API_KEY")
        elif model_name == 'local-gptzero':
            self.api_key = None

    def detect(self, text: str):
        if self.model_name == 'detecting-ai':
            return self.detect_ai(text)
        elif self.model_name == 'zerogpt':
            return self.detect_zerogpt(text)
        elif self.model_name == 'gptzero':
            return self.detect_gptzero(text)
        elif self.model_name == 'local-gptzero':
            return self.local_detect_gptzero(text)

    def detect_ai(self, text: str):
        endpoint = "https://api.detecting-ai.com/api/detect/"
        response = requests.post(
            endpoint,
            headers = {"X-API-Key": self.api_key},
            json={'text': text, 'version': 'v1' or 'v2'},
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()['data']
        else:
            result = None
            print(f"Error: {response.status_code}")

        return result

    def detect_zerogpt(self, text: str):
        endpoint = "https://api.zerogpt.com/api/detect/detectText"

        payload = json.dumps({
            "input_text": text
        })

        headers = {
            'ApiKey': self.api_key,
        }

        response = requests.request("POST", endpoint, headers=headers, data=payload, timeout=30)

        return response.text

    def detect_gptzero(self, text: str):
        endpoint = "https://api.gptzero.me/v2/predict/text"

        payload = {
            "document": text,
            "multilingual": False
        }

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        response = requests.post(endpoint, json=payload, headers=headers, timeout=30)

        return response.json()

    def local_detect_gptzero(self, text: str):
        model = GPT2PPL()
        sentence = text
        return model.getPPL(sentence)

# 아래는 예시입니다 :)
gkt = GPTKillerTools('local-gptzero')
text = "안녕 반가워"
result = gkt.detect(text)
print(result)
