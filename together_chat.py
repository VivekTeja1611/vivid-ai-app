# together_chat.py

import os
import requests
from dotenv import load_dotenv

load_dotenv()

class TogetherAIChat:
    def __init__(self, api_key=None, model="mistralai/Mistral-7B-Instruct-v0.2"):
        self.api_key ="95a3d9c254c3bbff4ef32a8973d31516ef0e9b0b1f3e3940b3185c2adff0c913"
        self.model = model
        self.url = "https://api.together.xyz/v1/chat/completions"

    def chat(self, messages):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7
        }

        response = requests.post(self.url, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            print("[ERROR]", response.status_code, response.text)
            return "[ERROR] Failed to get response from Together AI."
