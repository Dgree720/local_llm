import requests

url = "http://localhost:11434/api/generate"

resp = requests.post(
    url,
    json={
        "model": "dolphin-llama3",  # 👈 model slug here
        "prompt": "Write a haiku about local llms.",
        "stream": False,  # set True for streaming responses
    },
)

print(resp.json()["response"])
