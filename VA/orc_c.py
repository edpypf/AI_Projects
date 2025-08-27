import base64
import requests

def vision_extract(base64_image, prompt, api_key):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]}
        ],
        "max_tokens": 3000,
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()

with open("Image1.png", "rb") as f:
    base64_image = base64.b64encode(f.read()).decode('utf-8')

api_key = "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"
result = vision_extract(base64_image, "Extract text from the image.", api_key)
print(result["choices"][0]["message"]["content"])