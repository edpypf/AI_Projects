from openai import OpenAI
import os
import json
from dotenv import load_dotenv

load_dotenv()
def get_ai_generated_data(prompt = f"Create 2 interview Q&A pairs for a software developer in JSON format. Output only JSON"):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("API key not found. Using placeholder data.")
        return [{"instruction": "What are your technical skills?", "response": "Python, data analysis"}]
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }], temperature=0.7, max_tokens=500
    )
    content = response.choices[0].message.content
    # If model returns Markdown-style JSON block
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1]
            
    if content:
        return content
    else:
        print("No content received from AI. Using placeholder data.")
        return [{"instruction": "What are your technical skills?", "response": "Python, data analysis"}]

# example call
result = get_ai_generated_data()
print(result)
