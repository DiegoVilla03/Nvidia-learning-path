import os 
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = os.getenv("NVIDIA_API_KEY")
)

MODEL_NAME = "meta/llama-3.1-8b-instruct"

response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {
            "role": "system",
            "content": "Eres un asistente conciso. Responde en máximo dos oraciones."
        },
        {
            "role": "user",
            "content": "¿Qué es un transformer en el contexto de NLP?"
        }
    ],
    max_tokens=200,
    temperature=0.5
)

print(response.choices[0].message.content)