from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

def chat(messages, user_input):
    messages.append({
        "role": "user",
        "content": user_input
    })
    
    response = client.chat.completions.create(
        model="meta/llama-3.1-8b-instruct",
        messages=messages,
        max_tokens=300,
        temperature=0.5
    )
    
    assistant_message = response.choices[0].message.content
    
    messages.append({
        "role": "assistant",
        "content": assistant_message
    })
    
    return assistant_message, messages


messages = [
    {
        "role": "system",
        "content": "Eres un asistente experto en ciencia de datos. Responde de forma concisa."
    }
]

# Turno 1
respuesta, messages = chat(messages, "¿Qué es un embedding?")
print(f"Turno 1:\n{respuesta}\n")

# Turno 2 — referencia algo del turno anterior
respuesta, messages = chat(messages, "¿Y cómo se calcula la similitud entre dos de ellos?")
print(f"Turno 2:\n{respuesta}\n")

# Turno 3
respuesta, messages = chat(messages, "Dame un ejemplo en Python de eso último")
print(f"Turno 3:\n{respuesta}\n")

# Ver cómo creció el historial
print(f"Mensajes totales en historial: {len(messages)}")