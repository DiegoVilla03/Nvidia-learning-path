from openai import OpenAI
import os
from dotenv import load_dotenv
import time

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
    
    t0 = time.time()
    
    stream = client.chat.completions.create(
        model="meta/llama-3.1-8b-instruct",
        messages=messages,
        max_tokens=300,
        temperature=0.5,
        stream=True
    )
    assistant_message = ""
    t1 = None
    

    for chunk in stream:
        if not chunk.choices:
            continue
        token =chunk.choices[0].delta.content
        if token:
            if t1 is None:
                t1 = time.time()
            assistant_message += token
            print(token, end="", flush=True)

    t2 = time.time()
    print("\n")
    print(f"Tiempo total: {t2 - t0:.2f} segundos")
    print(f"Tiempo hasta el primer token: {t1 - t0:.2f} segundos")
    print(f"tokens generados: {len(assistant_message.split())}")

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
# Turno 2 — referencia algo del turno anterior
respuesta, messages = chat(messages, "¿Y cómo se calcula la similitud entre dos de ellos?")
# Turno 3
respuesta, messages = chat(messages, "Dame un ejemplo en Python de eso último")


# Ver cómo creció el historial
#print(f"Mensajes totales en historial: {len(messages)}")