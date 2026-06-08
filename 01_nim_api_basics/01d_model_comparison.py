from openai import OpenAI
import os
from dotenv import load_dotenv
import time

load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

MODELS = [
    "meta/llama-3.1-8b-instruct",
    "meta/llama-3.1-70b-instruct",
    #"meta/llama-3.2-1b-instruct",
]

PROMPT = "Explica en 3 puntos concretos cómo funciona la atención en un transformer."

def query_model(model, prompt):
    t0 = time.time()
    t1 = None
    
    response_text = ""
    stream = client.chat.completions.create(
        model = model,
        messages = [
            {"role": "system", "content":"Responde de manera técnica y consisa"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.5,
        stream=True
    )
    
    for chunk in stream:
        if not chunk.choices:
            continue
        token =chunk.choices[0].delta.content
        if token:
            if t1 is None:
                t1 = time.time()
            response_text += token
        
    t2 = time.time()
    return {
        "model": model,
        "response": response_text,
        "total_time": round(t2 - t0, 2),
        "time_to_first_token": round(t1-t0,2),
        "tokens_generated": len(response_text.split())
    }
    
    
results = []
for model in MODELS:
    print(f"consultando modelo {model}...")
    result = query_model(model, PROMPT)
    results.append(result)


print("\n" + "="*60)
print(f"{'Modelo':<40} {'TTFT':>6} {'Total':>7} {'Tokens':>7}")
print("="*60)
for r in results:
    name = r["model"].split("/")[-1]
    print(f"{name:<40} {r['time_to_first_token']:>5}s {r['total_time']:>6}s {r['tokens_generated']:>7}")

print("\n--- Respuestas ---\n")
for r in results:
    print(f"[ {r['model']} ]")
    print(r["response"])
    print()