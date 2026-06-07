# Referencia de Modelos en NIM

Guía rápida para elegir modelo según el caso de uso.

## LLMs de Chat

| Modelo | Parámetros | Contexto | Cuándo usar |
|---|---|---|---|
| meta/llama-3.1-8b-instruct | 8B | 128k | Desarrollo, iteración rápida, bajo costo |
| meta/llama-3.1-70b-instruct | 70B | 128k | Calidad alta, producción general |
| meta/llama-3.1-405b-instruct | 405B | 128k | Tareas complejas, máxima calidad |
| mistralai/mistral-7b-instruct-v0.3 | 7B | 32k | Alternativa a Llama 8B, bueno en código |
| microsoft/phi-3-mini-128k-instruct | 3.8B | 128k | Muy ligero, contexto largo |

**Recomendación para desarrollo:** Llama 3.1 8B. Rápido, barato, suficiente para iterar.  
**Recomendación para producción RAG:** Llama 3.1 70B.

## Embeddings

| Modelo | Dimensiones | Contexto máximo | Notas |
|---|---|---|---|
| nvidia/nv-embed-v2 | 4096 | 32k | Default recomendado, top MTEB |
| nvidia/embed-qa-4 | 1024 | 512 | Optimizado específicamente para Q&A |

**Nota importante:** NV-Embed-v2 requiere especificar `input_type`:
- `"passage"` para documentos a indexar
- `"query"` para queries de búsqueda

## Rerankers

| Modelo | Notas |
|---|---|
| nvidia/nv-rerankqa-mistral-4b-v3 | Default para RAG, acepta query + lista de passages |

## Cómo consultar modelos disponibles desde código

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="<NVIDIA_API_KEY>"
)

models = client.models.list()
for m in models.data:
    print(m.id)
```
