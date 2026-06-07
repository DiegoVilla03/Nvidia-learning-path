# nvidia-llm-path

Repositorio de aprendizaje progresivo del ecosistema NVIDIA para LLMs, RAG y Embeddings. Construido sin GPU local — todo corre sobre la API gratuita de NVIDIA NIM.

## Objetivo

Documentar el camino desde cero hasta implementaciones production-ready usando el stack nativo de NVIDIA: NIM, NV-Embed, cuVS, NeMo Guardrails y Milvus.

## Estructura

```
nvidia-llm-path/
├── 00_setup/
├── 01_nim_api_basics/
├── 02_embeddings/
├── 03_vector_stores/
├── 04_rag_basic/
├── 05_rag_advanced/
├── 06_guardrails/
├── 07_production_pipeline/
└── notes/
```

## Requisitos

- Python 3.10+
- Cuenta gratuita en [build.nvidia.com](https://build.nvidia.com) para obtener API key
- No se requiere GPU

## Stack

| Componente | Tecnología NVIDIA |
|---|---|
| Inferencia LLM | NIM API (Llama 3, Mistral) |
| Embeddings | NV-Embed-v2 |
| Reranking | nv-rerankqa-mistral-4b-v3 |
| Vector Store | Milvus Lite / cuVS |
| Orquestación | LangChain + langchain-nvidia-ai-endpoints |
| Control | NeMo Guardrails |

## Progreso

- [ ] 00 — Setup y primera llamada a NIM
- [ ] 01 — NIM API: chat, streaming, parámetros
- [ ] 02 — Embeddings con NV-Embed-v2
- [ ] 03 — Vector stores: FAISS → Milvus Lite → cuVS
- [ ] 04 — RAG básico end-to-end
- [ ] 05 — RAG avanzado con reranking y metadata filtering
- [ ] 06 — NeMo Guardrails
- [ ] 07 — Pipeline de producción

---

*Aprendizaje progresivo — cada módulo construye sobre el anterior.*
