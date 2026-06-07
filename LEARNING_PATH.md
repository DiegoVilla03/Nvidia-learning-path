# Learning Path: NVIDIA LLMs, RAG & Embeddings

Path de aprendizaje progresivo. Cada módulo tiene un objetivo claro, los conceptos que cubre, y un ejercicio concreto que termina en código en el repo.

---

## Módulo 00 — Setup

**Duración estimada:** 1-2 días  
**Carpeta:** `00_setup/`

### Objetivo
Tener el entorno funcionando y hacer la primera llamada exitosa a la API de NVIDIA.

### Pasos
1. Crear cuenta en [build.nvidia.com](https://build.nvidia.com)
2. Generar una API key desde el dashboard
3. Crear entorno virtual y instalar dependencias base
4. Verificar conectividad con un `hello world` hacia NIM

### Conceptos
- Qué es NIM (NVIDIA Inference Microservices) y por qué existe
- Diferencia entre consumir NIM via API remota vs desplegarlo local con GPU
- Cómo está estructurado el catálogo de modelos en `build.nvidia.com`

### Entregable del módulo
`00_setup/setup_check.py` — script que imprime la lista de modelos disponibles y hace un ping de chat básico.

---

## Módulo 01 — NIM API: LLM Basics

**Duración estimada:** 1 semana  
**Carpeta:** `01_nim_api_basics/`

### Objetivo
Entender la API de NIM a fondo antes de introducir RAG. La API es compatible con OpenAI, pero hay particularidades de NVIDIA que vale conocer.

### Conceptos
- Estructura de una request: `messages`, `model`, `temperature`, `top_p`, `max_tokens`
- Streaming de respuestas con `stream=True`
- Modelos disponibles: Llama 3.1, Mistral, Phi-3, Gemma — cuándo usar cada uno
- Manejo de contexto y límites de tokens por modelo
- System prompts y control de comportamiento básico

### Ejercicios progresivos

**01a** — Chat simple: pregunta y respuesta, sin estado.

**01b** — Conversación con historial: mantener `messages` como lista y acumular turnos.

**01c** — Streaming: imprimir tokens conforme llegan, medir tiempo de primera respuesta vs tiempo total.

**01d** — Comparativa de modelos: mismo prompt enviado a Llama 3.1 8B, 70B y Mistral 7B. Comparar calidad, latencia y tokens usados.

**01e** — Structured output: forzar respuesta en JSON con instrucciones en el system prompt. Parsear y validar el resultado.

### Entregable del módulo
Un notebook `01_nim_api_basics/exploration.ipynb` con los cinco ejercicios documentados y un `README.md` del módulo con observaciones propias sobre diferencias entre modelos.

---

## Módulo 02 — Embeddings con NV-Embed-v2

**Duración estimada:** 1 semana  
**Carpeta:** `02_embeddings/`

### Objetivo
Entender qué son los embeddings a nivel práctico y dominar NV-Embed-v2 como modelo de embeddings.

### Conceptos
- Qué representa un embedding: distancia semántica, espacio vectorial
- Por qué NV-Embed-v2 y no OpenAI `text-embedding-ada-002`: benchmarks MTEB, dimensiones, contexto máximo
- Diferencia entre embeddings para documentos (`passage`) vs para queries (`query`) — NV-Embed-v2 usa input types distintos
- Normalización de vectores y similitud coseno vs producto punto
- Batching: cómo enviar múltiples textos en una sola llamada

### Ejercicios progresivos

**02a** — Generar un embedding simple, inspeccionar dimensiones y valores.

**02b** — Calcular similitud coseno manualmente entre tres frases. Verificar que frases semánticamente cercanas tienen score alto.

**02c** — Corpus pequeño (20-30 fragmentos de texto propio — usar fragmentos de cualquier documento personal). Generar embeddings para todos, guardar como numpy array.

**02d** — Búsqueda semántica manual: dada una query, rankear el corpus por similitud. Sin vector store todavía, solo numpy.

**02e** — Visualización: reducir a 2D con UMAP o t-SNE, plotear con matplotlib. Ver si los clusters tienen sentido semántico.

### Entregable del módulo
`02_embeddings/semantic_search.py` — clase `SemanticSearcher` que acepta un corpus, lo embeds, y expone un método `search(query, top_k)`. Documentado con docstrings.

---

## Módulo 03 — Vector Stores

**Duración estimada:** 1 semana  
**Carpeta:** `03_vector_stores/`

### Objetivo
Entender por qué numpy no escala y aprender a usar vector stores reales. Progresión deliberada: FAISS → Milvus Lite → cuVS.

### Conceptos
- El problema de búsqueda exhaustiva a escala: O(n) vs ANN (Approximate Nearest Neighbor)
- Algoritmos de indexado: IVF, HNSW — intuición sin matemática profunda
- FAISS: el baseline estándar de Meta, corre en CPU
- Milvus Lite: Milvus embebido sin servidor, default de los blueprints de NVIDIA
- cuVS: la librería de NVIDIA, más rápida que FAISS con GPU pero también usable conceptualmente
- Metadata filtering: buscar vectores con restricciones adicionales (fecha, categoría, fuente)

### Ejercicios progresivos

**03a** — Indexar el corpus del módulo anterior en FAISS. Comparar velocidad de búsqueda vs numpy.

**03b** — Migrar a Milvus Lite: mismas operaciones, API diferente. Familiarizarse con colecciones, schemas, y particiones.

**03c** — Metadata filtering en Milvus: agregar campos de metadata al corpus (ej: fuente, fecha, categoría) y hacer búsquedas filtradas.

**03d** — Persistencia: guardar y cargar el índice. Simular que el proceso muere y reinicia.

### Entregable del módulo
`03_vector_stores/vector_store_benchmark.py` — script que indexa el mismo corpus en FAISS y Milvus Lite, corre 100 búsquedas, y reporta latencia promedio y resultados. Tabla comparativa en el README del módulo.

---

## Módulo 04 — RAG Básico End-to-End

**Duración estimada:** 1-2 semanas  
**Carpeta:** `04_rag_basic/`

### Objetivo
Construir un pipeline RAG funcional desde cero, sin frameworks. Entender cada paso antes de que LangChain lo abstraiga.

### Arquitectura del módulo
```
PDF/TXT → chunking → NV-Embed-v2 → Milvus Lite → [query] → retrieve → NIM LLM → respuesta
```

### Conceptos
- Chunking: tamaño de chunk, overlap, estrategias (fixed, sentence, recursive)
- Por qué el chunking afecta drásticamente la calidad del RAG
- El prompt de RAG: cómo insertar contexto recuperado en el system prompt
- Problema de contextos irrelevantes: cuando el retrieval trae basura
- Métricas básicas de evaluación: relevancia del contexto recuperado, faithfulness de la respuesta

### Ejercicios progresivos

**04a** — Ingesta: tomar un PDF o TXT, chunkearlo con diferentes estrategias, comparar visualmente los chunks.

**04b** — Pipeline completo manual en ~100 líneas de Python. Sin LangChain, sin abstracciones. Cada paso explícito.

**04c** — Introducir LangChain + `langchain-nvidia-ai-endpoints`. Reescribir el pipeline usando `NVIDIAEmbeddings` y `ChatNVIDIA`. Comparar el código resultante.

**04d** — Evaluación cualitativa: 10 preguntas sobre el documento, inspeccionar qué chunks recuperó y si la respuesta es correcta.

### Entregable del módulo
`04_rag_basic/rag_pipeline.py` — clase `RAGPipeline` con métodos `ingest(path)` y `query(question) -> str`. Con una demo en `demo.py` que corre sobre un documento de muestra incluido en el repo.

---

## Módulo 05 — RAG Avanzado

**Duración estimada:** 2 semanas  
**Carpeta:** `05_rag_advanced/`

### Objetivo
Resolver los problemas reales del RAG básico: retrieval impreciso, respuestas alucinadas, documentos múltiples, preguntas complejas.

### Conceptos
- **Reranking**: por qué el retrieval por similitud no es suficiente. El modelo `nv-rerankqa-mistral-4b-v3` de NVIDIA como segundo filtro.
- **Hybrid search**: combinar búsqueda densa (embeddings) con búsqueda sparse (BM25). Milvus soporta ambas.
- **Query expansion**: reformular la query antes de buscar para mejorar recall.
- **Multi-document RAG**: manejar múltiples fuentes con metadata, citar fuentes en la respuesta.
- **Contextual compression**: reducir los chunks recuperados a solo la parte relevante antes de pasarlos al LLM.
- **Evaluación con RAGAS**: framework de evaluación automática de pipelines RAG.

### Ejercicios progresivos

**05a** — Agregar reranker: después del retrieval, pasar los top-20 por el reranker y quedarse con los top-3. Comparar calidad vs sin reranker.

**05b** — Hybrid search en Milvus: agregar índice BM25 al mismo corpus, combinar scores con weighted fusion.

**05c** — Multi-documento: ingestar 3-5 documentos distintos, agregar campo `source` a metadata, citar fuentes en respuesta.

**05d** — Evaluación con RAGAS: correr métricas de `faithfulness`, `answer_relevancy`, `context_precision` sobre un test set de 20 preguntas.

**05e** — Optimización de chunking guiada por métricas: variar tamaño de chunk y overlap, medir impacto en RAGAS scores.

### Entregable del módulo
`05_rag_advanced/advanced_pipeline.py` — pipeline extendido con reranking y hybrid search. Notebook `evaluation.ipynb` con resultados de RAGAS sobre el corpus de prueba. Reporte en README comparando métricas del módulo 04 vs módulo 05.

---

## Módulo 06 — NeMo Guardrails

**Duración estimada:** 1-2 semanas  
**Carpeta:** `06_guardrails/`

### Objetivo
Agregar control sobre el comportamiento del LLM en producción. Relevante para contextos empresariales como T-Systems.

### Conceptos
- Qué problema resuelven los Guardrails: jailbreaks, off-topic, respuestas peligrosas, flujos forzados
- Arquitectura de NeMo Guardrails: rails de input, output y diálogo
- Colang: el lenguaje de configuración de Guardrails
- Integración con pipelines RAG existentes

### Ejercicios progresivos

**06a** — Setup básico: instalar `nemoguardrails`, configurar un rail simple que rechace preguntas fuera de topic.

**06b** — Rail de input: detectar y bloquear prompts de jailbreak comunes.

**06c** — Rail de output: validar que la respuesta cite fuentes cuando el sistema lo requiere.

**06d** — Integrar Guardrails al pipeline del módulo 05.

### Entregable del módulo
`06_guardrails/` con la configuración Colang documentada y un notebook de demo mostrando el comportamiento con y sin rails.

---

## Módulo 07 — Pipeline de Producción

**Duración estimada:** 2-3 semanas  
**Carpeta:** `07_production_pipeline/`

### Objetivo
Empaquetar todo como un servicio real: API REST, manejo de errores, logging, configuración por entorno.

### Conceptos
- FastAPI como servidor del pipeline RAG
- Manejo de API keys con variables de entorno, nunca hardcodeadas
- Rate limiting y manejo de errores de la API de NVIDIA
- Caching de embeddings: no re-embedear documentos que no cambiaron
- Logging estructurado para debuggear el pipeline en producción
- Docker básico para empaquetar el servicio

### Ejercicios progresivos

**07a** — Envolver el pipeline del módulo 05 en una API FastAPI con endpoint `/ingest` y `/query`.

**07b** — Agregar caching de embeddings con hash de contenido.

**07c** — Logging estructurado: registrar query, chunks recuperados, modelo usado, latencia.

**07d** — Dockerfile y `docker-compose.yml` con Milvus como servicio separado.

### Entregable del módulo
Un servicio deployable con `docker-compose up`. README con instrucciones claras de despliegue. Este es el proyecto final del repo.

---

## Notas de estudio

Ver carpeta `notes/` para conceptos transversales: glosario, comparativas de modelos, recursos externos, papers relevantes.

---

## Timeline sugerido

| Semana | Módulo |
|---|---|
| 1 | 00 + 01a-01c |
| 2 | 01d-01e + 02a-02b |
| 3 | 02c-02e |
| 4 | 03 completo |
| 5-6 | 04 completo |
| 7-8 | 05a-05c |
| 9 | 05d-05e |
| 10 | 06 completo |
| 11-13 | 07 completo |

