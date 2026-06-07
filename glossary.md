# Glosario

Términos del ecosistema NVIDIA y RAG que aparecen a lo largo del repo.

**NIM (NVIDIA Inference Microservices)** — Contenedores preoptimizados de NVIDIA que empaquetan un modelo + TensorRT-LLM + servidor de inferencia. API compatible con OpenAI. Se pueden consumir remotamente vía `api.nvidia.com` sin GPU.

**TensorRT-LLM** — Motor de inferencia de NVIDIA que optimiza modelos para hardware NVIDIA. Responsable de la velocidad de NIM. No se interactúa con él directamente cuando se usa la API remota.

**NV-Embed-v2** — Modelo de embeddings de NVIDIA. Soporte de 32k tokens de contexto, 4096 dimensiones. Usa `input_type` para distinguir entre embeddings de documentos (`passage`) y de queries (`query`).

**cuVS** — Librería de NVIDIA para búsqueda vectorial acelerada por GPU. Sucesor de RAFT. Benchmarks superiores a FAISS en hardware NVIDIA.

**Milvus** — Base de datos vectorial open source. NVIDIA la usa como default en sus blueprints de RAG. Milvus Lite corre embebido sin servidor, ideal para desarrollo.

**RAG (Retrieval-Augmented Generation)** — Patrón que combina búsqueda en una base de conocimiento con generación de texto. Permite que el LLM responda sobre documentos que no estaban en su entrenamiento.

**Reranker** — Modelo secundario que toma los resultados del retrieval y los reordena por relevancia real respecto a la query. Más costoso que embeddings pero más preciso.

**Hybrid Search** — Combinación de búsqueda densa (embeddings) y búsqueda sparse (BM25/keyword). Milvus soporta ambas y puede fusionar los scores.

**Chunking** — Proceso de dividir documentos en fragmentos antes de generar embeddings. El tamaño y estrategia de chunking es uno de los factores más impactantes en la calidad del RAG.

**RAGAS** — Framework de evaluación de pipelines RAG. Métricas principales: faithfulness (¿la respuesta está soportada por el contexto?), answer relevancy (¿responde la pregunta?), context precision (¿los chunks recuperados son relevantes?).

**NeMo Guardrails** — Framework de NVIDIA para controlar el comportamiento de LLMs en producción. Usa un lenguaje propio llamado Colang para definir rails de input, output y diálogo.

**Colang** — Lenguaje de configuración de NeMo Guardrails. Define flujos de conversación y restricciones de comportamiento.

**BM25** — Algoritmo clásico de recuperación de información basado en frecuencia de términos. Complementa a los embeddings en hybrid search, especialmente útil para términos técnicos o nombres propios.
