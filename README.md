# Poe-RAG

Sistema de **Recuperación Aumentada por Generación (RAG)** basado en cuentos de Edgar Allan Poe.  
El proyecto combina:

- Ingesta y normalización de PDFs con **Docling**.
- Chunking y enriquecimiento semántico de los textos con **Gemini**.
- Indexación vectorial en **ChromaDB** usando **Ollama + embeddinggemma**.
- Interfaz de chat en **Gradio** que responde preguntas sobre los cuentos.

---

## Arquitectura general

1. `data/`
   - `silver/`  → documentos normalizados (JSONL por archivo).
   - `silver/chunked/` → chunks generados para RAG.
   - `gold/`   → chunks enriquecidos (resumen, keywords, entidades).

2. `src/`
   - `ingest/`
     - `ingest.py`      → carga PDFs, normaliza texto y metadata, genera capa *silver*.
     - `normalize.py`   → reglas de limpieza y normalización de metadata.
     - `spliter.py`     → divide documentos en chunks y genera `silver/chunked`.
     - `enrich.py`      → enriquece chunks con Gemini y genera capa *gold*.
   - `backend/`
     - `vectorstore.py` → construye/actualiza la colección de Chroma a partir de *gold*.
     - `retriever.py`   → define retrievers BM25 + vectorial y reranker con Ollama.
     - `generator.py`   → arma la cadena RAG (retriever + Gemini) y expone `generate_answer`.
   - `frontend/`
     - `gradio_app.py`  → interfaz de chat en Gradio.

3. `scripts/`
   - `ec2_chorma_db.sh`         → instalación y despliegue de ChromaDB en Docker.
   - `ec2_ollama_embeddings.sh` → instalación de Ollama y descarga de `embeddinggemma`.

---

## Requisitos

- Python 3.10+
- Docker (para ChromaDB)
- Ollama (para embeddings y, opcionalmente, reranking)
- Cuenta y API Key de Google para **Gemini**

---

## Instalación básica

Clonar el repositorio y crear entorno virtual:

```bash
git clone https://github.com/jpospinalo/poe_rag.git
cd poe_rag

python -m venv .venv
source .venv/bin/activate  # en Windows: .venv\Scripts\activate

pip install -r requirements.txt

## Pipeline básico de datos y ejecución

# 1) Ingesta y normalización (genera data/silver/*.jsonl)
python -m src.ingest.ingest

# 2) Chunking (genera data/silver/chunked/*.jsonl)
python -m src.spliter

# 3) Enriquecimiento con Gemini (genera data/gold/*.jsonl)
python -m src.ingest.enrich

# 4) Construcción/actualización de la colección en Chroma
python -m src.backend.vectorstore

# 5) Lanzar la interfaz Gradio
python -m src.frontend.gradio_app
```
La aplicación quedará disponible, por defecto, en:
```bash    
http://0.0.0.0:7860

