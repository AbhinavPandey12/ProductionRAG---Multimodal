# Multimodal RAG Studio

An end-to-end **Multimodal Retrieval-Augmented Generation (RAG)** system for textbooks, technical manuals, engineering notes, and financial reports. The system renders every PDF page as a high-resolution image, embeds it together with extracted text using **Google Gemini Multimodal Embeddings**, indexes it in **ChromaDB** (or optionally **[Amazon S3 Vectors] ── Vector Bucket & Index (Cosine, Float32, 2KB filterable metadata)**), and answers questions using **Gemini 2.0 Flash** vision with full citation support.

---

## 🏗️ Architecture

```
[PDF Documents in data/]
          │
          ▼
 [Document Processor]
  • Renders each page → PNG (150 DPI)
  • Extracts plain-text snippet
  • Attaches rich metadata
          │
          ▼
 [1. Gemini Embedding 2 Model]    ← gemini-embedding-exp-03-07
  • Multimodal: image + text → 3072-dim float32 vector
  • Captures text, tables, diagrams, formulas & visual layout
          │
          ▼
 [2. ChromaDB / Amazon S3 Vectors]
  • ChromaDB: local persistent vector index in .chroma_db/
  • [Amazon S3 Vectors] ── Vector Bucket & Index (Cosine, Float32, 2KB filterable metadata)
  • Filterable metadata: document_name, page_number, total_pages, file_size
          │
          ▼
 [FastAPI Service + Modern Web UI Studio]
  • /query  → vector similarity search + optional document/page filters
  • /upload → drag-and-drop browser PDF upload → automated ingestion
  • /health → real-time vector index and document stats
          │
          ▼
 [Generation LLM — choose any of the options]
  ├── Google Gemini 2.0 Flash  (multimodal vision + text, default)
  ├── 3. Groq LLM Key          (Llama 3.3 70B text generation, ultra-fast)
  └── 4. Amazon Bedrock Key    (Claude / Nova foundation models via AWS)
```

---

## 🔑 Supported Models & Components (Use Any of 4)

This project is built with modular abstraction layers. You can configure any of the following 4 options:

| # | Component | Provider / Option | Description & Role | Required? | Config / Keys |
|---|-----------|-------------------|--------------------|-----------|---------------|
| 1 | **Gemini Embedding 2 Model Key** | Google Gemini (`gemini-embedding-exp-03-07`) | Multimodal embeddings generating 3072-dimensional vectors from rendered page images and text snippets simultaneously. | ✅ **Required for embeddings** | `GEMINI_API_KEY` |
| 2 | **ChromaDB** *(or Amazon S3 Vectors)* | **ChromaDB** (local) & **[Amazon S3 Vectors] ── Vector Bucket & Index (Cosine, Float32, 2KB filterable metadata)** | Persistent vector storage with cosine distance metric and 2KB filterable metadata (`document_name`, `page_number`, `total_pages`). Zero-cloud local ChromaDB by default, or cloud-scale Amazon S3 Vectors via `S3VectorManager`. | ✅ **Built-in / Zero-config** | `CHROMA_COLLECTION` (default: `rag_documents`) |
| 3 | **Groq LLM Key** | Groq (`llama-3.3-70b-versatile`) | High-speed text-only generation model option. Available directly in the Web UI model dropdown and via CLI `--model groq`. | ⚡ **Optional alternative** | `GROQ_API_KEY` |
| 4 | **Amazon Bedrock Key** | Amazon Bedrock (Anthropic Claude 3 / Amazon Nova) | Enterprise foundation model inference hosted on AWS Bedrock for text and multimodal generation. | ⚡ **Optional alternative** | `AWS_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` |

> **Minimal Quickstart:** Only **`GEMINI_API_KEY`** is required for full multimodal capability! ChromaDB works locally out of the box without any external setup.

---

## ⚙️ Environment Variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```env
# ── 1. Gemini Embedding 2 Model Key (REQUIRED) ──────────────────────────────
# Obtain free at: https://aistudio.google.com/app/apikey
# Powers both multimodal embeddings and Gemini 2.0 Flash generation
GEMINI_API_KEY=your_gemini_api_key_here

# ── 2. Vector Store: ChromaDB or [Amazon S3 Vectors] ────────────────────────
# ChromaDB runs locally in .chroma_db/ without cloud setup.
# Can also target [Amazon S3 Vectors] ── Vector Bucket & Index (Cosine, Float32, 2KB filterable metadata).
CHROMA_COLLECTION=rag_documents

# ── 3. Groq LLM Key (OPTIONAL) ──────────────────────────────────────────────
# Obtain free at: https://console.groq.com/keys
# Powers ultra-fast text generation with Llama 3.3 70B
# GROQ_API_KEY=your_groq_api_key_here

# ── 4. Amazon Bedrock Key (OPTIONAL) ────────────────────────────────────────
# Requires AWS credentials with Bedrock access
# AWS_REGION=us-east-1
# AWS_ACCESS_KEY_ID=your_aws_access_key_id
# AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
# BEDROCK_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0
```

---

## 📦 Installation & Dependencies

Install dependencies using the root `requirements.txt`:

```bash
pip install -r requirements.txt
```

Core dependencies installed:
- `google-generativeai`: Gemini Embedding 2 (`gemini-embedding-exp-03-07`) and Gemini 2.0 Flash
- `chromadb`: Local persistent vector store with cosine indexing (supports migration to **[Amazon S3 Vectors] ── Vector Bucket & Index (Cosine, Float32, 2KB filterable metadata)**)
- `PyMuPDF` (`fitz`) & `Pillow`: PDF high-res page rendering (150 DPI) and image processing
- `fastapi` & `uvicorn[standard]`: Production async web API
- `groq`: Client for Groq Llama alternative generation
- `python-multipart` & `httpx`: File uploads and asynchronous request routing

---

## 🚀 Deployment Guide (Production Deployable)

This codebase is structured for zero-friction local or cloud deployment.

### Option A: Local / Virtual Machine Deployment

```bash
# 1. Ingest PDF documents in data/
python main.py --ingest

# 2. Run production server on port 8080
python main.py
# or with direct Uvicorn:
uvicorn src.generation_service.app:app --host 0.0.0.0 --port 8080 --workers 2
```

Access the Web UI at **`http://localhost:8080/ui/`**.

---

### Option B: Docker Container Deployment

A production-ready [Dockerfile](file:///Dockerfile) and [.dockerignore](file:///.dockerignore) are included.

```bash
# Build Docker image
docker build -t multimodal-rag-studio:latest .

# Run container with environment file
docker run -d \
  --name multimodal-rag \
  -p 8080:8080 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/.chroma_db:/app/.chroma_db \
  multimodal-rag-studio:latest
```

---

### Option C: Cloud Platform Deployment (Render / Railway / AWS ECS / Cloud Run)
1. Push repository to GitHub/GitLab.
2. Set Environment Variables in platform dashboard (`GEMINI_API_KEY`, optional `GROQ_API_KEY`).
3. Set Start Command:
   ```bash
   uvicorn src.generation_service.app:app --host 0.0.0.0 --port 8080
   ```
4. Health check URL: `http://<domain>/health`
5. Root URL `http://<domain>/` automatically redirects to `http://<domain>/ui/`.

---

## 🏃 Usage & Commands

### 1. Ingest Documents into Vector Store
Drop any PDF into the `data/` folder, then run:
```bash
python main.py --ingest
```
To force re-embedding of previously indexed documents:
```bash
python main.py --ingest --force
```

Every page is rendered to PNG, embedded via Gemini multimodal embedding, and indexed in ChromaDB (or [Amazon S3 Vectors] ── Vector Bucket & Index (Cosine, Float32, 2KB filterable metadata)) with page numbers, document name, and text snippets.

> **Pre-indexed sample documents:** `ELECTRONIC DEVICES AND CIRCUITS.pdf`, `PROPERTIES OF FLUIDS.pdf`

---

### 2. Launch the Web UI Studio
```bash
python main.py
```
Open **[http://localhost:8080/ui/](http://localhost:8080/ui/)** in your browser.

- **Tab 1: Ask Questions & RAG Insights**
  - Chat interface with Markdown, code highlighting, tables, and formula rendering
  - Filter by document or specific page number
  - Switch between Gemini 2.0 Flash or Groq Llama LLM
  - Image attachment / drag-and-drop for visual queries
  - Inspection drawer displaying retrieved context sources with cosine similarity scores
- **Tab 2: Document Management & Upload**
  - Drag-and-drop PDF upload with real-time progress
  - Automatic page extraction, multimodal embedding, and vector upsert
  - Live indexed document library with page counts and deletion options

---

### 3. CLI Query & Search
```bash
# General search with Gemini LLM
python main.py --query "Explain the PN junction diode characteristics"

# Use Groq Llama LLM
python src/generation_service/query.py "Explain viscosity" --model groq

# Filter query to a specific document
python src/generation_service/query.py "What is dynamic viscosity?" \
  --filter-doc "PROPERTIES OF FLUIDS.pdf"

# Filter to an exact page number
python src/generation_service/query.py "Explain the circuit diagram" \
  --filter-doc "ELECTRONIC DEVICES AND CIRCUITS.pdf" --filter-page 5

# Retrieve more context chunks
python src/generation_service/query.py "Summarize fluid properties" --limit 6
```

---

### 4. Vector Store Inspection & Maintenance
```bash
# Show summary statistics
python main.py --stats

# Inspect individual vectors and metadata in ChromaDB / Amazon S3 Vectors
python inspect_vectors.py

# Delete a specific document's vectors
python src/embedding_service/delete_vectors.py "PROPERTIES OF FLUIDS.pdf"

# Drop entire vector collection
python src/embedding_service/delete_vectors.py --all

# Start continuous directory watcher daemon (auto-ingests new PDFs placed in data/)
python src/embedding_service/s3_worker.py
```

---

## 🌐 REST API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Redirects to `/ui/` Studio |
| `/ui/` | GET | Modern dark-mode Web UI interface |
| `/health` | GET | Vector store health, vector count, indexed documents list, API key status |
| `/query` | POST | Multimodal question answering with context retrieval and optional image attachment |
| `/upload` | POST | Upload and automatically ingest a new PDF document into vector storage |
| `/documents` | GET | Returns array of all currently indexed document names |
| `/documents/{name}` | DELETE | Deletes all vector embeddings associated with a document |
| `/docs` | GET | Interactive Swagger UI API documentation and testing sandbox |

---

## 📁 Clean Repository Structure

```text
ProductionRaG - Multimodal/
├── Dockerfile                          ← Production container image configuration
├── .dockerignore                       ← Docker build exclusions
├── requirements.txt                    ← Root project dependencies for deployment
├── pyproject.toml                      ← Packaging and metadata specification
├── main.py                             ← Unified CLI entrypoint (server / ingest / query / stats)
├── inspect_vectors.py                  ← Vector inspection tool for ChromaDB / S3 Vectors
├── README.md                           ← Project documentation and deployment guide
├── .env.example                        ← Template with all 4 model & key configurations
├── .gitignore                          ← Excludes virtual environments, .chroma_db/, and caches
├── data/                               ← PDF storage and ingestion registry
│   ├── ELECTRONIC DEVICES AND CIRCUITS.pdf
│   └── PROPERTIES OF FLUIDS.pdf
├── ui/                                 ← Modern frontend interface
│   ├── index.html                      ← Semantic HTML5 layout
│   ├── style.css                       ← Glassmorphism & dark-mode styling
│   └── app.js                          ← Frontend API integration, chat & upload logic
└── src/
    ├── RAG_Agent_Notebook.ipynb        ← Experimental development notebook
    ├── embedding_service/
    │   ├── main.py                     ← Ingestion CLI & interactive terminal Q&A
    │   ├── document_processor.py       ← PDF rendering (150 DPI PNG) & text extraction
    │   ├── embedder.py                 ← Gemini Embedding 2 (multimodal 3072-dim vectors)
    │   ├── gemini_rag_llm.py           ← Gemini 2.0 Flash multimodal generation
    │   ├── groq_rag_llm.py             ← Groq Llama 3.3 70B generation
    │   ├── s3_vector_manager.py        ← ChromaDB & [Amazon S3 Vectors] vector management abstraction
    │   ├── s3_worker.py                ← File-watcher daemon for automated continuous ingestion
    │   ├── delete_vectors.py           ← Vector deletion CLI
    │   └── requirements.txt            ← Embedding service dependencies
    └── generation_service/
        ├── app.py                      ← Production FastAPI application
        └── query.py                    ← Standalone CLI query utility
```

---

## 🔄 Recent Changes & Conflict Resolutions

All redundant files and architectural conflicts have been resolved to produce a clean, deployable codebase:

| Action / File | Details & Resolution |
|---------------|----------------------|
| **ChromaDB & [Amazon S3 Vectors] integration** | Added dual support for local ChromaDB and `[Amazon S3 Vectors] ── Vector Bucket & Index (Cosine, Float32, 2KB filterable metadata)` through `S3VectorManager`. |
| **4 Key Integrations Documented** | Configured and documented: 1. Gemini Embedding 2, 2. ChromaDB / Amazon S3 Vectors, 3. Groq LLM Key, 4. Amazon Bedrock Key. |
| **Removed redundant legacy files** | Deleted deprecated, unused helper files: `src/generation_service/agent.py` (old LangGraph agent), `src/generation_service/bedrock_rag_llm.py`, `src/generation_service/gemini_rag_llm1.py`, and `src/generation_service/validation.py`. |
| **Resolved UI architecture** | Decoupled monolithic HTML into cleanly separated, deployable [index.html](file:///ui/index.html), [style.css](file:///ui/style.css), and [app.js](file:///ui/app.js). |
| **Added Deployment Artifacts** | Created root [requirements.txt](file:///requirements.txt), [Dockerfile](file:///Dockerfile), and [.dockerignore](file:///.dockerignore) for containerization. |
| **Fixed Windows stdout encoding** | Added safe UTF-8 stdout reconfiguration across all entry points to eliminate `UnicodeEncodeError` with emojis on Windows consoles. |
| **Cleaned `.gitignore`** | Added `.chroma_db/`, `.env`, temporary SQLite databases, and `__pycache__/` to ensure clean git tracking. |
| **Unified CLI Controller** | Standardized `main.py` with flags `--server`, `--ingest`, `--force`, `--query`, and `--stats`. |
| **Preserved Notebook** | Retained `src/RAG_Agent_Notebook.ipynb` intact as requested. |