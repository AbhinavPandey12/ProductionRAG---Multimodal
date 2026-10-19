# Financial Multimodal RAG with Amazon S3 Vectors & Gemini

This repository implements an end-to-end Multimodal Retrieval-Augmented Generation (RAG) system for financial reports and PDFs, utilizing **Amazon S3 Vectors** as the vector database, **Google Gemini Embedding 2** (`gemini-embedding-2`) for multimodal embeddings, and **Gemini Pro** for multimodal vision-based generation.

---

## 🏗️ Architecture

```
[Financial PDFs (local / S3)]
            │
            ▼
 [Document Processor] ── Page images + Text snippets + Metadata
            │
            ▼
[Gemini Embedding 2] ── Multimodal embeddings
            │
            ▼
[Amazon S3 Vectors] ── Vector Bucket & Index (Cosine, Float32, 2KB filterable metadata)
            │
            ▼
[FastAPI / CLI Query] ── Tandem Vector Similarity + Metadata Filtering
            │
            ▼
 [Gemini Vision LLM] ── Rich, citation-backed answers with page context
```

---

## 🚀 Key Features

1. **Amazon S3 Vectors Integration**:
   - Uses native `s3vectors` client in `boto3`.
   - Automated creation and management of Vector Buckets (`create_vector_bucket`) and Vector Indexes (`create_index`).
   - Standard `float32` vector data type with `cosine` similarity metric.

2. **Native Metadata Filtering**:
   - Vector similarity search and metadata filter evaluation execute **in tandem** (pre-filtering), ensuring top-\(K\) results strictly satisfy filter conditions.
   - All metadata fields are **filterable by default**:
     - `document_name`: Filter by filename (e.g. `EY_Financial_report_2025.pdf`).
     - `page_number`: Filter by exact page number or page ranges.
     - `total_pages`: Filter by total pages.
     - `source_uri`, `source_bucket`, `source_key`: Document lineage and provenance.
     - `file_size`, `ingested_at`, `text_snippet`: Document details and extracted text.

3. **Gemini Embedding 2**:
   - High-dimensional multimodal embeddings (`gemini-embedding-2`) capturing layout, text, tables, and financial charts.

4. **Dual Ingestion Pipelines**:
   - **Local Ingestion (`src/embedding_service/main.py`)**: Directly embeds all PDFs in `data/` and uploads to S3 Vectors.
   - **Cloud Event Ingestion (`src/embedding_service/s3_worker.py`)**: SQS-driven daemon that automatically triggers ingestion upon S3 PDF uploads (`ObjectCreated:*`).

5. **FastAPI Generation Service & CLI**:
   - REST API (`src/generation_service/app.py`) with `/query`, `/upload`, and `/health` endpoints.
   - CLI Query tool (`src/generation_service/query.py`) supporting `--filter-doc` and `--filter-page` flags.

---

## ⚙️ Environment Variables

Create a `.env` file in the root directory:

```env
# AWS Credentials and Region
AWS_REGION=eu-north-1
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key

# Amazon S3 Vectors Configuration
S3_VECTOR_BUCKET_NAME=financial-rag-vectors
S3_VECTOR_INDEX_NAME=financial-index

# S3 Document Storage & SQS Ingestion (for Worker)
S3_BUCKET_NAME=financial-rag-documents-001
SQS_QUEUE_URL=https://sqs.eu-north-1.amazonaws.com/123456789012/financial-ingestion-queue

# Google Gemini API Key
GEMINI_API_KEY=your_gemini_api_key
```

---

## 📦 Installation

```bash
pip install -r src/embedding_service/requirements.txt
```

---

## 🏃 How to Run

### 1. Ingest Local Documents in `data/`
```bash
python3 src/embedding_service/main.py
```
This will:
1. Scan all PDFs in the `data/` folder (`EY_Financial_report_2025.pdf`, `JPM_Annual_2023.pdf`, etc.).
2. Generate Gemini Embedding 2 vectors.
3. Attach rich filterable metadata and upload to Amazon S3 Vectors.
4. Open an interactive question-answering CLI.

### 2. Start the FastAPI Service & Web UI
```bash
uvicorn src.generation_service.app:app --host 0.0.0.0 --port 8080 --reload
```
Once started:
- **Interactive Web UI**: Open your browser at [**http://localhost:8080**](http://localhost:8080) (or `http://localhost:8080/ui/`).
  - **Tab 1 (Upload)**: Upload new PDF documents directly to your S3 bucket.
  - **Tab 2 (Ask Questions & RAG Insights)**:
    - Query the S3 Vectors index with optional metadata filters (document filter & page number).
    - **Tables**: Rendered in clean, responsive tabular format.
    - **Code Blocks**: Formatted with syntax highlighting and a one-click copy button.
    - **Metadata & Citations**: Inspect retrieved vector keys, document names, page numbers, cosine distance scores, and text previews.
    - **Visual Page Inspector**: View rendered context pages and click to enlarge.
- Alternatively, open [`ui/index.html`](ui/index.html) directly in any modern browser.

#### Querying with Metadata Filter via curl:
```bash
# Query specific document
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the net revenue for 2023?",
    "document_name": "JPM_Annual_2023.pdf",
    "limit": 3
  }'

# Query with custom filter expression
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Summarize the balance sheet",
    "filter": {"document_name": "EY_Financial_report_2025.pdf"},
    "limit": 2
  }'
```

### 3. Query via Command Line
```bash
# General query
python3 src/generation_service/query.py "What are the key financial highlights?"

# Query filtered by document name
python3 src/generation_service/query.py "What were the investment banking fees?" --filter-doc "JPM_Annual_2023.pdf"

# Query filtered by specific page
python3 src/generation_service/query.py "Explain the cash flows table" --filter-doc "EY_Financial_report_2025.pdf" --filter-page 14
```

### 4. Run S3 Event Worker Daemon (Cloud Ingestion)
```bash
python3 src/embedding_service/s3_worker.py
```
