# 🛍️ Product Query Bot via RAG Pipeline with LangGraph and LLama

This project is a LangGraph-powered microservice that receives user questions about products, retrieves relevant context from local documents, and returns grounded answers.

---

## 🚀 Features

- ✅ **LLM-based multi-agent flow** using LangGraph (Retriever + Responder)
- ✅ **RAG pipeline** with persistent Chroma vector DB (HuggingFace embeddings)
- ✅ **FastAPI endpoint** for incoming queries
- ✅ **Webhook callback** with configurable `CALLBACK_URL`
- ✅ **Memory tracking** using `thread_id`
- ✅ **HTML form** to simulate user input from a web UI
- ✅ **Cited sources** and similarity scores (cosine distance) per answer

---

## 🗂️ Project Structure

```
PRODUCT_QUERY_BOT/
├── agents/             # LangChain tools: retrieve, llm
├── vectorstore/        # Setup for Chroma DB (build + load)
├── tests/              # Unit tests (pytest)
├── docs/               # Dummy product description text files (~1000 words each)
├── db/                 # Persistent vector store
├── main.py             # FastAPI app (entry point)
├── rag_pipeline.py     # LangGraph pipeline logic
├── config.py           # Environment/config loader
├── .env                # Configuration (CALLBACK_URL, TOP_K, etc.)
├── requirements.txt
```

---

## ⚙️ Setup Instructions

### 1. ✅ Create virtual environment

```bash
python -m venv env
.\env\Scripts\Activate.ps1       # (use activate.bat on CMD)
```

### 2. ✅ Install dependencies

```bash
pip install -r requirements.txt
```

### 3. ✅ Build the vector store

```bash
python vectorstore/build.py
```

This loads `.txt` files from the `docs/` folder, splits them into chunks, and stores embeddings in Chroma.

---

## ▶️ Run the App

```bash
uvicorn main:app --reload
```

- Visit: [http://127.0.0.1:8000](http://127.0.0.1:8000) for the web form
- POST to: `http://127.0.0.1:8000/query` with:

```json
{
  "user_id": "user123",
  "query": "What are the features of the LensPro 4K Webcam?"
}
```

---

## 📩 Webhook Callback

Set `CALLBACK_URL=http://localhost:9000/callback` and TOP_K = 3 in your `.env` file.

If no external service is available, a test `/callback` endpoint is included locally.

---

## 🧪 Testing

Run unit tests:

```bash
pytest tests/test_retrieve_basic.py
```

This verifies:
- Retrieval tool returns relevant documents
- Artifact structure is valid

---

## 🧠 Memory Support

Each user session is tracked with `thread_id`, enabling:
- Multi-turn conversations
- Context-aware follow-ups

---

## ⏱️ Time Taken

| Task                       | Duration |
|---------------------------|----------|
| Project Setup             | ~20 mins |
| LangGraph + LLM wiring    | ~30 mins |
| VectorDB + Embeddings     | ~25 mins |
| FastAPI + Webhook         | ~20 mins |
| Testing + Debugging       | ~25 mins |
| Web UI + Memory Tracking  | ~30 mins |
| Total                     | **~2.5 hours** |

---

## 🐳 (Optional) Docker

> Not implemented due to time constraints. Easily added using `Dockerfile` + `CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]`.

---

## 🧠 Example Questions

- "What are the features of the LensPro 4K Webcam?"
- "Would AeroClean be a better fit for working from home?"
- "Which product has the longest battery life?"

---
