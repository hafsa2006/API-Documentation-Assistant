# 📘 API Documentation Assistant

> A RAG-powered conversational assistant that answers questions about your API documentation using local LLMs, semantic search, and ChromaDB.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58-red?logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-1.3-green)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

- 🔍 **Semantic Search** — ChromaDB vector store with `nomic-embed-text` embeddings
- 🤖 **Local LLM** — Powered by `llama3.1` via Ollama (no cloud API cost)
- ⚡ **Streaming Responses** — Real-time token-by-token answer generation
- 🧠 **Conversation Memory** — Maintains context across multiple questions
- 🛡️ **Input Guardrails** — Rejects off-topic questions automatically
- 📂 **Source Attribution** — Every answer cites which document it came from
- 🚀 **Cached Resources** — LLM, embeddings, and vectorstore are cached for fast responses
- 📊 **Performance Metrics** — Displays retrieval time, generation time, and total time

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Llama 3.1 via Ollama (local) |
| Embeddings | nomic-embed-text via Ollama |
| Vector Database | ChromaDB (persistent) |
| RAG Framework | LangChain |
| Tracing | LangSmith (optional) |
| Deployment | Docker |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com) installed and running

### 1. Install Ollama and pull models

```bash
# Install Ollama (Linux/Mac)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required models
ollama pull llama3.1
ollama pull nomic-embed-text
```

### 2. Clone the repository

```bash
git clone https://github.com/hafsa2006/API-Documentation-Assistant.git
cd API-Documentation-Assistant
```

### 3. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows

pip install -r requirements.txt
```

### 4. Set up environment variables (optional)

```bash
cp .env.example .env
# Edit .env and fill in your keys
```

### 5. Add your API documentation

Place your markdown (`.md`) files in the `api_docs/` folder.

### 6. Run the app

```bash
streamlit run streamlit_api_assistant.py
```

Visit **http://localhost:8501**

---

## 📁 Project Structure

```
API-Documentation-Assistant/
├── streamlit_api_assistant.py   # Main Streamlit app
├── api_docs/                    # API documentation source files
│   ├── api_guide.md
│   ├── authentication_guide.md
│   └── endpoints_reference.md
├── notebooks/
│   └── Handson_lab1.ipynb       # Lab 1: RAG pipeline exploration
├── Dockerfile                   # Container deployment
├── requirements.txt             # Pinned Python dependencies
├── .env.example                 # Environment variable template
├── .gitignore
└── README.md
```

---

## 🐳 Docker Deployment

```bash
# Build the image
docker build -t api-doc-assistant .

# Run the container
docker run -p 8501:8501 api-doc-assistant
```

---

## 🔒 Security

- API keys are never hardcoded — use `.env` file locally
- `.env` is included in `.gitignore` and will never be committed
- See `.env.example` for required environment variables

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).