# Agentic-AI-GitHub-Repository-Assistant
Agentic AI system for semantic GitHub repository understanding using FAISS, SentenceTransformers, and Gemini.
# CodeRAG — Agentic AI-Powered GitHub Repository Assistant

CodeRAG is an Agentic AI system that clones GitHub repositories, semantically indexes source code using SentenceTransformers and FAISS, and answers repository-related questions using Retrieval-Augmented Generation (RAG) with Gemini.

---

## Features

- Clone and analyze GitHub repositories
- Semantic code search using FAISS
- Retrieval-Augmented Generation (RAG)
- Agentic AI workflow with intelligent tool selection
- Local embedding generation using SentenceTransformers
- Graph-based repository analysis using NetworkX
- Streamlit-based interactive UI
- CLI support for terminal querying
- Modular plugin-based architecture
- Unit testing support.

---

## System Architecture

```text
GitHub Repository URL
        ↓
Repository Cloning
        ↓
Code Ingestion
        ↓
Embedding Generation
        ↓
FAISS Vector Indexing
        ↓
Semantic Retrieval
        ↓
Agentic Tool Selection
        ↓
Gemini Reasoning
        ↓
Repository-Aware Response
```

---

## Project Structure

```text
CODERAG/
│
├── coderag/
│   ├── graph/
│   │   └── graph_builder.py
│   │
│   ├── tools/
│   │   ├── general_reasoning.py
│   │   ├── local_search.py
│   │   └── read_file.py
│   │
│   ├── __init__.py
│   ├── agent.py
│   ├── cli.py
│   ├── config.py
│   ├── embeddings.py
│   ├── index.py
│   ├── planner.py
│   ├── search.py
│   └── tool_registry.py
│
├── tests/
│   ├── test_faiss.py
│   ├── test_index.py
│   └── test_search.py
│
├── .github/
├── .gitignore
├── .pre-commit-config.yaml
├── app.py
├── ingestor.py
├── repo_loader.py
├── requirements.txt
├── pyproject.toml
├── README.md
└── .env.example
```

---

## Technologies Used

| Component | Technology |
|-----------|-------------|
| Embeddings | SentenceTransformers |
| Vector Database | FAISS |
| LLM | Gemini 2.5 Flash |
| UI | Streamlit |
| Graph Analysis | NetworkX |
| Language | Python 3.11 |

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/vasudev1876961/Agentic-AI-GitHub-Repository-Assistant.git

cd Agentic-AI-GitHub-Repository-Assistant
```

---

### 2. Create Virtual Environment

#### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

#### Linux / Mac

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the root directory.

```env
WATCHED_DIR=./repos

FAISS_INDEX_FILE=./coderag_index.faiss

EMBEDDING_DIM=384

GEMINI_API_KEY=YOUR_GEMINI_API_KEY
GEMINI_CHAT_MODEL=gemini-2.5-flash

TOP_K=5
LOG_LEVEL=INFO
```

---

## Running the Project

Launch the Streamlit application:

```bash
streamlit run app.py
```

Application runs at:

```text
http://localhost:8501
```

---

## Usage Workflow

### Step 1 — Enter GitHub Repository URL

Example:

```text
https://github.com/karpathy/micrograd
```

---

### Step 2 — Repository Analysis

The system automatically:

- Clones the repository
- Reads source files
- Generates embeddings
- Stores vectors in FAISS

---

### Step 3 — Ask Repository Questions

Example queries:

```text
Explain the architecture of this repository

How does gradient computation work?

What does the Value class do?

Explain the indexing pipeline
```

---

## Agentic Workflow

```text
User Query
     ↓
Planner
     ↓
Tool Selection
     ↓
Semantic Retrieval
     ↓
LLM Reasoning
     ↓
Repository-Aware Answer
```

---

## Semantic Search Pipeline

```text
Code Files
     ↓
Chunk Processing
     ↓
SentenceTransformer Embeddings
     ↓
FAISS Vector Storage
     ↓
Similarity Search
     ↓
Relevant Context Retrieval
```

---

## Graph-Based Repository Analysis

Current graph capabilities include:

- File dependency extraction
- Function extraction
- Import relationship mapping
- Directed repository graphs

---

## Running Tests

```bash
pytest tests/
```

---

## CLI Usage

```bash
python -m coderag.cli "Explain the embedding generation pipeline"
```

---

## Current Capabilities

- Repository semantic understanding
- Vector-based code retrieval
- Agentic reasoning
- Context-aware repository QA
- Graph-based dependency analysis
- Plugin-based tool architecture

---

## Future Improvements

- AST-based semantic parsing
- Chunk-level indexing
- Cross-encoder reranking
- Multi-step agent planning
- True GraphRAG integration
- Call graph analysis
- Hybrid graph + vector retrieval

---

## Security Notes

- Never commit `.env`
- Never expose API keys
- Use `.gitignore`
- Rotate compromised API keys immediately

---

## Recommended .gitignore

```gitignore
.venv/
.env
__pycache__/
*.pyc
*.faiss
*.npy
repos/
coderag-egg-info/
.vscode/
```

---

## License

This project is intended for educational and research purposes under the MIT License.
