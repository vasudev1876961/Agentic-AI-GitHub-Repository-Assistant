import os
from dotenv import load_dotenv

#  Load environment variables from .env file
load_dotenv()

#  Core Project Settings
WATCHED_DIR = os.getenv("WATCHED_DIR", "./sample_code")
FAISS_INDEX_FILE = os.getenv("FAISS_INDEX_FILE", "./coderag_index.faiss")

# Directory for cloned GitHub repositories
REPOS_DIR = os.getenv("REPOS_DIR", "./repos")
os.makedirs(REPOS_DIR, exist_ok=True)

#  Allowed File Extensions for Indexing
ALLOWED_EXTENSIONS = set(
    os.getenv("ALLOWED_EXTENSIONS", ".py,.md,.txt,.json,.ipynb")
    .replace(" ", "")
    .lower()
    .split(",")
)

#  Embedding Configuration
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))

#  Gemini Configuration (google-genai ≥ v1.0)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Accept both old and new naming conventions
_default_gemini_models = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-pro"]


GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "").strip()
if not GEMINI_CHAT_MODEL:
    GEMINI_CHAT_MODEL = "gemini-2.5-flash" 
elif GEMINI_CHAT_MODEL.startswith("models/"):
    GEMINI_CHAT_MODEL = GEMINI_CHAT_MODEL.replace("models/", "", 1)
elif GEMINI_CHAT_MODEL not in _default_gemini_models:
    GEMINI_CHAT_MODEL = "gemini-2.5-flash"

# Embedding model (still uses text-embedding-004)
GEMINI_EMBEDDING_MODEL = os.getenv(
    "GEMINI_EMBEDDING_MODEL", "text-embedding-004"
).replace("models/", "")

#  OpenAI (Optional Fallback)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

#  Monitoring Configuration
IGNORE_PATHS = [
    "__pycache__",
    ".git",
    ".venv",
    "node_modules",
    ".idea",
    ".vscode",
    "env",
    "venv",
    "__init__.py",
]

#  Misc Settings
TOP_K = int(os.getenv("TOP_K", "5"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

#  Utility Function
def print_config_summary():
    """Print a summary of the current configuration (for debugging)."""
    print("\n CodeRAG Configuration Summary")
    print("=" * 50)
    print(f" WATCHED_DIR            : {WATCHED_DIR}")
    print(f" REPOS_DIR              : {REPOS_DIR}")
    print(f" FAISS_INDEX_FILE       : {FAISS_INDEX_FILE}")
    print(f" EMBEDDING_DIM          : {EMBEDDING_DIM}")
    print(f" ALLOWED_EXTENSIONS     : {ALLOWED_EXTENSIONS}")
    print(f" IGNORE_PATHS           : {IGNORE_PATHS}")
    print("-" * 50)
    print(f" GEMINI_CHAT_MODEL      : {GEMINI_CHAT_MODEL}")
    print(f" GEMINI_EMBEDDING_MODEL : {GEMINI_EMBEDDING_MODEL}")
    print(f" GEMINI_API_KEY         : {' Set' if GEMINI_API_KEY else ' Missing'}")
    print("-" * 50)
    print(f" OPENAI_CHAT_MODEL      : {OPENAI_CHAT_MODEL}")
    print(f" OPENAI_EMBEDDING_MODEL : {OPENAI_EMBEDDING_MODEL}")
    print(f" OPENAI_API_KEY         : {' Set' if OPENAI_API_KEY else ' Missing'}")
    print("=" * 50)
    print(f" TOP_K                  : {TOP_K}")
    print(f" LOG_LEVEL              : {LOG_LEVEL}")
    print("=" * 50)
