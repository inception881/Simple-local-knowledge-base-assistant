"""
Configuration Management
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration class for the application"""
    
    # Path configurations
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    DOCUMENTS_DIR = DATA_DIR / "documents"
    FAISS_INDEX_PATH = DATA_DIR / "faiss_index"
    PROCESSED_DOCS_RECORD = DATA_DIR / "processed_docs.txt"
    LOG_DIR = BASE_DIR / "logs"
    LOG_FILE = LOG_DIR / "app.log"
    DOCUMENTS_PATH = DOCUMENTS_DIR
    
    # LLM configuration
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_LLM_API_KEY")
    ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL")
    ANTHROPIC_MODEL_NAME = os.getenv("ANTHROPIC_MODEL_NAME")
    
    # OpenAI Embedding configuration
    OPENAI_API_KEY = os.getenv("OPENAI_EMBEDDING_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_EMBEDDING_BASE_URL")
    OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL_NAME")
    
    # General configuration
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))
    
    # Retrieval configuration
    TOP_K = 5
    SIMILARITY_THRESHOLD = 0.7
    
    # Chunking configuration
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Memory configuration
    MAX_HISTORY_LENGTH = 20
    MEMORY_KEY = "chat_history"
    
    
    @classmethod
    def ensure_dirs(cls):
        """Ensure all required directories exist"""
        for dir_path in [
            cls.DATA_DIR, 
            cls.DOCUMENTS_DIR, 
            cls.FAISS_INDEX_PATH,
            cls.LOG_DIR
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

Config.ensure_dirs()
