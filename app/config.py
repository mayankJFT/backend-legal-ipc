import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# === Logging Setup ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("NyayaGPT-API")

# === Redis Configuration ===
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_TTL = int(os.getenv("REDIS_TTL", 60 * 60 * 24 * 7))  # Default 7 days
CACHE_TTL = int(os.getenv("CACHE_TTL", 60 * 60 * 24))  # Cache responses for 24 hours

# === Pinecone Configuration ===
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "2025-judgements-index")

# === OpenAI Configuration ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# === Server Configuration ===
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
WORKERS = int(os.getenv("WORKERS", 1))

# === Available Models ===
AVAILABLE_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]