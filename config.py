"""
Load configuration from environment (.env). Sensible defaults are provided.

Edit .env (or export env variables) to change behavior:
  - HF_TOKEN: your Hugging Face token (optional if models are public)
  - WHISPER_MODEL: tiny|base|small|medium|large-v1|large-v2  (default: large-v2)
  - WHISPER_LANGUAGE: language code (default: en)
  - WHISPER_DEVICE: cpu|cuda|mps  (default: cpu)
  - WHISPER_COMPUTE: int8|float16  (default: int8)
  - SIMILARITY_THRESHOLD: float between 0 and 1 for speaker matching (default: 0.37)
"""
from dotenv import load_dotenv
import os

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", None)

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v2")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")

# When matching segment embeddings against reference embeddings, the cosine
# similarity must be >= this threshold to accept the match. Tune as needed.
SIMILARITY_THRESHOLD = os.getenv("SIMILARITY_THRESHOLD", "0.37")
