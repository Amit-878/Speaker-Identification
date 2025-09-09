"""
Entry point for the Speaker-Identification pipeline.

Usage:
    python main.py <audio_file>

Only an audio path is required. All runtime settings (HF token, Whisper model,
language, device, compute type, threshold) are read from the .env file via config.py.
"""
import sys
import os
from dotenv import load_dotenv

# load env first so config.py can read environment variables
load_dotenv()

from config import (
    HF_TOKEN, WHISPER_MODEL, WHISPER_LANGUAGE, WHISPER_DEVICE, WHISPER_COMPUTE,
    SIMILARITY_THRESHOLD
)
from src.pipeline import run_pipeline

def print_usage():
    print("Usage: python main.py <audio_file>")
    print("Example: python main.py meeting.m4a")

def main():
    if len(sys.argv) < 2:
        print("Error: missing audio file path.")
        print_usage()
        sys.exit(1)

    input_path = sys.argv[1]
    if not os.path.exists(input_path):
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    # config options (read from .env via config.py)
    run_pipeline(
        input_path=input_path,
        ref_config_path=os.path.join("config", "references.json"),
        output_dir="outputs",
        hf_token=HF_TOKEN,
        model_name=WHISPER_MODEL,
        language=WHISPER_LANGUAGE,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE,
        similarity_threshold=float(SIMILARITY_THRESHOLD)
    )

if __name__ == "__main__":
    main()
