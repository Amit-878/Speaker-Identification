import librosa
import soundfile as sf
import os

def extract_segment(audio_file, start, end, output_path):
    """Extract a segment from audio and save as wav."""
    y, sr = librosa.load(audio_file, sr=None, offset=start, duration=end-start)
    sf.write(output_path, y, sr)
    return output_path

def ensure_dir(path):
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)
