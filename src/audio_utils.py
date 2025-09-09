import os
import subprocess

def ensure_wav(input_path, sr=16000):
    """
    Ensure the audio file is in WAV format. 
    If it's already .wav, return it.
    Otherwise convert to WAV and return converted path.
    """
    base, ext = os.path.splitext(input_path)
    if ext.lower() == ".wav":
        return input_path  # no conversion needed
    
    output_path = f"{base}.wav"
    subprocess.run([
        "ffmpeg", "-y",
        "-i", input_path,
        "-ar", str(sr),
        "-ac", "1",   # mono
        output_path
    ], check=True)
    return output_path
