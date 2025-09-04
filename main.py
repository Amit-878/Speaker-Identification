import whisperx
import json
import os
from utils.speaker_utils import SpeakerUtils
from utils.audio_utils import extract_segment, ensure_dir

# Config
AUDIO_FILE = "meeting2sepconverted.wav"
OUTPUT_DIR = "outputs/Meeting_output2Sep_4"
RESULT_FILE = os.path.join("outputs", "MeetingResult2sep_4.txt")
MODEL_NAME = "base"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
HF_TOKEN = "your_hf_token_here"  # replace with your HuggingFace token

def load_references(config_path="config/references.json"):
    with open(config_path, "r") as f:
        return json.load(f)

def main():
    ensure_dir(OUTPUT_DIR)

    # Load references
    reference_files = load_references()

    # Load WhisperX model
    model = whisperx.load_model(MODEL_NAME, DEVICE, compute_type=COMPUTE_TYPE)
    audio = whisperx.load_audio(AUDIO_FILE)
    result = model.transcribe(audio, batch_size=16)

    # Alignment
    align_model, metadata = whisperx.load_align_model(language_code=result['language'], device=DEVICE)
    result = whisperx.align(result["segments"], align_model, metadata, audio, DEVICE)

    # Diarization
    diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    # Speaker utils
    speaker_utils = SpeakerUtils()

    # Write results
    with open(RESULT_FILE, "w", encoding="utf-8") as txt:
        for i, segment in enumerate(result["segments"]):
            start = segment["start"]
            end = segment["end"]
            segment_path = os.path.join(OUTPUT_DIR, f"segment_{i}.wav")

            # Extract segment
            extract_segment(AUDIO_FILE, start, end, segment_path)

            # Match speaker
            speaker_name = speaker_utils.get_speaker_name(segment_path, reference_files)

            # Write transcript
            text = segment.get("text", "").strip()
            txt.write(f"[{speaker_name}]: {text}\n")

    print(f"Speaker-labeled transcript saved at {RESULT_FILE}")

if __name__ == "__main__":
    main()
