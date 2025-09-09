"""
Wrap WhisperX transcription, alignment and diarization.

Function:
  transcribe_and_assign_speakers(wav_path, model_name, language, device, compute_type, hf_token)
    -> returns whisperx-style 'result' dict with segments and assigned speakers

Notes:
  - model_name, language, device, compute_type are passed so they can be changed at runtime.
  - hf_token is used for any HF model downloads that require authentication (e.g. diarization pipeline).
"""

import whisperx

def transcribe_and_assign_speakers(
    wav_path: str,
    model_name: str = "large-v2",
    language: str = "en",
    device: str = "cpu",
    compute_type: str = "int8",
    hf_token: str = None
):
    # Load a WhisperX model for transcription (model loading may download weights)
    print(f"[transcription] Loading WhisperX model '{model_name}' on device '{device}' (compute: {compute_type})...")
    model = whisperx.load_model(model_name, device=device, compute_type=compute_type, language=language)

    # Load audio and transcribe
    print("[transcription] Loading audio and running transcription...")
    audio = whisperx.load_audio(wav_path)
    result = model.transcribe(audio, batch_size=16)  # returns {'text', 'segments', ...}

    # Alignment (word-level timing)
    print("[transcription] Loading alignment model and aligning words...")
    align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
    result = whisperx.align(result["segments"], align_model, metadata, audio, device)

    # Diarization: produce speaker segments (requires HF token for some models)
    print("[transcription] Running diarization (this may download diarization models)...")
    diarize_pipeline = whisperx.diarize.DiarizationPipeline(use_auth_token=hf_token, device=device)
    diarize_segments = diarize_pipeline(audio)

    # Combine diarization with aligned transcription to assign speaker labels to words/segments
    print("[transcription] Assigning word speakers to transcription segments...")
    result = whisperx.assign_word_speakers(diarize_segments, result)

    return result
