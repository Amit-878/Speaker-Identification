# """
# High-level pipeline orchestration: conversion -> transcription -> embeddings -> matching -> write file.
# """

# import os
# import librosa
# from .audio_utils import ensure_wav
# from .transcription import transcribe_and_assign_speakers
# from .speaker_verification import (
#     load_verification_model, compute_reference_embeddings,
#     embed_segments, match_speaker
# )

# def run_pipeline(
#     input_path: str,
#     ref_config_path: str = "config/references.json",
#     output_dir: str = "outputs",
#     hf_token: str = None,
#     model_name: str = "large-v2",
#     language: str = "en",
#     device: str = "cpu",
#     compute_type: str = "int8",
#     similarity_threshold: float = 0.37
# ):
#     # 1) Ensure WAV exists (converted file will be <basename>.wav)
#     print(f"[pipeline] Ensuring WAV for '{input_path}'...")
#     wav_path = ensure_wav(input_path, sr=16000)

#     # 2) Output filepath
#     os.makedirs(output_dir, exist_ok=True)
#     base_name = os.path.splitext(os.path.basename(input_path))[0]
#     output_file = os.path.join(output_dir, f"{base_name}_result.txt")

#     # 3) Transcription + diarization + alignment
#     print("[pipeline] Transcribing & diarizing...")
#     result = transcribe_and_assign_speakers(
#         wav_path,
#         model_name=model_name,
#         language=language,
#         device=device,
#         compute_type=compute_type,
#         hf_token=hf_token
#     )

#     # 4) Load speaker verification model and compute reference embeddings
#     verifier = load_verification_model(device=device)
#     ref_embeddings = compute_reference_embeddings(ref_config_path, verifier)

#     # 5) Load full audio waveform for slicing per segment
#     print("[pipeline] Loading full audio for segment slicing...")
#     full_audio, sr = librosa.load(wav_path, sr=None)

#     # 6) Embed all segments in batch
#     segment_infos, all_embeddings = embed_segments(verifier, full_audio, sr, result.get("segments", []))

#     # 7) Match embeddings to references and write output
#     print(f"[pipeline] Writing results to '{output_file}' ...")
#     with open(output_file, "w", encoding="utf-8") as out_f:
#         for seg, emb in zip(segment_infos, all_embeddings):
#             speaker = match_speaker(emb, ref_embeddings, threshold=similarity_threshold)
#             text = seg.get("text", "").strip()
#             out_f.write(f"[{speaker}]: {text}\n")

#     print(f"[pipeline] Done. Transcript written to: {output_file}")



"""
High-level pipeline orchestration: conversion -> transcription -> embeddings -> matching -> write file.
"""

import os
import librosa
from .audio_utils import ensure_wav
from .transcription import transcribe_and_assign_speakers
from .speaker_verification import (
    load_verification_model,
    embed_segments,
    match_speaker
)
from .speaker_utils import load_reference_embeddings   # <-- NEW

def run_pipeline(
    input_path: str,
    ref_config_path: str = "config/references.json",
    output_dir: str = "outputs",
    hf_token: str = None,
    model_name: str = "large-v2",
    language: str = "en",
    device: str = "cpu",
    compute_type: str = "int8",
    similarity_threshold: float = 0.37
):
    # 1) Ensure WAV exists (converted file will be <basename>.wav)
    print(f"[pipeline] Ensuring WAV for '{input_path}'...")
    wav_path = ensure_wav(input_path, sr=16000)

    # 2) Output filepath
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_file = os.path.join(output_dir, f"{base_name}_result.txt")

    # 3) Transcription + diarization + alignment
    print("[pipeline] Transcribing & diarizing...")
    result = transcribe_and_assign_speakers(
        wav_path,
        model_name=model_name,
        language=language,
        device=device,
        compute_type=compute_type,
        hf_token=hf_token
    )

    # 4) Load speaker verification model
    verifier = load_verification_model(device=device)

    # 5) Load (or update) reference embeddings with caching
    ref_embeddings = load_reference_embeddings(ref_config_path, verifier)

    # 6) Load full audio waveform for slicing per segment
    print("[pipeline] Loading full audio for segment slicing...")
    full_audio, sr = librosa.load(wav_path, sr=None)

    # 7) Embed all segments in batch
    segment_infos, all_embeddings = embed_segments(verifier, full_audio, sr, result.get("segments", []))

    # 8) Match embeddings to references and write output
    print(f"[pipeline] Writing results to '{output_file}' ...")
    with open(output_file, "w", encoding="utf-8") as out_f:
        for seg, emb in zip(segment_infos, all_embeddings):
            speaker = match_speaker(emb, ref_embeddings, threshold=similarity_threshold)
            text = seg.get("text", "").strip()
            out_f.write(f"[{speaker}]: {text}\n")

    print(f"[pipeline] Done. Transcript written to: {output_file}")
