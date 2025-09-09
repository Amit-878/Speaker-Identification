"""
Speaker verification / embedding utilities using SpeechBrain's ECAPA-TDNN model.

Functions:
  - load_verification_model(device): returns a SpeakerRecognition instance
  - compute_reference_embeddings(ref_config_path, verifier): loads refs from JSON and computes embeddings
  - embed_segments(verification, full_audio, sr, segments, target_sr=16000): batch-embeds segments
  - cosine_similarity / match_speaker: helpers for matching

Important implementation notes:
  - Reference files are expected to be reasonably clean and at least several seconds long.
  - The matching threshold (cosine) can be tuned for your dataset.
"""

import json
import numpy as np
import librosa
import torch
from speechbrain.inference import SpeakerRecognition

def load_verification_model(device: str = "cpu"):
    """
    Load the SpeechBrain ECAPA model for extracting speaker embeddings.
    The SpeechBrain method handles device placement via run_opts.
    """
    print(f"[speaker_verif] Loading SpeechBrain speaker-verification model on device '{device}'...")
    verifier = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device}
    )
    return verifier

def compute_reference_embeddings(ref_config_path: str, verifier) -> dict:
    """
    Reads a JSON mapping of {name: path} and returns a dict {name: embedding_np}.
    JSON path example: config/references.json
    """
    print("[speaker_verif] Loading reference files and computing embeddings...")
    with open(ref_config_path, "r", encoding="utf-8") as f:
        reference_files = json.load(f)

    embeddings = {}
    for name, path in reference_files.items():
        # librosa load ensures consistent sample rate 16 kHz
        audio, sr = librosa.load(path, sr=16000)
        audio_tensor = torch.tensor(audio).unsqueeze(0)  # shape: [1, T]
        with torch.no_grad():
            emb = verifier.encode_batch(audio_tensor)
        emb_np = emb.squeeze().detach().cpu().numpy()
        embeddings[name] = emb_np
        print(f"  - {name}: embedding shape {emb_np.shape}")

    return embeddings

def embed_segments(verifier, full_audio, sr, segments, target_sr=16000):
    """
    Given the full audio waveform and whisperx segments, slice each segment,
    resample if needed to target_sr, pad, and run batch embedding.
    Returns (segment_infos_list, embeddings_np_array)
    """
    print("[speaker_verif] Preparing segments for embedding...")
    tensors = []
    infos = []

    for seg in segments:
        start = int(seg["start"] * sr)
        end = int(seg["end"] * sr)
        segment_audio = full_audio[start:end]

        # resample to 16k if needed
        if sr != target_sr:
            segment_audio = librosa.resample(segment_audio, orig_sr=sr, target_sr=target_sr)

        if segment_audio.size == 0:
            # skip empty segments (very short)
            continue

        t = torch.tensor(segment_audio, dtype=torch.float32)  # shape [T]
        tensors.append(t)
        infos.append(seg)

    if len(tensors) == 0:
        return infos, np.zeros((0,))  # nothing to do

    # pad to make a batch (B, max_T)
    batched = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)

    # Move to expected device if possible (SpeechBrain was created with run_opts={"device": device})
    # We try to move batched to same device as verifier modules if accessible. If not, leave on CPU.
    try:
        # Inspect the first parameter of model to find device
        device = next(verifier.modules()).parameters().__iter__().__next__().device
    except Exception:
        device = batched.device  # fallback

    try:
        batched = batched.to(device)
    except Exception:
        pass

    print("[speaker_verif] Running batch embedding (this may use GPU if available)...")
    with torch.no_grad():
        embeddings = verifier.encode_batch(batched)

    embeddings_np = embeddings.detach().cpu().numpy()
    return infos, embeddings_np

def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1).flatten()
    v2 = np.array(vec2).flatten()
    dot = np.dot(v1, v2)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-9
    return float(dot / denom)

def match_speaker(segment_emb, reference_embeddings: dict, threshold: float = 0.37):
    """
    Given a segment embedding and dict of reference embeddings {name: emb},
    return the best-matching name or "Unknown" if below threshold.
    """
    scores = {}
    for name, ref_emb in reference_embeddings.items():
        scores[name] = cosine_similarity(segment_emb, ref_emb)

    if not scores:
        return "Unknown"

    best = max(scores, key=scores.get)
    best_score = scores[best]
    return best if best_score >= threshold else "Unknown"
