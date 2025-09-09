# import numpy as np
# import os
# import torch
# import librosa

# # -----------------------------
# # Speaker Embedding Helpers
# # -----------------------------

# def embed_from_file(file_path, verification):
#     """
#     Compute speaker embedding from a reference audio file.
#     """
#     audio, sr = librosa.load(file_path, sr=16000)
#     audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)  # [1, T]
#     emb = verification.encode_batch(audio)
#     return emb.squeeze().detach().cpu().numpy().flatten()


# def load_reference_embeddings(reference_files, verification, cache_file="refs/embeddings.npy"):
#     """
#     Load reference speaker embeddings with caching.

#     Features:
#     - Computes embeddings for new speakers
#     - Updates embeddings if the reference audio file changed
#     - Removes embeddings if the reference file was deleted
#     - Saves cache to `refs/embeddings.npy`
#     """
#     embeddings = {}
#     metadata = {}

#     # Load existing cache
#     if os.path.exists(cache_file):
#         cache = np.load(cache_file, allow_pickle=True).item()
#         embeddings = cache.get("embeddings", {})
#         metadata = cache.get("metadata", {})

#     updated = False
#     current_speakers = set(reference_files.keys())

#     # Add/update embeddings
#     for name, path in reference_files.items():
#         if not os.path.exists(path):
#             print(f"[speaker_verif] Warning: Missing file for {name} ({path}), skipping.")
#             continue

#         file_mtime = os.path.getmtime(path)  # last modified timestamp

#         if (
#             name not in embeddings or      # new speaker
#             name not in metadata or        # no metadata saved
#             metadata[name] != file_mtime   # file changed
#         ):
#             print(f"[speaker_verif] Computing embedding for {name} (new or updated)...")
#             embeddings[name] = embed_from_file(path, verification)
#             metadata[name] = file_mtime
#             updated = True

#     # Prune deleted speakers
#     cached_speakers = set(embeddings.keys())
#     removed = cached_speakers - current_speakers
#     if removed:
#         print(f"[speaker_verif] Removing {len(removed)} deleted speakers: {', '.join(removed)}")
#         for r in removed:
#             embeddings.pop(r, None)
#             metadata.pop(r, None)
#         updated = True

#     # Save cache if needed
#     if updated:
#         np.save(cache_file, {"embeddings": embeddings, "metadata": metadata})
#         print(f"[speaker_verif] Cache updated with {len(embeddings)} speakers.")
#     else:
#         print(f"[speaker_verif] Loaded cached embeddings for {len(embeddings)} speakers.")

#     return embeddings


"""
Speaker utilities:
- Manage reference embeddings with caching (add, update, prune).
- Safe load/save of cache to prevent corruption.
"""
import shutil
import os
import json
import hashlib
import numpy as np
import librosa
import torch

# Paths (relative to project root)
REFS_DIR = "refs"
CACHE_FILE = os.path.join(REFS_DIR, "embeddings.npy")


# ----------------------------
# Safe cache I/O
# ----------------------------
def load_cache() -> dict:
    """Safely load cached embeddings, return {} if invalid/corrupt."""
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        data = np.load(CACHE_FILE, allow_pickle=True).item()
        if isinstance(data, dict):
            return data
        else:
            print("[speaker_utils] Cache format invalid, resetting...")
            return {}
    except Exception as e:
        print(f"[speaker_utils] Failed to load cache ({e}), resetting...")
        return {}


# def save_cache(cache: dict):
#     """Atomically save cache to avoid corruption."""
#     os.makedirs(REFS_DIR, exist_ok=True)
#     tmp_file = CACHE_FILE + ".tmp"
#     np.save(tmp_file, cache, allow_pickle=True)
#     os.replace(tmp_file, CACHE_FILE)

def save_cache(cache: dict):
    """Atomically save cache to avoid corruption (Windows-safe)."""
    os.makedirs(REFS_DIR, exist_ok=True)

    # Use .npy explicitly
    tmp_file = os.path.join(REFS_DIR, "embeddings_tmp.npy")
    
    # Save cache to temp file
    np.save(tmp_file, cache, allow_pickle=True)
    
    # Move temp file to final location
    shutil.move(tmp_file, os.path.join(REFS_DIR, "embeddings.npy"))


# ----------------------------
# Embedding helpers
# ----------------------------
def file_hash(path: str) -> str:
    """Return SHA1 hash of file contents (to detect changes)."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def embed_from_file(file_path: str, verifier) -> np.ndarray:
    """Compute embedding from an audio file."""
    audio, _ = librosa.load(file_path, sr=16000)
    tensor = torch.tensor(audio).unsqueeze(0)  # [1, T]
    emb = verifier.encode_batch(tensor)
    return emb.squeeze().detach().cpu().numpy().flatten()


# ----------------------------
# Main reference embeddings loader
# ----------------------------
def load_reference_embeddings(ref_config_path: str, verifier) -> dict:
    """
    Load reference embeddings from config file.
    Keeps a persistent cache synced with references.json.
    Auto-adds new, updates changed, prunes removed.
    """
    # Load JSON reference config
    with open(ref_config_path, "r", encoding="utf-8") as f:
        references = json.load(f)

    # Load existing cache safely
    cache = load_cache()

    # Ensure all refs exist
    updated = False
    valid_refs = {}

    for name, path in references.items():
        if not os.path.exists(path):
            print(f"[speaker_utils] WARNING: file not found for {name}: {path}")
            continue

        fhash = file_hash(path)
        cache_entry = cache.get(name)

        # If new speaker or file changed → recompute
        if cache_entry is None or cache_entry.get("hash") != fhash:
            print(f"[speaker_utils] Updating embedding for {name}...")
            emb = embed_from_file(path, verifier)
            cache[name] = {"hash": fhash, "embedding": emb}
            updated = True
        else:
            print(f"[speaker_utils] Using cached embedding for {name}")

        valid_refs[name] = cache[name]

    # Prune speakers no longer in references.json
    to_remove = set(cache.keys()) - set(references.keys())
    for r in to_remove:
        print(f"[speaker_utils] Removing stale speaker: {r}")
        del cache[r]
        updated = True

    # Save updated cache if needed
    if updated:
        save_cache(cache)

    # Return dict {name: embedding}
    return {name: entry["embedding"] for name, entry in valid_refs.items()}
