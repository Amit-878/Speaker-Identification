import os
from speechbrain.inference import SpeakerRecognition

class SpeakerUtils:
    def __init__(self, model_name="speechbrain/spkrec-ecapa-voxceleb"):
        self.verification = SpeakerRecognition.from_hparams(source=model_name)

    def is_same_speaker(self, file1: str, file2: str) -> bool:
        """Check if two audio files are from the same speaker."""
        score, prediction = self.verification.verify_files(file1, file2)
        return bool(prediction)

    def get_speaker_name(self, segment_path: str, reference_files: dict, threshold: float = 0.2) -> str:
        """Find best matching speaker for a segment."""
        scores = {}
        for name, ref_path in reference_files.items():
            if not os.path.exists(ref_path):
                continue
            score, _ = self.verification.verify_files(segment_path, ref_path)
            scores[name] = score

        if not scores:
            return "Unknown"

        best_speaker = max(scores, key=scores.get)
        best_score = scores[best_speaker]

        return best_speaker if best_score >= threshold else "Unknown"
