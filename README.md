
# 🎙️ Speaker Identification with WhisperX + SpeechBrain

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python\&logoColor=white)](https://www.python.org/)
[![WhisperX](https://img.shields.io/badge/WhisperX-Transcription-orange?logo=openai\&logoColor=white)](https://github.com/m-bain/whisperx)
[![SpeechBrain](https://img.shields.io/badge/SpeechBrain-Speaker%20Verification-green)](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
[![HuggingFace](https://img.shields.io/badge/Models-HuggingFace-yellow?logo=huggingface\&logoColor=white)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

![header](https://capsule-render.vercel.app/api?type=waving\&color=gradient\&height=200\&section=header\&text=Meeting+Transcriber\&fontSize=40\&fontAlignY=35\&desc=WhisperX+%2B+SpeechBrain\&descAlignY=55\&animation=fadeIn)

[![Typing SVG](https://readme-typing-svg.herokuapp.com?font=Fira+Code\&duration=3000\&pause=1000\&color=F77D26\&center=true\&vCenter=true\&width=435\&lines=🎙️+Meeting+Transcription;👥+Speaker+Diarization;🗣️+Custom+Speaker+Labels)](https://git.io/typing-svg)

Automatically transcribe meetings and assign **real speaker names** using reference audio files.
This project combines:

* **[WhisperX](https://github.com/m-bain/whisperX)** → Fast, accurate speech recognition + forced alignment
* **[SpeechBrain](https://speechbrain.github.io/)** → Speaker verification (ECAPA-TDNN)
* **PyTorch** → Deep learning backend

---

## 🚀 Features

✅ Transcribes meeting audio using **WhisperX**
✅ Aligns timestamps with word-level precision
✅ Performs **speaker diarization** (who spoke when)
✅ Identifies speakers using **reference voice samples**
✅ Exports transcript in **speaker-labeled text file**

Example Output:

```
[Amit]: Good morning everyone, let's start the meeting.  
[Shivali]: Yes, I wanted to share an update on the project.  
[Unknown]: Sorry, could you repeat that?  
```

---

## 📂 Project Structure

```
speaker-identification/
│── main.py                        # Entry point
│── config.py                       # Reads .env for HF token and Whisper settings
│── requirements.txt                # Dependencies
│── README.md                       # Documentation
│── .env                            # Environment variables (HF_TOKEN, Whisper model, etc.)
│── .gitignore                      # Git ignore rules
│
├── refs/                            # Reference speaker audio files
│   ├── amit.wav
│   ├── shivali.wav
│   ├── mudit.wav
│   └── ...                          # Add more speaker files here
│
├── config/                          # JSON config for references
│   └── references.json
│
├── src/                             # Source code modules
│   ├── pipeline.py                  # High-level orchestration
│   ├── speaker_utils.py             # Speaker caching, embedding, and matching
│   ├── speaker_verification.py      # Speaker embedding helpers
│   ├── audio_utils.py               # Audio conversion / slicing utilities
│   └── transcription.py             # WhisperX transcription & diarization
│
└── outputs/                         # Generated transcripts
    └── <audio_basename>_result.txt  # Example: meeting_result.txt
```

---

## ⚡ Prerequisites

This project depends on models hosted on **Hugging Face**.

1. Python **3.9+**
2. Install [PyTorch](https://pytorch.org/get-started/locally/)
3. Create a Hugging Face account (free) → [Sign up here](https://huggingface.co/join)
4. Generate a **User Access Token** → [settings/tokens](https://huggingface.co/settings/tokens)
5. Accept model licenses on the Hugging Face pages (one-time):

   * WhisperX alignment models: [https://huggingface.co/openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3)
   * SpeechBrain ECAPA VoxCeleb: [https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)

> ⚠️ If you don’t accept the licenses, the script may pause and request manual approval.

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/speaker-identification.git
cd speaker-identification
```

### 2️⃣ Create & activate virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Set environment variables

Create `.env` in the root directory:

```env
HF_TOKEN=your_hf_token_here
WHISPER_MODEL=large-v2
WHISPER_LANGUAGE=en
WHISPER_DEVICE=cpu
WHISPER_COMPUTE=int8
SIMILARITY_THRESHOLD=0.37
```

> These settings are now **dynamic**, so you can change the model, device, or threshold without modifying `main.py`.

---

## 🎤 Usage

1. Prepare your meeting audio (`.wav`, `.mp3`, or `.m4a`).
2. Add reference speakers in `config/references.json`:

```json
{
  "Amit": "refs/amit.wav",
  "Shivali": "refs/shivali.wav",
  "Mudit": "refs/mudit.wav"
}
```

3. Run the pipeline:

```bash
python main.py path/to/your_audio_file.wav
```

4. Transcript will be saved to `outputs/<audio_basename>_result.txt`.

> Audio files will be automatically converted to 16kHz WAV if needed.

---

## 🧠 Models Used

* **WhisperX** → Speech-to-text + word-level alignment
* **SpeechBrain ECAPA-TDNN** → Speaker verification against reference audios
* **Librosa / SoundFile / PyTorch** → Audio preprocessing and embeddings

---

## 🚧 Roadmap

* [x] Support `.wav`, `.mp3`, `.m4a` input with auto-conversion
* [x] Dynamic Whisper model, language, device, compute type via `.env`
* [x] Automatic embeddings cache update when adding/removing/updating reference audios
* [ ] Export transcripts to Word/Excel/JSON
* [ ] Build a simple web UI
* [ ] Support real-time transcription

---

## 🤝 Contributing

Pull requests are welcome!
Report bugs or request features via GitHub issues.
Star the project 🌟 to support development.

---

## 📜 License

This project is licensed under **MIT License** – feel free to use and modify.


If you want, I can also **update the “Example Workflow” diagrams/snippets** to explicitly show `.mp3/.m4a` conversion in the README for new users. Do you want me to do that?
