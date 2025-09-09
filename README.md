# 🎙️ Speaker Identification with WhisperX + SpeechBrain

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python\&logoColor=white)](https://www.python.org/)
[![WhisperX](https://img.shields.io/badge/WhisperX-Transcription-orange?logo=openai\&logoColor=white)](https://github.com/m-bain/whisperx)
[![SpeechBrain](https://img.shields.io/badge/SpeechBrain-Speaker%20Verification-green)](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
[![HuggingFace](https://img.shields.io/badge/Models-HuggingFace-yellow?logo=huggingface\&logoColor=white)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

![header](https://capsule-render.vercel.app/api?type=waving\&color=gradient\&height=200\&section=header\&text=Meeting+Transcriber\&fontSize=40\&fontAlignY=35\&desc=WhisperX+%2B+SpeechBrain\&descAlignY=55\&animation=fadeIn)

[![Typing SVG](https://readme-typing-svg.herokuapp.com?font=Fira+Code\&duration=3000\&pause=1000\&color=F77D26\&center=true\&vCenter=true\&width=435\&lines=🎙️+Meeting+Transcription;👥+Speaker+Diarization;🗣️+Custom+Speaker+Labels)](https://git.io/typing-svg)

Automatically transcribe meetings and assign **real speaker names** using reference audio files.

---

## 🚀 Features

✅ Transcribes meeting audio using **WhisperX**
✅ Aligns timestamps with **word-level precision**
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

This project depends on models hosted on **Hugging Face**, and **FFmpeg** for audio conversion.

> ⚠️ **Python Version:** Python **3.11** is **recommended**. Versions 3.12+ are **not supported** due to lack of compatible wheels for WhisperX, SpeechBrain, and other dependencies.

1. **Install Python 3.11**

   * On Windows, if `python` command fails, try `py -3.11`
   * [Download Python 3.11](https://www.python.org/downloads/release/python-311x/)

2. **Install PyTorch**

   * [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)

3. **Install FFmpeg (full version)**

   * Download from [Gyan.dev FFmpeg Builds](https://www.gyan.dev/ffmpeg/builds/)
   * Extract and add the `bin` folder to your **PATH**
   * Verify installation: `ffmpeg -version`

4. **Create a Hugging Face account** → [Sign up here](https://huggingface.co/join)

5. **Generate a User Access Token** → [settings/tokens](https://huggingface.co/settings/tokens)

6. Accept model licenses (one-time):

   * WhisperX alignment models: [https://huggingface.co/openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3)
   * SpeechBrain ECAPA VoxCeleb: [https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/speaker-identification.git
cd speaker-identification
```

### 2️⃣ Create & activate virtual environment

**Windows PowerShell:**

```powershell
python -m venv venv
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
.\venv\Scripts\Activate.ps1
```

**Windows CMD:**

```cmd
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**

```bash
python -m venv venv
source venv/bin/activate
```

> ⚠️ If `python` commands fail, try `py` on Windows, e.g., `py -3.11 -m venv venv`

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

```if pip is not defined
py -m pip install -r requirements.txt
```

### 4️⃣ Set environment variables

Create `.env` in the root directory:

```env
HF_TOKEN=your_hf_token_here
WHISPER_MODEL=large-v2       # options: tiny, base, small, medium, large-v1, large-v2
WHISPER_LANGUAGE=en          # language of the audio (e.g., en, fr, de, es, it, ja, zh, hi)
WHISPER_DEVICE=cpu           # options: cpu, cuda, mps
WHISPER_COMPUTE=int8         # options: int8, float16, float32
SIMILARITY_THRESHOLD=0.37    # threshold for speaker matching
```

> These settings are **dynamic** and can be changed anytime without modifying `main.py`.

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

> Audio files will be automatically converted to **16kHz WAV** if needed.

4. View transcript:

```
outputs/<audio_basename>_result.txt
```

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

Pull requests welcome!
Star the project 🌟 if you find it useful.

---

## 📜 License

MIT License – modify and use freely.

