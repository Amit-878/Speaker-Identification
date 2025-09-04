
# 🎙️ Speaker Identification with WhisperX + SpeechBrain

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
│── main.py                # Entry point
│── requirements.txt       # Dependencies
│── README.md              # Documentation
│── config/
│   └── references.json    # Reference speakers and their audio
│── utils/
│   ├── speaker_utils.py   # Speaker recognition helper functions
│   ├── audio_utils.py     # Audio segmentation helpers
│── outputs/
│   ├── MeetingResult.txt  # Transcript results
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/speaker-identification.git
cd speaker-identification
```

### 2️⃣ Create & activate virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ Make sure you have **Python 3.9+** and **PyTorch** installed.
> If `torch` doesn’t install automatically, follow [PyTorch installation guide](https://pytorch.org/get-started/locally/).

### 4️⃣ Hugging Face Token (required for diarization)

1. Create a free account on [Hugging Face](https://huggingface.co/join).
2. Get your **access token** from [settings/tokens](https://huggingface.co/settings/tokens).
3. Open `main.py` and replace:

   ```python
   HF_TOKEN = "your_hf_token_here"
   ```

---

## 🎤 Usage

### 1️⃣ Prepare your audio

* Place your meeting audio file in the project root (e.g., `meeting.wav`) or provide the path when executing the code.
* Recommended format: `.wav` (16kHz mono for best results).

### 2️⃣ Add reference speakers

Edit `config/references.json`:

```json
{
  "Amit": "refs/amit.wav",
  "Shivali": "refs/shivali.wav",
  "Mudit": "refs/mudit.wav"
}
```

👉 Each key is a **speaker name**, each value is the **path to their sample audio**.
👉 Sample clips should be at least **10 seconds** for best accuracy.

### 3️⃣ Run transcription + speaker labeling 

```bash
python main.py your_audio_file.wav
```

### 4️⃣ View results

Transcript with speaker names will be saved in:

```
outputs/MeetingResult.txt
```

---

## 🧠 Models Used

### 🎧 WhisperX

* Based on [OpenAI Whisper](https://github.com/openai/whisper)
* Supports **word-level alignment**
* Used for **speech-to-text**

### 🗣️ SpeechBrain ECAPA-TDNN

* Model: `speechbrain/spkrec-ecapa-voxceleb`
* Used for **speaker verification**
* Compares diarized segments to **reference audios**

### 🔊 PyTorch + Librosa + SoundFile

* Audio preprocessing & segment extraction

---

## 🛠️ Example Workflow

1. Input: `meeting.wav` in the base directory.
2. WhisperX → Transcribe + align words
3. Diarization → Split speakers
4. Speaker verification → Match with reference audios
5. Output: `MeetingResult.txt` with **real names**

---

## 📌 Example Reference File (`config/references.json`)

```json
{
    "Amit": "refs/amit.wav",
    "Chirag": "refs/chirag.wav",
    "Shivali": "refs/shivali.wav"
}
```

---

## 📋 Requirements

* Python 3.9+
* torch >= 2.0
* whisperx
* speechbrain
* librosa
* numpy
* soundfile
* scipy
* transformers

(Already included in `requirements.txt`)

---

## 🤝 Contributing

Pull requests are welcome!
If you find a bug or want a feature, open an issue.

---

## 📜 License

This project is licensed under the **MIT License** – feel free to use and modify.

