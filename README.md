# Meeting Transcriber

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/Platform-macOS%20Apple%20Silicon-lightgrey.svg)](https://support.apple.com/en-us/HT211814)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-95%20passed-brightgreen.svg)](tests/)
[![Offline](https://img.shields.io/badge/works-100%25%20offline-blueviolet.svg)]()

**Turn your meeting recordings into searchable, speaker-labeled transcripts — 100% locally, no cloud required.**

[Leia em Português](README.pt.md)

---

## What It Does

Meeting Transcriber converts audio files into text with automatic speaker identification. Your audio never leaves your computer.

**Perfect for:**
- Team meetings and interviews
- Lectures and presentations
- WhatsApp voice messages
- Podcasts and recordings

**Output example:**
```
[00:00] SPEAKER_00: Good morning everyone, let's start the meeting.
[00:05] SPEAKER_01: Thanks for joining. First item on the agenda...
[00:12] SPEAKER_00: Before we begin, any updates from last week?
```

---

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/yourusername/meeting-transcriber.git
cd meeting-transcriber
python3.12 -m venv venv && source venv/bin/activate

# 2. Install
brew install ffmpeg
pip install -r requirements.txt

# 3. Transcribe!
python src/transcribe.py your-audio.mp3
```

> **First time?** You'll need a free [HuggingFace account](#huggingface-setup) for speaker identification.

---

## Supported Audio Formats

| Format | Extension | Common Source |
|--------|-----------|---------------|
| MP3 | `.mp3` | Most audio players |
| WAV | `.wav` | Professional recordings |
| M4A | `.m4a` | iPhone/Mac recordings |
| Opus | `.opus` | WhatsApp voice messages |
| FLAC | `.flac` | Lossless audio |
| OGG | `.ogg` | Web recordings |
| WebM | `.webm` | Browser recordings |
| AAC | `.aac` | Digital broadcasts |

---

## Installation

### Requirements

| Requirement | Details |
|-------------|---------|
| **Computer** | Mac with Apple Silicon (M1, M2, M3, or M4 chip) |
| **Python** | Version 3.12 (not 3.14) |
| **Disk Space** | ~10GB for models |
| **Internet** | Only for initial setup |

### Step-by-Step Setup

<details>
<summary><strong>1. Install Python 3.12</strong> (if not installed)</summary>

```bash
# Using Homebrew
brew install python@3.12

# Verify installation
python3.12 --version
```
</details>

<details>
<summary><strong>2. Install FFmpeg</strong> (audio processing)</summary>

```bash
brew install ffmpeg
```

FFmpeg is a free tool that handles audio format conversion.
</details>

<details>
<summary><strong>3. Set up the project</strong></summary>

```bash
# Clone the repository
git clone https://github.com/yourusername/meeting-transcriber.git
cd meeting-transcriber

# Create isolated Python environment
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
</details>

<details>
<summary><strong>4. Configure HuggingFace</strong> (required for speaker identification)</summary>

<a name="huggingface-setup"></a>

HuggingFace provides the AI models for speaker identification. It's free.

1. **Create account** at [huggingface.co](https://huggingface.co/join)

2. **Get your token:**
   - Go to [Settings → Access Tokens](https://huggingface.co/settings/tokens)
   - Click "New token" → Name it anything → Create
   - Copy the token (starts with `hf_`)

3. **Save the token:**
   ```bash
   cp .env.example .env
   # Edit .env and paste your token after HF_TOKEN=
   ```

4. **Accept model terms** (one-time):
   - Visit [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - Click "Agree and access repository"
   - Visit [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   - Click "Agree and access repository"

</details>

### Optional: Additional Backends

The default installation covers most needs. For specialized use cases:

| Mode | Install Command | Best For |
|------|-----------------|----------|
| `fast` | `pip install mlx-whisper` | Quick transcriptions (no speaker ID) |
| `precise` | `pip install transformers accelerate` | Maximum accuracy (IBM Granite) |

```bash
# Install all backends (recommended for power users)
pip install mlx-whisper transformers accelerate
```

<details>
<summary><strong>Low RAM? Use the smaller Granite model</strong></summary>

The default Granite model (8B) requires ~16GB RAM. For machines with less memory, use the 2B model (~6GB RAM):

```bash
# Add to your .env file:
GRANITE_MODEL=ibm-granite/granite-speech-3.3-2b
```

| Model | Parameters | RAM Required | Accuracy |
|-------|------------|--------------|----------|
| granite-speech-3.3-8b | 8 billion | ~16GB | Higher |
| granite-speech-3.3-2b | 2 billion | ~6GB | Good |

</details>

---

## Usage Guide

### Basic Usage

```bash
# Transcribe any supported audio file
python src/transcribe.py meeting.mp3

# Output: Creates meeting.json, meeting.txt, and meeting.md
```

### Choose Your Mode

| I want... | Command | Notes |
|-----------|---------|-------|
| **Best quality** (default) | `python src/transcribe.py audio.mp3` | Speaker identification included |
| **Fastest result** | `python src/transcribe.py audio.mp3 --mode fast` | No speaker identification |
| **Maximum accuracy** | `python src/transcribe.py audio.mp3 --mode precise` | Slower but most accurate |

### Common Options

```bash
# Specify language (improves accuracy)
python src/transcribe.py meeting.mp3 --language pt

# Know how many speakers? Tell the system
python src/transcribe.py meeting.mp3 --num-speakers 3

# Get notified when done (useful for long files)
python src/transcribe.py meeting.mp3 --notify

# Only generate text file (faster)
python src/transcribe.py meeting.mp3 --format txt
```

### All Options Reference

<details>
<summary>Click to expand full options table</summary>

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--model` | `-m` | AI model size (tiny/base/small/medium/large-v3) | large-v3 |
| `--language` | `-l` | Audio language (en, pt, es, etc.) | auto-detect |
| `--num-speakers` | `-n` | Exact number of speakers | auto-detect |
| `--min-speakers` | | Minimum speakers expected | - |
| `--max-speakers` | | Maximum speakers expected | - |
| `--output` | `-o` | Where to save files | data/transcripts |
| `--format` | `-f` | Output format (json/txt/md/all) | all |
| `--mode` | | Transcription mode (fast/meeting/precise) | meeting |
| `--device` | `-d` | Processor (cpu/cuda/mps) | cpu |
| `--notify` | | macOS notification when done | off |
| `--vocab` | | Custom vocabulary file | - |
| `--ui-lang` | | Interface language (en/pt) | auto |
| `--verbose` | `-v` | Show detailed logs | off |

</details>

---

## Output Formats

### Text (.txt) — Human-readable
```
[00:00] SPEAKER_00: Good morning everyone.
[00:05] SPEAKER_01: Thanks for joining.
```

### Markdown (.md) — Formatted for documents
```markdown
## Meeting Transcript

**[00:00] Speaker 1:** Good morning everyone.

**[00:05] Speaker 2:** Thanks for joining.
```

### JSON (.json) — For developers
```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 4.2,
      "text": "Good morning everyone.",
      "speaker": "SPEAKER_00"
    }
  ],
  "metadata": {
    "language": "en",
    "num_speakers": 2
  }
}
```

---

## Model Selection

Larger models are more accurate but slower and use more memory.

| Model | Accuracy | Speed | RAM Needed | Recommended For |
|-------|----------|-------|------------|-----------------|
| tiny | Low | Very fast | 1GB | Testing only |
| base | Medium | Fast | 1GB | Quick drafts |
| small | Good | Moderate | 2GB | Daily use |
| medium | Very good | Slow | 5GB | Important meetings |
| **large-v3** | Excellent | Slower | 10GB | Production (default) |

```bash
# Use smaller model for testing
python src/transcribe.py meeting.mp3 --model small

# Use largest model for important recordings
python src/transcribe.py meeting.mp3 --model large-v3
```

---

## Troubleshooting

<details>
<summary><strong>Error: "No module named 'whisperx'"</strong></summary>

Your virtual environment isn't activated.

```bash
source venv/bin/activate
pip install -r requirements.txt
```
</details>

<details>
<summary><strong>Out of memory error</strong></summary>

Use a smaller model:

```bash
python src/transcribe.py audio.mp3 --model small
```
</details>

<details>
<summary><strong>Speakers not identified correctly</strong></summary>

Tell the system how many speakers:

```bash
python src/transcribe.py audio.mp3 --num-speakers 3
```
</details>

<details>
<summary><strong>Wrong language detected</strong></summary>

Specify the language explicitly:

```bash
python src/transcribe.py audio.mp3 --language pt
```
</details>

<details>
<summary><strong>Slow transcription</strong></summary>

- Use `--mode fast` for speed (no speaker identification)
- Use `--model small` for faster processing
- Close other applications to free up memory
</details>

<details>
<summary><strong>HuggingFace authentication error</strong></summary>

1. Check your token in `.env` file
2. Make sure you accepted terms for both pyannote models
3. Verify token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
</details>

---

## Privacy & Security

**Your data stays on your computer.**

- All processing happens locally — no audio is uploaded anywhere
- No internet connection needed after initial setup
- Models are downloaded once and stored locally
- No telemetry, analytics, or data collection

This makes Meeting Transcriber ideal for:
- Confidential business meetings
- Medical consultations
- Legal proceedings
- Personal recordings

---

## Custom Vocabulary

Improve accuracy for domain-specific terms:

```bash
# Create vocab/default.txt with your terms (one per line):
ACME Corporation
Dr. Smith
Kubernetes
API
```

The system automatically loads `vocab/default.txt` if it exists, or specify a custom file:

```bash
python src/transcribe.py meeting.mp3 --vocab my-terms.txt
```

---

## Integration with Claude

After transcription, use AI to generate meeting minutes:

```bash
# 1. Transcribe
python src/transcribe.py meeting.mp3

# 2. Ask Claude to process the transcript
# See prompts/ folder for templates:
#   - prompts/ata_sei.md — Formal meeting minutes (Brazilian SEI! format)
#   - prompts/resumo_executivo.md — Executive summary with action items
```

---

## Acknowledgments

Built with these excellent open-source projects:

- [WhisperX](https://github.com/m-bain/whisperX) — Speech recognition with word-level timestamps
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — Optimized Whisper inference
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) — Speaker diarization
- [MLX](https://github.com/ml-explore/mlx) — Apple Silicon ML framework

---

## License

MIT License — free for personal and commercial use.

---

## Author

**Iuri Almeida**
Medical Doctor | Public Safety Manager | Computer Science Student

*January 2026*
