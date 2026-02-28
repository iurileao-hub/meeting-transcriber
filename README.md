# Meeting Transcriber

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/Platform-macOS%20Apple%20Silicon-lightgrey.svg)](https://support.apple.com/en-us/HT211814)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-289%20passed-brightgreen.svg)](tests/)
[![Offline](https://img.shields.io/badge/works-100%25%20offline-blueviolet.svg)]()

**Turn your meeting recordings into searchable, speaker-labeled transcripts — 100% locally, no cloud required.**

[Leia em Portugues](README.pt.md)

---

## What It Does

Meeting Transcriber listens to an audio recording of a meeting (or interview, lecture, voice message, podcast...) and produces a written document that shows **who said what, and when**. It identifies different speakers automatically and labels each part of the conversation.

Everything happens on your own computer. Your audio files are never uploaded anywhere.

**Here is what the output looks like:**

```
[00:00] SPEAKER_00: Good morning everyone, let's start the meeting.
[00:05] SPEAKER_01: Thanks for joining. First item on the agenda...
[00:12] SPEAKER_00: Before we begin, any updates from last week?
[00:18] SPEAKER_02: Yes, the client approved the proposal yesterday.
```

You get three output files: a plain text version (easy to read), a formatted Markdown version (looks nice in documents), and a JSON file (useful if you want to process the data further).

---

## What You Need

Before you start, make sure you have the following:

| What | Details |
|------|---------|
| **A Mac with Apple Silicon** | That means M1, M2, M3, or M4 chip. You can check by clicking the Apple menu and selecting "About This Mac." |
| **About 10 GB of free disk space** | The AI models that do the transcription are large files. They are downloaded once and stored on your computer. |
| **An internet connection** | Only needed the first time, to download the program and its models. After that, everything works offline. |

---

## Installation

This section walks you through setting up Meeting Transcriber step by step. Each step includes a way to verify it worked. If something goes wrong, check the [Troubleshooting](#troubleshooting) section.

### Step 1: Install Homebrew (if you don't have it)

Homebrew is a tool that makes it easy to install software on your Mac. Open the **Terminal** app (you can find it in Applications > Utilities, or search for "Terminal" with Spotlight) and paste this command:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**Verify it worked:**
```bash
brew --version
```
You should see a version number like `Homebrew 4.x.x`.

### Step 2: Install Python 3.12

Python is the programming language this tool is written in. You need version 3.12 specifically (newer versions like 3.14 are not compatible with some of the libraries we use).

```bash
brew install python@3.12
```

**Verify it worked:**
```bash
python3.12 --version
```
You should see `Python 3.12.x`.

### Step 3: Install FFmpeg

FFmpeg is a free tool that handles audio file conversion behind the scenes. Meeting Transcriber uses it to read different audio formats.

```bash
brew install ffmpeg
```

**Verify it worked:**
```bash
ffmpeg -version
```
You should see version information (the first line is enough).

### Step 4: Download Meeting Transcriber

This downloads the program to your computer:

```bash
git clone https://github.com/iurileao-hub/meeting-transcriber.git
cd meeting-transcriber
```

### Step 5: Create a virtual environment

A virtual environment is like a separate folder where the program and all its files live, without affecting the rest of your computer. This keeps things clean and avoids conflicts with other software.

```bash
python3.12 -m venv venv
source venv/bin/activate
```

After running the second command, you should see `(venv)` at the beginning of your terminal line. This means the virtual environment is active.

> **Important:** Every time you open a new Terminal window to use Meeting Transcriber, you need to activate the virtual environment again:
> ```bash
> cd meeting-transcriber
> source venv/bin/activate
> ```

### Step 6: Install the program's dependencies

Dependencies are the libraries and tools that Meeting Transcriber needs to work. This command downloads and installs all of them:

```bash
pip install -r requirements.txt
```

This may take a few minutes. You will see many lines scrolling by — that is normal.

**Verify it worked:**
```bash
python -c "import whisperx; print('OK')"
```
You should see `OK`.

### Step 7: Set up HuggingFace (needed for speaker identification)

HuggingFace is a website that hosts free AI models. Meeting Transcriber needs access to two of these models to tell speakers apart. This is completely free.

**7a. Create a free account:**
- Go to [huggingface.co/join](https://huggingface.co/join) and sign up.

**7b. Get your access token:**

A token is like a special password that lets the program download AI models from HuggingFace.

- Go to [Settings > Access Tokens](https://huggingface.co/settings/tokens)
- Click "New token", give it any name (like "meeting-transcriber"), and click Create.
- Copy the token (it starts with `hf_`).

**7c. Save the token:**
```bash
cp .env.example .env
```

Now open the `.env` file in any text editor and replace `hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx` with the token you just copied. Save the file.

**7d. Accept the AI model terms (one-time):**

You need to visit two pages and click "Agree and access repository" on each:

1. [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) -- click "Agree and access repository"
2. [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) -- click "Agree and access repository"

That's it! You only need to do this once.

---

## Your First Transcription

Let's make sure everything is working. Place an audio file (any `.mp3`, `.wav`, `.m4a`, or other [supported format](#supported-audio-formats)) inside the `data/audio/` folder, then run:

```bash
python src/transcribe.py data/audio/your-file.mp3
```

The program will show a progress bar as it works through several stages: loading the AI model, detecting speech, transcribing, and identifying speakers. Depending on the length of your audio and the model size, this can take from a few seconds to several minutes.

When it finishes, you will find your transcripts in the `data/transcripts/` folder:
- `your-file.txt` -- Plain text, easy to read
- `your-file.md` -- Formatted version, good for sharing
- `your-file.json` -- Structured data

Open the `.txt` file to see your transcription with speaker labels and timestamps.

### Supported Audio Formats

Meeting Transcriber works with these audio file types:

`.wav` `.mp3` `.m4a` `.flac` `.ogg` `.webm` `.aac` `.opus`

The `.m4a` format is what iPhones and Macs use for voice recordings. The `.opus` format is what WhatsApp uses for voice messages.

---

## Transcription Modes

Meeting Transcriber has three modes. Think of them as different approaches to the same job:

### Which mode should I use?

- **"I'm transcribing a meeting and need to know who said what"** -- Use **meeting** mode (this is the default, you don't need to add anything):
  ```bash
  python src/transcribe.py meeting.mp3
  ```

- **"I just need the text quickly, I don't care about speakers"** -- Use **fast** mode:
  ```bash
  python src/transcribe.py meeting.mp3 --mode fast
  ```

- **"I need the text quickly, but I also want speaker names"** -- Use **fast** mode with the `--diarize` flag:
  ```bash
  python src/transcribe.py meeting.mp3 --mode fast --diarize
  ```

- **"Accuracy is the top priority, and I don't mind waiting longer"** -- Use **precise** mode:
  ```bash
  python src/transcribe.py meeting.mp3 --mode precise
  ```

### Mode comparison

| | Meeting (default) | Fast | Precise |
|---|---|---|---|
| **Speed** | Moderate | Very fast (10-15x real time) | Slower |
| **Identifies speakers?** | Yes, always | Only with `--diarize` | Yes, always |
| **Accuracy** | Very good | Good | Best |
| **Memory needed** | ~10 GB | ~4 GB | ~16 GB |
| **Best for** | Most meetings | Quick drafts, single-speaker audio | Important recordings, legal/medical |

> **Note about precise mode:** It uses a large AI model (IBM Granite) that needs about 16 GB of RAM. If your Mac has 8 GB of memory, stick with **meeting** mode.

---

## Useful Options

Here are the options you will use most often:

### Tell it the language (improves accuracy)

```bash
python src/transcribe.py meeting.mp3 --language pt
```

Common language codes: `en` (English), `pt` (Portuguese), `es` (Spanish), `fr` (French), `de` (German).

### Tell it how many speakers are in the recording

If you know there were exactly 3 people in the meeting, telling the program helps it identify them more accurately:

```bash
python src/transcribe.py meeting.mp3 --num-speakers 3
```

### Get a notification when it's done

For long recordings, you can have your Mac notify you when the transcription is complete:

```bash
python src/transcribe.py meeting.mp3 --notify
```

### Choose what output files you want

By default, you get all three formats (txt, md, json). If you only want the plain text:

```bash
python src/transcribe.py meeting.mp3 --format txt
```

### See detailed progress information

If something seems wrong, verbose mode shows you everything the program is doing:

```bash
python src/transcribe.py meeting.mp3 --verbose
```

---

## Output Formats

Meeting Transcriber produces up to three files from each audio recording:

### Plain Text (.txt)

The simplest format. Easy to read, easy to search, works everywhere.

```
[00:00] SPEAKER_00: Good morning everyone, let's start the meeting.
[00:05] SPEAKER_01: Thanks for joining. First item on the agenda is
        the client proposal.
[00:12] SPEAKER_00: Before we begin, any updates from last week?
```

### Markdown (.md)

A formatted version that looks nice when opened in apps like Notion, Obsidian, or GitHub. Great for sharing.

```markdown
## Meeting Transcript

**[00:00] Speaker 1:** Good morning everyone, let's start the meeting.

**[00:05] Speaker 2:** Thanks for joining. First item on the agenda is
the client proposal.

**[00:12] Speaker 1:** Before we begin, any updates from last week?
```

### JSON (.json)

A structured format that contains all the details, including exact start and end times for each segment. Useful if you want to process the data with other tools or scripts.

```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 4.2,
      "text": "Good morning everyone, let's start the meeting.",
      "speaker": "SPEAKER_00"
    }
  ],
  "metadata": {
    "language": "en",
    "num_speakers": 3
  }
}
```

---

## Custom Vocabulary

If your meetings use specialized terms (medical terminology, legal jargon, company names, acronyms), you can teach the transcriber to recognize them. This is especially helpful for words that the AI might not know or might mishear.

**How to set it up:**

1. Open the file `vocab/default.txt` (or create it based on `vocab/default.txt.example`).
2. Add your terms, one per line. Lines starting with `#` are ignored.

```
# People
Dr. Martinez
Prof. Johnson

# Company names
Acme Corporation
NovaTech

# Acronyms and technical terms
HIPAA
laparoscopy
amortization
```

3. That's it! The program automatically loads `vocab/default.txt` every time it runs.

If you have different vocabulary files for different projects, you can specify which one to use:

```bash
python src/transcribe.py meeting.mp3 --vocab vocab/legal-terms.txt
```

---

## Using AI to Generate Meeting Minutes

Once you have a transcription, you can use an AI assistant (like Claude) to turn it into a polished document. Meeting Transcriber includes ready-to-use templates for the most common scenarios:

### Available Templates

| Template | What it produces | When to use it |
|----------|-----------------|----------------|
| [Meeting Minutes](examples/en/meeting_minutes.md) | Formal meeting minutes with agenda, decisions, and action items | After any team meeting |
| [Executive Summary](examples/en/executive_summary.md) | High-level summary with a structured action plan (5W2H methodology) | For leadership reports |
| [Action Items](examples/en/action_items.md) | A quick table of who needs to do what, by when | When you just need the to-do list |

### How to use them

1. **Transcribe your meeting:**
   ```bash
   python src/transcribe.py data/audio/team-meeting.mp3
   ```

2. **Open Claude Code** (or any AI assistant) and give it a prompt like:

   ```
   Read data/transcripts/team-meeting.txt and generate formal meeting
   minutes following the template in examples/en/meeting_minutes.md
   ```

3. The AI will read your transcript and produce a polished, organized document.

### Tips

- For the best results, specify the language and number of speakers when transcribing.
- You can customize the templates to match your organization's format -- just edit the files in `examples/en/`.
- If you create your own templates, save them in a `prompts/` folder (this folder is not shared when you update the program).

---

## Troubleshooting

### "No module named 'whisperx'" (or similar missing module errors)

Don't worry, this is easy to fix. It usually means the virtual environment is not active. Run these commands:

```bash
cd meeting-transcriber
source venv/bin/activate
pip install -r requirements.txt
```

The key thing to remember: every time you open a new Terminal window, you need to run `source venv/bin/activate` before using the program.

### The program runs out of memory

This means the AI model is too large for your available RAM. Try using a smaller model:

```bash
python src/transcribe.py meeting.mp3 --model small
```

The `small` model uses about 2 GB of RAM and still produces good results. You can also close other applications to free up memory.

### Speakers are not identified correctly

The AI does its best to tell speakers apart, but it is not perfect. You can help it by telling it how many speakers were in the recording:

```bash
python src/transcribe.py meeting.mp3 --num-speakers 3
```

If you know there were between 2 and 5 speakers but are not sure of the exact number:

```bash
python src/transcribe.py meeting.mp3 --min-speakers 2 --max-speakers 5
```

### The wrong language was detected

If the transcription comes out in the wrong language, tell the program which language to expect:

```bash
python src/transcribe.py meeting.mp3 --language en
```

### The transcription is taking too long

A few things you can try:

- Use `--mode fast` if you don't need speaker identification.
- Use `--model small` for a smaller, faster AI model.
- Close other applications to free up memory and processing power.
- For very long recordings (2+ hours), consider splitting the audio into smaller parts first.

### HuggingFace authentication error

This means the program cannot access the speaker identification models. Check these things:

1. Open the `.env` file and make sure your token is there (it should start with `hf_`).
2. Make sure you visited **both** of these pages and clicked "Agree and access repository":
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
3. Verify your token is still valid at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### "command not found: python" or "command not found: pip"

Make sure you activated the virtual environment first:

```bash
cd meeting-transcriber
source venv/bin/activate
```

If you still get the error, try using `python3.12` instead of `python`.

---

## Privacy & Security

**Your recordings and transcriptions never leave your computer.**

- All processing happens locally on your Mac. No audio is uploaded to any server.
- After the initial setup, no internet connection is needed.
- AI models are downloaded once and stored on your computer.
- There is no telemetry, analytics, or data collection of any kind.

This makes Meeting Transcriber ideal for:
- **Confidential business meetings** -- board discussions, strategy sessions
- **Medical consultations** -- patient interviews, clinical notes
- **Legal proceedings** -- depositions, client meetings
- **Personal recordings** -- interviews, lectures, voice memos

---

<details>
<summary><strong>Model Selection (advanced)</strong></summary>

A model is the AI brain that converts speech to text. Bigger models are more accurate but slower and use more memory. The default (`large-v3`) is the best choice for most people.

| Model | Accuracy | Speed | RAM Needed | When to use it |
|-------|----------|-------|------------|----------------|
| tiny | Low | Very fast | ~1 GB | Just testing if the program works |
| base | Medium | Fast | ~1 GB | Quick rough drafts |
| small | Good | Moderate | ~2 GB | Everyday use when speed matters |
| medium | Very good | Slow | ~5 GB | Important meetings where accuracy matters |
| **large-v3** | Excellent | Slower | ~10 GB | Best quality (this is the default) |

```bash
# Use a smaller model if you need speed or have limited memory
python src/transcribe.py meeting.mp3 --model small

# Explicitly use the largest model for important recordings
python src/transcribe.py meeting.mp3 --model large-v3
```

### Additional models for fast mode

When using `--mode fast`, you have access to a few extra models optimized for Apple Silicon:

| Model | Description |
|-------|-------------|
| large-v3-turbo | Best balance of speed and quality |
| distil-large-v3 | Faster, slightly less accurate |
| large-v3-8bit | Uses less memory |

```bash
python src/transcribe.py meeting.mp3 --mode fast --model large-v3-turbo
```

</details>

<details>
<summary><strong>Full Options Reference</strong></summary>

| Option | Short form | What it does | Default |
|--------|------------|--------------|---------|
| `--model` | `-m` | Choose the AI model size (see Model Selection above) | large-v3 |
| `--language` | `-l` | Set the audio language (en, pt, es, fr, de, etc.) | auto-detect |
| `--num-speakers` | `-n` | Tell it the exact number of speakers | auto-detect |
| `--min-speakers` | | Minimum number of speakers expected | -- |
| `--max-speakers` | | Maximum number of speakers expected | -- |
| `--output` | `-o` | Folder where transcripts are saved | data/transcripts |
| `--format` | `-f` | Output format: json, txt, md, or all | all |
| `--mode` | | Transcription approach: fast, meeting, or precise | meeting |
| `--device` | `-d` | Processor to use: cpu, cuda, or mps | auto |
| `--notify` | | Show a macOS notification when done | off |
| `--vocab` | | Path to a custom vocabulary file | -- |
| `--ui-lang` | | Interface language: en or pt | auto |
| `--diarize` | | Identify speakers in fast mode | off |
| `--verbose` | `-v` | Show detailed logs (useful for debugging) | off |

</details>

<details>
<summary><strong>Installing additional transcription backends</strong></summary>

The default installation includes the **meeting** mode backend. If you want to use the other modes:

**Fast mode** (uses MLX-Whisper, optimized for Apple Silicon):
```bash
pip install mlx-whisper
```

**Precise mode** (uses IBM Granite, highest accuracy):
```bash
pip install transformers accelerate
```

**Install everything at once:**
```bash
pip install mlx-whisper transformers accelerate
```

> **Note:** Precise mode requires about 16 GB of RAM. If your Mac has 8 GB of memory, the meeting and fast modes will serve you well.

</details>

---

## Acknowledgments

Built with these excellent open-source projects:

- [WhisperX](https://github.com/m-bain/whisperX) -- Speech recognition with word-level timestamps
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) -- Optimized Whisper inference
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) -- Speaker diarization
- [MLX](https://github.com/ml-explore/mlx) -- Apple Silicon machine learning framework
- [IBM Granite Speech](https://huggingface.co/ibm-granite) -- High-accuracy speech recognition

---

## License

MIT License -- free for personal and commercial use.

---

## Author

**Iuri Almeida**

*February 2026*
