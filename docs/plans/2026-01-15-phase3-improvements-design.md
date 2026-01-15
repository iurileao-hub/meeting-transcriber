# Phase 3: Performance, Quality & UX Improvements

> Design document for Meeting Transcriber enhancements

**Date:** January 15, 2026
**Status:** Approved
**Author:** Iuri Almeida + Claude Code

---

## Overview

This document describes the Phase 3 improvements for the Meeting Transcriber project, including:

1. **Multiple transcription backends** via `--mode` flag
2. **Progress reporting** with stage indicators
3. **System notifications** via `--notify` flag
4. **Custom vocabulary** support
5. **Basic text normalization**
6. **Internationalization (i18n)** - English/Portuguese

---

## 1. Transcription Backends

### Modes

| Mode | Backend | Diarization | Use Case |
|------|---------|-------------|----------|
| `fast` | mlx-whisper | No | Quick transcription, single speaker |
| `meeting` | whisperX | Yes | Meetings with multiple speakers (default) |
| `precise` | Granite + pyannote | Yes | High accuracy, technical vocabulary |

### Architecture

```
src/backends/
├── __init__.py          # get_backend(mode) factory
├── base.py              # TranscriptionBackend ABC
├── mlx_backend.py       # --mode fast
├── whisperx_backend.py  # --mode meeting
└── granite_backend.py   # --mode precise
```

### Base Interface

```python
class TranscriptionBackend(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """Returns segments with text, timestamps, and speakers (if available)."""
        pass

    @property
    @abstractmethod
    def supports_diarization(self) -> bool:
        pass
```

### Factory Pattern

```python
def get_backend(mode: str) -> TranscriptionBackend:
    backends = {
        "fast": MLXBackend,
        "meeting": WhisperXBackend,
        "precise": GraniteBackend,
    }
    return backends[mode]()
```

---

## 2. Progress Reporting

### Stage-based Progress Bar

```
[2/4] Transcribing... [████████░░░░] 65%
```

### Implementation

```python
class Stage(Enum):
    LOADING = ("Loading model", "Carregando modelo")
    TRANSCRIBING = ("Transcribing", "Transcrevendo")
    ALIGNING = ("Aligning", "Alinhando")
    DIARIZING = ("Diarizing", "Diarizando")
    SAVING = ("Saving", "Salvando")

class ProgressReporter:
    def update(self, stage: Stage, percent: float):
        """Updates progress display."""
        pass
```

### Completion Message

```
✓ Transcription complete: data/transcripts/reuniao.json (4m32s)
```

---

## 3. System Notifications

### Optional Flag

```bash
python src/transcribe.py audio.wav --notify
```

### macOS Implementation

```python
def notify(title: str, message: str):
    if platform.system() == "Darwin":
        subprocess.run([
            "osascript", "-e",
            f'display notification "{message}" with title "{title}"'
        ])
```

---

## 4. Custom Vocabulary

### File Structure

```
vocab/
├── default.txt      # Auto-loaded if exists (gitignored)
└── default.txt.example  # Template for repo
```

### File Format

```
# One word/term per line
# Comments start with #
PMDF
laparoscopia
Dr. Silva
```

### Loading Logic

- `vocab/default.txt` loads automatically if present
- `--vocab file.txt` adds additional vocabulary files
- Duplicates are removed automatically

---

## 5. Text Normalization

### Scope

Basic normalization only (preserve original transcription):

- Capitalize after sentence endings (. ! ?)
- Capitalize first character of text
- No word substitutions or corrections

### Implementation

```python
def normalize_text(text: str) -> str:
    text = re.sub(r'([.!?])\s*(\w)',
                  lambda m: m.group(1) + ' ' + m.group(2).upper(), text)
    if text:
        text = text[0].upper() + text[1:]
    return text
```

---

## 6. Internationalization (i18n)

### Supported Languages

- English (en) - Primary
- Portuguese (pt)

### File Structure

```
src/i18n/
├── __init__.py    # get_translator(lang)
├── en.json        # English strings
└── pt.json        # Portuguese strings
```

### Auto-detection

System language is detected automatically. Override with `--ui-lang`.

### Translation Keys

```json
{
  "stages": { "loading": "...", "transcribing": "..." },
  "messages": { "complete": "...", "error": "..." },
  "cli": { "help_mode": "...", "help_notify": "..." }
}
```

---

## 7. Updated CLI

### New Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--mode` | Transcription mode (fast/meeting/precise) | meeting |
| `--ui-lang` | Interface language (en/pt) | auto |
| `--notify` | Send system notification when done | false |
| `--vocab` | Additional vocabulary file(s) | none |

### Usage Examples

```bash
# Quick transcription (single speaker)
python src/transcribe.py audio.wav --mode fast

# Meeting with notification (default mode)
python src/transcribe.py audio.wav --notify

# High precision with custom vocabulary
python src/transcribe.py audio.wav --mode precise --vocab vocab/medical.txt

# Portuguese interface
python src/transcribe.py audio.wav --ui-lang pt
```

---

## 8. Project Structure

```
meeting-transcriber/
├── README.md                    # English documentation
├── README.pt.md                 # Portuguese documentation
├── src/
│   ├── transcribe.py            # Main CLI
│   ├── backends/                # Transcription backends
│   ├── i18n/                    # Internationalization
│   ├── progress.py              # Progress reporting
│   ├── notify.py                # System notifications
│   ├── vocabulary.py            # Vocabulary loading
│   └── normalize.py             # Text normalization
├── vocab/
│   ├── default.txt              # User vocabulary (gitignored)
│   └── default.txt.example      # Template
├── prompts/                     # Claude Code prompts
├── data/                        # Audio/transcripts/outputs
└── tests/                       # Unit tests
```

---

## 9. New Dependencies

```txt
# Mode fast
mlx-whisper>=0.4.0

# Mode precise
transformers>=4.40.0
accelerate>=0.30.0
```

---

## 10. Speaker Mapping

Speaker mapping (SPEAKER_00 → "Dr. João") is handled in post-processing with Claude Code, not during transcription. This keeps the transcription pipeline simple and leverages Claude's context understanding.

---

## Next Steps

1. Create backend architecture (`src/backends/`)
2. Extract current whisperX logic to `whisperx_backend.py`
3. Implement `mlx_backend.py`
4. Implement `granite_backend.py`
5. Add i18n support
6. Add progress reporting
7. Add notification support
8. Update CLI with new flags
9. Write tests
10. Update documentation (bilingual)

---

*Document generated during brainstorming session with Claude Code*
