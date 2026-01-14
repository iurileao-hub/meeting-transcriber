# Meeting Transcriber

[English](#english) | [Português](#português)

---

## English

Local meeting transcription system with speaker identification (diarization).

### Features

- Audio transcription in Portuguese and English
- Automatic speaker identification (diarization)
- Word-level timestamps with confidence scores
- Structured JSON output
- 100% local processing (no cloud APIs)

### Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.12 (3.14 not supported yet)
- FFmpeg
- HuggingFace account (free)

### Installation

```bash
# 1. Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# 2. Install FFmpeg
brew install ffmpeg

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure HuggingFace
cp .env.example .env
# Edit .env and add your HF_TOKEN

# 5. Accept pyannote terms (required for diarization)
# Visit: https://huggingface.co/pyannote/speaker-diarization-3.1
# Visit: https://huggingface.co/pyannote/segmentation-3.0
# Click "Agree and access repository" on both
```

### Usage

```bash
# Basic transcription
python src/transcribe.py data/audio/meeting.wav

# With options
python src/transcribe.py meeting.mp3 --model medium --language en

# Specify number of speakers
python src/transcribe.py meeting.wav --num-speakers 4

# Show help
python src/transcribe.py --help
```

### Output

The script generates a JSON file in `data/transcripts/` with this structure:

```json
{
  "segments": [
    {
      "start": 0.5,
      "end": 5.2,
      "text": "Good morning everyone, let's start the meeting.",
      "speaker": "SPEAKER_00",
      "words": [
        {"word": "Good", "start": 0.5, "end": 0.7, "score": 0.95, "speaker": "SPEAKER_00"}
      ]
    }
  ],
  "metadata": {
    "source_file": "meeting.wav",
    "language": "en",
    "num_speakers": 3
  }
}
```

### Integration with Claude Code

After transcription, use Claude Code to generate meeting minutes:

```
Read the file data/transcripts/meeting.json and generate:
1. Meeting minutes (participants, agenda, decisions)
2. Action items with owners
3. Next steps
```

### Available Models

| Model | Accuracy | Speed | RAM |
|-------|----------|-------|-----|
| tiny | Low | Very fast | 1GB |
| base | Medium | Fast | 1GB |
| small | Good | Moderate | 2GB |
| medium | Very good | Slow | 5GB |
| large-v3 | Excellent | Slower | 10GB |

**Recommendation:** Start with `small` for testing, use `large-v3` for production.

---

## Português

Sistema local de transcrição de reuniões com identificação de speakers (diarização).

### Funcionalidades

- Transcrição de áudio em português e inglês
- Identificação automática de speakers (diarização)
- Timestamps a nível de palavra com scores de confiança
- Output em JSON estruturado
- Processamento 100% local (sem APIs na nuvem)

### Requisitos

- macOS com Apple Silicon (M1/M2/M3/M4)
- Python 3.12 (3.14 ainda não suportado)
- FFmpeg
- Conta HuggingFace (gratuita)

### Instalação

```bash
# 1. Criar ambiente virtual
python3.12 -m venv venv
source venv/bin/activate

# 2. Instalar FFmpeg
brew install ffmpeg

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Configurar HuggingFace
cp .env.example .env
# Editar .env e adicionar seu HF_TOKEN

# 5. Aceitar termos do pyannote (necessário para diarização)
# Acessar: https://huggingface.co/pyannote/speaker-diarization-3.1
# Acessar: https://huggingface.co/pyannote/segmentation-3.0
# Clicar em "Agree and access repository" em ambos
```

### Uso

```bash
# Transcrição básica
python src/transcribe.py data/audio/reuniao.wav

# Com opções
python src/transcribe.py reuniao.mp3 --model medium --language pt

# Especificar número de speakers
python src/transcribe.py reuniao.wav --num-speakers 4

# Ver ajuda
python src/transcribe.py --help
```

### Output

O script gera um arquivo JSON em `data/transcripts/` com esta estrutura:

```json
{
  "segments": [
    {
      "start": 0.5,
      "end": 5.2,
      "text": "Bom dia a todos, vamos começar a reunião.",
      "speaker": "SPEAKER_00",
      "words": [
        {"word": "Bom", "start": 0.5, "end": 0.7, "score": 0.95, "speaker": "SPEAKER_00"}
      ]
    }
  ],
  "metadata": {
    "source_file": "reuniao.wav",
    "language": "pt",
    "num_speakers": 3
  }
}
```

### Integração com Claude Code

Após a transcrição, use Claude Code para gerar atas:

```
Leia o arquivo data/transcripts/reuniao.json e gere:
1. Ata da reunião (participantes, pauta, decisões)
2. Action items com responsáveis
3. Próximos passos
```

### Modelos Disponíveis

| Modelo | Precisão | Velocidade | RAM |
|--------|----------|------------|-----|
| tiny | Baixa | Muito rápido | 1GB |
| base | Média | Rápido | 1GB |
| small | Boa | Moderado | 2GB |
| medium | Muito boa | Lento | 5GB |
| large-v3 | Excelente | Mais lento | 10GB |

**Recomendação:** Começar com `small` para testes, usar `large-v3` para produção.

---

## Troubleshooting

### Error: "No module named 'whisperx'"
```bash
source venv/bin/activate  # Activate virtual environment
pip install whisperx
```

### Memory error
Use a smaller model:
```bash
python src/transcribe.py audio.wav --model small
```

### Incorrect diarization
Specify the number of speakers:
```bash
python src/transcribe.py audio.wav --num-speakers 3
```

---

## License

Educational project for personal use.

---

*Author: Iuri Almeida*
*January 2026*
