# Meeting Transcriber

[English](#english) | [Português](#português)

---

## English

Local meeting transcription system with speaker identification (diarization).

### Features

- Audio transcription in Portuguese and English
- Automatic speaker identification (diarization)
- Word-level timestamps with confidence scores
- Multiple output formats (JSON, TXT, Markdown)
- 100% local processing (no cloud APIs)
- **[Phase 3]** Multiple transcription backends (MLX-Whisper, WhisperX, Granite)
- **[Phase 3]** Bilingual interface (English/Portuguese)
- **[Phase 3]** Progress bar with time estimates
- **[Phase 3]** macOS notifications on completion
- **[Phase 3]** Custom vocabulary support

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
# Basic transcription (generates .json, .txt, and .md)
python src/transcribe.py data/audio/meeting.wav

# With options
python src/transcribe.py meeting.mp3 --model medium --language en

# Specify number of speakers
python src/transcribe.py meeting.wav --num-speakers 4

# Choose output format
python src/transcribe.py meeting.wav --format txt

# Show help
python src/transcribe.py --help
```

### Command-Line Options

| Flag | Short | Description | Default |
|------|-------|-------------|---------|
| `--model` | `-m` | Whisper model size (tiny/base/small/medium/large-v3) | large-v3 |
| `--language` | `-l` | Language code (en, pt, etc.) or auto-detect | auto |
| `--num-speakers` | `-n` | Exact number of speakers (if known) | auto |
| `--min-speakers` | | Minimum expected speakers | - |
| `--max-speakers` | | Maximum expected speakers | - |
| `--output` | `-o` | Output directory | data/transcripts |
| `--format` | `-f` | Output format (json/txt/md/all) | all |
| `--device` | `-d` | Processing device (cpu/cuda/mps) | cpu |
| `--verbose` | `-v` | Show detailed logs and warnings | false |
| `--mode` | | Transcription mode (fast/meeting/precise) | meeting |
| `--ui-lang` | | Interface language (en/pt) | auto |
| `--notify` | | Send macOS notification on completion | false |
| `--vocab` | | Path to custom vocabulary file | - |

### Transcription Modes

| Mode | Backend | Diarization | Speed | Use Case |
|------|---------|-------------|-------|----------|
| `fast` | MLX-Whisper | No | 10-15x realtime | Quick transcription, Apple Silicon |
| `meeting` | WhisperX | Yes | Moderate | Default, meetings with multiple speakers |
| `precise` | Granite + pyannote | Yes | Slower | High accuracy, important recordings |

```bash
# Fast mode - quick transcription without speaker identification
python src/transcribe.py audio.wav --mode fast

# Meeting mode (default) - full diarization
python src/transcribe.py audio.wav --mode meeting

# Precise mode - highest accuracy
python src/transcribe.py audio.wav --mode precise
```

### Custom Vocabulary

Create `vocab/default.txt` with domain-specific terms (auto-loaded if exists):

```
# Company names
ACME Corp
TechStartup Inc

# Technical terms
API
SDK
Kubernetes
```

Or specify a custom file:
```bash
python src/transcribe.py audio.wav --vocab custom_terms.txt
```

### Notifications

Enable macOS notifications for long transcriptions:
```bash
python src/transcribe.py long_meeting.wav --notify
```

### Output Formats

| Format | Description | Use Case |
|--------|-------------|----------|
| `json` | Complete with word-level timestamps | Processing by code/Claude |
| `txt` | Simple text with speaker labels | Quick reading |
| `md` | Formatted Markdown | Review/editing |
| `all` | All formats above (default) | Maximum flexibility |

**Example output (.txt):**
```
[00:00] SPEAKER_00: Good morning everyone, let's start the meeting.

[00:05] SPEAKER_01: Thanks for joining. First item on the agenda...
```

### JSON Structure

The JSON file contains detailed data for programmatic processing:

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
    "model": "large-v3",
    "num_speakers": 3
  }
}
```

### Integration with Claude Code

After transcription, use the prompts in `prompts/` to generate documents:

```bash
# 1. Transcribe audio
python src/transcribe.py data/audio/meeting.wav

# 2. In Claude Code, use one of the prompts:
```

**Available prompts:**

| Prompt | Description |
|--------|-------------|
| `prompts/ata_sei.md` | Formal meeting minutes for SEI! system |
| `prompts/resumo_executivo.md` | Executive summary + action plan (5W2H) |

**Example:**
```
Read data/transcripts/meeting.txt and generate a formal MEETING MINUTES
following the Brazilian institutional standard for SEI!.

[Paste the rest of the prompt from prompts/ata_sei.md]
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
- Múltiplos formatos de saída (JSON, TXT, Markdown)
- Processamento 100% local (sem APIs na nuvem)
- **[Fase 3]** Múltiplos backends de transcrição (MLX-Whisper, WhisperX, Granite)
- **[Fase 3]** Interface bilíngue (Inglês/Português)
- **[Fase 3]** Barra de progresso com estimativa de tempo
- **[Fase 3]** Notificações macOS ao concluir
- **[Fase 3]** Suporte a vocabulário customizado

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
# Transcrição básica (gera .json, .txt e .md)
python src/transcribe.py data/audio/reuniao.wav

# Com opções
python src/transcribe.py reuniao.mp3 --model medium --language pt

# Especificar número de speakers
python src/transcribe.py reuniao.wav --num-speakers 4

# Escolher formato de saída
python src/transcribe.py reuniao.wav --format txt

# Ver ajuda
python src/transcribe.py --help
```

### Opções de Linha de Comando

| Flag | Curto | Descrição | Padrão |
|------|-------|-----------|--------|
| `--model` | `-m` | Tamanho do modelo Whisper (tiny/base/small/medium/large-v3) | large-v3 |
| `--language` | `-l` | Código do idioma (en, pt, etc.) ou auto-detectar | auto |
| `--num-speakers` | `-n` | Número exato de speakers (se conhecido) | auto |
| `--min-speakers` | | Mínimo de speakers esperado | - |
| `--max-speakers` | | Máximo de speakers esperado | - |
| `--output` | `-o` | Diretório de saída | data/transcripts |
| `--format` | `-f` | Formato de saída (json/txt/md/all) | all |
| `--device` | `-d` | Dispositivo de processamento (cpu/cuda/mps) | cpu |
| `--verbose` | `-v` | Mostra logs e warnings detalhados | false |
| `--mode` | | Modo de transcrição (fast/meeting/precise) | meeting |
| `--ui-lang` | | Idioma da interface (en/pt) | auto |
| `--notify` | | Enviar notificação macOS ao concluir | false |
| `--vocab` | | Caminho para arquivo de vocabulário | - |

### Modos de Transcrição

| Modo | Backend | Diarização | Velocidade | Uso |
|------|---------|------------|------------|-----|
| `fast` | MLX-Whisper | Não | 10-15x tempo real | Transcrição rápida, Apple Silicon |
| `meeting` | WhisperX | Sim | Moderado | Padrão, reuniões com múltiplos speakers |
| `precise` | Granite + pyannote | Sim | Mais lento | Alta precisão, gravações importantes |

```bash
# Modo fast - transcrição rápida sem identificação de speakers
python src/transcribe.py audio.wav --mode fast

# Modo meeting (padrão) - diarização completa
python src/transcribe.py audio.wav --mode meeting

# Modo precise - máxima precisão
python src/transcribe.py audio.wav --mode precise
```

### Vocabulário Customizado

Crie `vocab/default.txt` com termos específicos do domínio (carregado automaticamente se existir):

```
# Nomes de empresas
PMDF
FIAP

# Termos técnicos
API
SDK
Kubernetes
```

Ou especifique um arquivo customizado:
```bash
python src/transcribe.py audio.wav --vocab termos_custom.txt
```

### Notificações

Habilite notificações macOS para transcrições longas:
```bash
python src/transcribe.py reuniao_longa.wav --notify
```

### Formatos de Saída

| Formato | Descrição | Uso |
|---------|-----------|-----|
| `json` | Completo com timestamps por palavra | Processamento por código/Claude |
| `txt` | Texto simples com speakers | Leitura rápida |
| `md` | Markdown formatado | Revisão/edição |
| `all` | Todos os formatos acima (padrão) | Máxima flexibilidade |

**Exemplo de saída (.txt):**
```
[00:00] SPEAKER_00: Bom dia a todos, vamos começar a reunião.

[00:05] SPEAKER_01: Obrigado pela presença. Primeiro item da pauta...
```

### Estrutura do JSON

O arquivo JSON contém dados detalhados para processamento programático:

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
    "model": "large-v3",
    "num_speakers": 3
  }
}
```

### Integração com Claude Code

Após a transcrição, use os prompts em `prompts/` para gerar documentos:

```bash
# 1. Transcrever áudio
python src/transcribe.py data/audio/reuniao.wav

# 2. No Claude Code, use um dos prompts:
```

**Prompts disponíveis:**

| Prompt | Descrição |
|--------|-----------|
| `prompts/ata_sei.md` | Ata formal para sistema SEI! |
| `prompts/resumo_executivo.md` | Resumo executivo + plano de ação (5W2H) |

**Exemplo:**
```
Leia data/transcripts/reuniao.txt e gere uma ATA DE REUNIÃO formal
seguindo o padrão institucional brasileiro para o SEI!.

[Cole o restante do prompt de prompts/ata_sei.md]
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

### Want to see detailed logs?
Use verbose mode to debug issues:
```bash
python src/transcribe.py audio.wav --verbose
```

---

## License

Educational project for personal use.

---

*Author: Iuri Almeida*
*January 2026*
