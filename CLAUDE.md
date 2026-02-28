# CLAUDE.md

Instrucoes para Claude Code ao trabalhar neste repositorio.

## Proposito

Sistema local de transcricao de reunioes com identificacao de speakers. Processa audio em texto com timestamps e labels de speaker, 100% local.

## Stack

- **Python 3.12** (3.14 incompativel com dependencias)
- **whisperX 3.7.4** — Transcricao + diarizacao (modo meeting)
- **faster-whisper 1.2.1** — Backend do whisperX
- **mlx-whisper** — Apple Silicon (modo fast)
- **IBM Granite Speech** — Alta precisao via transformers (modo precise)
- **pyannote.audio 3.4.0** — Speaker diarization
- **torch 2.8.0** — Framework ML

## Estrutura do Projeto

```
src/
  transcribe.py            # CLI principal + _save_results()
  backends/
    __init__.py            # Factory: get_backend(mode) com lazy imports
    base.py                # TranscriptionBackend ABC
    whisperx_backend.py    # Modo meeting (WhisperX + pyannote)
    mlx_backend.py         # Modo fast (MLX-Whisper)
    granite_backend.py     # Modo precise (Granite + pyannote)
  i18n/                    # get_translator(lang), en.json, pt.json
  progress.py              # ProgressReporter (spinner, timer, ETA, throttle 100ms)
  notify.py                # Notificacoes macOS via AppleScript
  vocabulary.py            # Vocabulario customizado (allowlist, limite 1MB)
  normalize.py             # Normalizacao de texto
examples/
  en/                      # Templates de prompts em ingles
  pt/                      # Templates de prompts em portugues
tests/                     # 289 testes unitarios
docs/
  ARCHITECTURE.md          # Decisoes tecnicas e patterns
  CHANGELOG.md             # Historico de mudancas
  DEVELOPMENT_HISTORY.md   # Historico de desenvolvimento
```

## Convencoes de Codigo

- **Idioma do codigo:** Ingles (variaveis, funcoes, docstrings)
- **Idioma da documentacao:** Portugues
- **Formatacao:** Black (88 colunas)
- **Type hints:** Sempre usar
- **Docstrings:** Google style

## Desenvolvimento

```bash
source venv/bin/activate           # Ativar ambiente virtual
pytest tests/ -v                   # Rodar todos os testes
pytest tests/test_progress.py -v   # Rodar testes de um modulo
pytest tests/ -v -k "test_name"   # Rodar teste especifico
python src/transcribe.py audio.wav --model small  # Teste rapido
```

## Configuracao

1. **Token HuggingFace** em `.env`: `HF_TOKEN=hf_xxxxxxxxxxxxx`
2. **Aceitar termos pyannote:** [speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) e [segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

## Padroes Arquiteturais

- **Factory pattern:** `get_backend(mode)` em `backends/__init__.py` com lazy imports e dict estatico de parametros
- **ABC base class:** `TranscriptionBackend` define `transcribe()`, `is_available()`, `total_stages`; `_load_hf_token()` e `_build_diarize_kwargs()` centralizados na base
- **Progress:** Monkey-patch de `pyannote.Inference.slide` para VAD/diarizacao; interceptacao de tqdm para MLX; `ProgressStreamer(BaseStreamer)` para Granite; rendering com throttle de 100ms
- **Seguranca:** `_allow_legacy_torch_load()` context manager (nao global); allowlist para vocab paths; `FORBIDDEN_OUTPUT_PATHS` para output; sanitizacao de AppleScript
- **HF Hub compat:** `patch_hf_hub_compat()` com `threading.Lock` para `use_auth_token` -> `token`

## Notas para Desenvolvimento

- Usar modelo `small` para testes rapidos, `large-v3` para producao
- Device auto-detectado: MPS (Apple Silicon), CUDA (NVIDIA), CPU (fallback)
- Nao versionar: audio, transcricoes, modelos baixados
- Testar com audios curtos (< 5min) durante desenvolvimento
- FFmpeg necessario: `brew install ffmpeg`
- Backends extras: `pip install mlx-whisper transformers accelerate`

## Problemas Conhecidos

1. **PyTorch weights_only:** Context manager `_allow_legacy_torch_load()` contorna mudanca de seguranca durante carregamento de modelos pyannote
2. **Warnings suprimidos:** Warnings de torchaudio/pyannote filtrados por padrao (`--verbose` para ver)
