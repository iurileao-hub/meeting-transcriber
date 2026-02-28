# CLAUDE.md

Instruções para Claude Code ao trabalhar neste repositório.

## Propósito

Sistema local de transcrição de reuniões com identificação de speakers. Processa áudio em texto com timestamps e labels de speaker, 100% local.

## Stack

- **Python 3.12** (3.14 incompatível com dependências)
- **whisperX 3.7.4** — Transcrição + diarização (modo meeting)
- **faster-whisper 1.2.1** — Backend do whisperX
- **mlx-whisper** — Apple Silicon (modo fast)
- **IBM Granite Speech** — Alta precisão via transformers (modo precise)
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
  notify.py                # Notificações macOS via AppleScript
  vocabulary.py            # Vocabulário customizado (allowlist, limite 1MB)
  normalize.py             # Normalização de texto
examples/
  en/                      # Templates de prompts em inglês
  pt/                      # Templates de prompts em português
tests/                     # 289 testes unitários
docs/
  ARCHITECTURE.md          # Decisões técnicas e patterns
  CHANGELOG.md             # Histórico de mudanças
  DEVELOPMENT_HISTORY.md   # Histórico de desenvolvimento
```

## Convenções de Código

- **Idioma do código:** Inglês (variáveis, funções, docstrings)
- **Idioma da documentação:** Português
- **Formatação:** Black (88 colunas)
- **Type hints:** Sempre usar
- **Docstrings:** Google style

## Desenvolvimento

```bash
source venv/bin/activate           # Ativar ambiente virtual
pytest tests/ -v                   # Rodar todos os testes
pytest tests/test_progress.py -v   # Rodar testes de um módulo
pytest tests/ -v -k "test_name"   # Rodar teste específico
python src/transcribe.py audio.wav --model small  # Teste rápido
```

## Configuração

1. **Token HuggingFace** em `.env`: `HF_TOKEN=hf_xxxxxxxxxxxxx`
2. **Aceitar termos pyannote:** [speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) e [segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

## Padrões Arquiteturais

- **Factory pattern:** `get_backend(mode)` em `backends/__init__.py` com lazy imports e dict estático de parâmetros
- **ABC base class:** `TranscriptionBackend` define `transcribe()`, `is_available()`, `total_stages`; `_load_hf_token()` e `_build_diarize_kwargs()` centralizados na base
- **Progress:** Monkey-patch de `pyannote.Inference.slide` para VAD/diarização; interceptação de tqdm para MLX; `ProgressStreamer(BaseStreamer)` para Granite; rendering com throttle de 100ms
- **Segurança:** `_allow_legacy_torch_load()` context manager (não global); allowlist para vocab paths; `FORBIDDEN_OUTPUT_PATHS` para output; sanitização de AppleScript
- **HF Hub compat:** `patch_hf_hub_compat()` com `threading.Lock` para `use_auth_token` -> `token`

## Notas para Desenvolvimento

- Usar modelo `small` para testes rápidos, `large-v3` para produção
- Device auto-detectado: MPS (Apple Silicon), CUDA (NVIDIA), CPU (fallback)
- Não versionar: áudio, transcrições, modelos baixados
- Testar com áudios curtos (< 5min) durante desenvolvimento
- FFmpeg necessário: `brew install ffmpeg`
- Backends extras: `pip install mlx-whisper transformers accelerate`

## Problemas Conhecidos

1. **PyTorch weights_only:** Context manager `_allow_legacy_torch_load()` contorna mudança de segurança durante carregamento de modelos pyannote
2. **Warnings suprimidos:** Warnings de torchaudio/pyannote filtrados por padrão (`--verbose` para ver)
