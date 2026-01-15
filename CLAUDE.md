# CLAUDE.md

Instruções para Claude Code ao trabalhar neste repositório.

---

## Propósito do Projeto

Sistema local de transcrição de reuniões com identificação de speakers. Desenvolvido como projeto educacional para aprendizado de ASR (Automatic Speech Recognition) e processamento de áudio.

**Contexto:** Projeto pessoal de Iuri Almeida (médico, gestor PMDF, estudante de Ciência da Computação - FIAP 2026).

---

## Stack Tecnológica

- **Python 3.12** (3.14 incompatível com dependências)
- **whisperX 3.7.4** — Transcrição + diarização integrada (modo meeting)
- **faster-whisper 1.2.1** — Backend de transcrição (usado pelo whisperX)
- **mlx-whisper** — Backend otimizado para Apple Silicon (modo fast)
- **granite-speech** — Backend de alta precisão (modo precise)
- **pyannote.audio 3.4.0** — Speaker diarization
- **torch 2.8.0** — Framework de ML

---

## Estrutura do Projeto

```
meeting-transcriber/
├── CLAUDE.md             # Este arquivo
├── PLAN.md               # Roadmap de desenvolvimento
├── README.md             # Documentação de uso (bilíngue)
├── requirements.txt      # Dependências Python
├── pytest.ini            # Configuração de testes
├── .env                  # Configuração (não versionado)
├── .env.example          # Template de configuração
├── src/
│   ├── transcribe.py     # CLI principal
│   ├── backends/         # Backends de transcrição
│   │   ├── __init__.py   # Factory: get_backend(mode)
│   │   ├── base.py       # TranscriptionBackend ABC
│   │   ├── whisperx_backend.py   # Modo meeting
│   │   ├── mlx_backend.py        # Modo fast
│   │   └── granite_backend.py    # Modo precise
│   ├── i18n/             # Internacionalização
│   │   ├── __init__.py   # get_translator(lang)
│   │   ├── en.json       # Strings em inglês
│   │   └── pt.json       # Strings em português
│   ├── progress.py       # Barra de progresso
│   ├── notify.py         # Notificações macOS
│   ├── vocabulary.py     # Vocabulário customizado
│   └── normalize.py      # Normalização de texto
├── prompts/              # Prompts para Claude Code
│   ├── README.md         # Instruções de uso
│   ├── ata_sei.md        # Ata formal (SEI!)
│   └── resumo_executivo.md  # Resumo + Plano de trabalho
├── vocab/                # Vocabulário
│   ├── default.txt       # Termos padrão (não versionado)
│   └── default.txt.example  # Template
├── data/
│   ├── audio/            # Arquivos de entrada (.wav, .mp3)
│   ├── transcripts/      # Saídas (.json, .txt, .md)
│   └── outputs/          # Atas e documentos
└── tests/                # 95+ testes unitários
    ├── test_transcribe.py
    ├── test_backends.py
    ├── test_whisperx_backend.py
    ├── test_mlx_backend.py
    ├── test_granite_backend.py
    ├── test_i18n.py
    ├── test_progress.py
    ├── test_notify.py
    ├── test_vocabulary.py
    └── test_normalize.py
```

---

## Comandos Comuns

```bash
# Ativar ambiente virtual
source venv/bin/activate

# Transcrever áudio (gera .json, .txt e .md)
python src/transcribe.py data/audio/reuniao.wav

# Com opções
python src/transcribe.py audio.wav --model medium --language pt --num-speakers 4

# Apenas texto simples (leitura rápida)
python src/transcribe.py audio.wav --format txt

# Com logs detalhados (debug)
python src/transcribe.py audio.wav --verbose

# Transcrição rápida (sem diarização)
python src/transcribe.py audio.wav --mode fast

# Transcrição de alta precisão
python src/transcribe.py audio.wav --mode precise

# Com notificação ao finalizar
python src/transcribe.py audio.wav --notify

# Com vocabulário customizado
python src/transcribe.py audio.wav --vocab vocab/termos_tecnicos.txt

# Interface em português
python src/transcribe.py audio.wav --ui-lang pt

# Ver ajuda
python src/transcribe.py --help

# Rodar testes
pytest tests/ -v
```

---

## Flags de Execução

| Flag | Descrição | Valores | Padrão |
|------|-----------|---------|--------|
| `--model`, `-m` | Modelo Whisper | tiny, base, small, medium, large-v3 | large-v3 |
| `--language`, `-l` | Idioma | pt, en, es, etc. | auto |
| `--num-speakers`, `-n` | Nº exato de speakers | inteiro | auto |
| `--format`, `-f` | Formato de saída | json, txt, md, all | all |
| `--device`, `-d` | Dispositivo | cpu, cuda, mps | cpu |
| `--verbose`, `-v` | Logs detalhados | flag | false |
| `--output`, `-o` | Diretório de saída | path | data/transcripts |
| `--mode` | Modo de transcrição | fast, meeting, precise | meeting |
| `--ui-lang` | Idioma da interface | en, pt | auto |
| `--notify` | Notificação macOS | flag | false |
| `--vocab` | Arquivo de vocabulário | path | - |

### Modos de Transcrição

| Modo | Backend | Diarização | Velocidade | Uso |
|------|---------|------------|------------|-----|
| `fast` | MLX-Whisper | Não | 10-15x tempo real | Transcrição rápida |
| `meeting` | WhisperX | Sim | Moderado | Reuniões (padrão) |
| `precise` | Granite + pyannote | Sim | Mais lento | Alta precisão |

---

## Workflow com Claude Code

Após transcrição, usar os prompts padronizados em `prompts/`:

```bash
# 1. Transcrever (gera .json, .txt, .md)
python src/transcribe.py data/audio/reuniao.wav

# 2. Usar prompt adequado no Claude Code
```

**Prompts disponíveis:**

| Arquivo | Uso |
|---------|-----|
| `prompts/ata_sei.md` | Ata formal para SEI! (padrão institucional) |
| `prompts/resumo_executivo.md` | Resumo + Plano de trabalho (5W2H/SMART) |

**Exemplo de uso:**
```
Leia data/transcripts/reuniao.txt e gere uma ATA DE REUNIÃO formal
seguindo o padrão institucional brasileiro para o SEI!.

[Cole o prompt completo de prompts/ata_sei.md]
```

---

## Convenções de Código

- **Idioma do código:** Inglês (variáveis, funções, docstrings)
- **Idioma da documentação:** Português
- **Formatação:** Black (88 colunas)
- **Type hints:** Sempre usar
- **Docstrings:** Google style

---

## Configuração Necessária

1. **Token HuggingFace** em `.env`:
   ```
   HF_TOKEN=hf_xxxxxxxxxxxxx
   ```

2. **Aceitar termos de dois modelos pyannote:**
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0

---

## Notas para Desenvolvimento

- **Não versionar:** Arquivos de áudio, transcrições, modelos baixados
- **Testar com áudios curtos** (< 5min) durante desenvolvimento
- **Modelo `small`** para testes rápidos, `large-v3` para produção
- **Device `cpu`** funciona bem em Apple Silicon

---

## Dependências Externas

- FFmpeg (conversão de áudio): `brew install ffmpeg`
- Modelos Whisper são baixados automaticamente (~3GB para large-v3)
- Python 3.12 via Homebrew: `brew install python@3.12`

---

## Problemas Conhecidos

1. **PyTorch 2.6+ weights_only:** O script inclui patch para contornar mudança de segurança
2. **Warnings suprimidos:** Warnings de torchaudio/pyannote são filtrados por padrão (use `--verbose` para ver)

---

## Funcionalidades Implementadas

- ✅ Transcrição com whisperX
- ✅ Identificação de speakers (diarização)
- ✅ Múltiplos formatos de saída (JSON, TXT, MD)
- ✅ Supressão de warnings de bibliotecas externas
- ✅ Tratamento de erros com mensagens úteis
- ✅ Otimização de performance (compute_type, batch_size)
- ✅ Liberação de memória após cada etapa
- ✅ Testes unitários (95+ testes)
- ✅ **[Fase 3]** Múltiplos backends (MLX-Whisper, WhisperX, Granite)
- ✅ **[Fase 3]** Interface bilíngue (en/pt)
- ✅ **[Fase 3]** Barra de progresso
- ✅ **[Fase 3]** Notificações macOS
- ✅ **[Fase 3]** Vocabulário customizado
- ✅ **[Fase 3]** Normalização de texto

---

*Última atualização: 15 de Janeiro de 2026*
