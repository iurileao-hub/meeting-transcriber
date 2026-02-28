# CLAUDE.md

Instruções para Claude Code ao trabalhar neste repositório.

---

## Propósito do Projeto

Sistema local de transcrição de reuniões com identificação de speakers. Converte arquivos de áudio em texto com timestamps e labels de speaker, processando 100% localmente sem envio de dados para nuvem.

**Casos de uso:** Reuniões de equipe, entrevistas, palestras, mensagens de voz, podcasts.

---

## Stack Tecnológica

- **Python 3.12** (3.14 incompatível com dependências)
- **whisperX 3.7.4** — Transcrição + diarização integrada (modo meeting)
- **faster-whisper 1.2.1** — Backend de transcrição (usado pelo whisperX)
- **mlx-whisper** — Backend otimizado para Apple Silicon (modo fast)
- **IBM Granite Speech** — Backend de alta precisão via transformers (modo precise)
- **pyannote.audio 3.4.0** — Speaker diarization
- **torch 2.8.0** — Framework de ML

---

## Estrutura do Projeto

```
meeting-transcriber/
├── CLAUDE.md             # Este arquivo (instruções para Claude Code)
├── README.md             # Documentação de uso (EN)
├── README.pt.md          # Documentação de uso (PT)
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
├── examples/             # Templates de prompts (genéricos, bilíngues)
├── prompts/              # Prompts pessoais (não versionado, gitignored)
├── vocab/                # Vocabulário
│   ├── default.txt       # Termos padrão (não versionado)
│   └── default.txt.example  # Template
├── data/
│   ├── audio/            # Arquivos de entrada (.wav, .mp3, .opus, etc.)
│   ├── transcripts/      # Saídas (.json, .txt, .md)
│   └── outputs/          # Atas e documentos
└── tests/                # 289 testes unitários
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

## Formatos de Áudio Suportados

```
.wav, .mp3, .m4a, .flac, .ogg, .webm, .aac, .opus
```

O formato `.opus` é especialmente útil para mensagens de voz do WhatsApp.

---

## Instalação dos Backends

```bash
# Backend padrão (meeting) - já incluído no requirements.txt
pip install -r requirements.txt

# Backend rápido (fast) - Apple Silicon only
pip install mlx-whisper

# Backend de alta precisão (precise) - IBM Granite
pip install transformers accelerate

# Instalar todos os backends (recomendado)
pip install mlx-whisper transformers accelerate
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

# Transcrição rápida COM diarização
python src/transcribe.py audio.wav --mode fast --diarize

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
| `--model`, `-m` | Modelo Whisper | tiny, base, small, medium, large-v3 (MLX extras: large-v3-turbo, distil-large-v3, large-v3-8bit) | large-v3 |
| `--language`, `-l` | Idioma | pt, en, es, etc. | auto |
| `--num-speakers`, `-n` | Nº exato de speakers | inteiro | auto |
| `--format`, `-f` | Formato de saída | json, txt, md, all | all |
| `--device`, `-d` | Dispositivo | cpu, cuda, mps | auto (mps/cuda/cpu) |
| `--verbose`, `-v` | Logs detalhados | flag | false |
| `--output`, `-o` | Diretório de saída | path | data/transcripts |
| `--mode` | Modo de transcrição | fast, meeting, precise | meeting |
| `--ui-lang` | Idioma da interface | en, pt | auto |
| `--notify` | Notificação macOS | flag | false |
| `--vocab` | Arquivo de vocabulário | path | - |
| `--diarize` | Diarização no modo fast | flag | false |

### Modos de Transcrição

| Modo | Backend | Diarização | Velocidade | Uso |
|------|---------|------------|------------|-----|
| `fast` | MLX-Whisper | Opcional (--diarize) | 10-15x tempo real | Transcrição rápida |
| `meeting` | WhisperX | Sim | Moderado | Reuniões (padrão) |
| `precise` | Granite + pyannote | Sim | Mais lento | Alta precisão |

---

## Workflow com Claude Code

Após transcrição, usar os templates em `examples/` (ou seus prompts pessoais em `prompts/`):

```bash
# 1. Transcrever (gera .json, .txt, .md)
python src/transcribe.py data/audio/reuniao.wav

# 2. Usar template adequado no Claude Code
```

**Templates disponíveis (bilíngues):**

| Arquivo | Uso |
|---------|-----|
| `examples/meeting_minutes.md` | Ata formal de reunião |
| `examples/executive_summary.md` | Resumo executivo + Plano de ação (5W2H/SMART) |
| `examples/action_items.md` | Extração rápida de tarefas |

**Exemplo de uso:**
```
Leia data/transcripts/reuniao.txt e gere uma ATA DE REUNIÃO formal
seguindo o template em examples/meeting_minutes.md
```

**Nota:** Crie versões personalizadas em `prompts/` (não versionado).

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
- **Device** auto-detectado: MPS em Apple Silicon, CUDA com NVIDIA, CPU como fallback

---

## Dependências Externas

- FFmpeg (conversão de áudio): `brew install ffmpeg`
- Modelos Whisper são baixados automaticamente (~3GB para large-v3)
- Python 3.12 via Homebrew: `brew install python@3.12`

---

## Problemas Conhecidos

1. **PyTorch 2.6+ weights_only:** O script usa context manager `_allow_legacy_torch_load()` para contornar mudança de segurança apenas durante carregamento de modelos pyannote (escopo limitado, não global)
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
- ✅ Testes unitários (289 testes)
- ✅ **[Fase 3]** Múltiplos backends (MLX-Whisper, WhisperX, Granite)
- ✅ **[Fase 3]** Interface bilíngue (en/pt)
- ✅ **[Fase 3]** Barra de progresso com spinner animado, timer e ETA
- ✅ **[Fase 3]** Notificações macOS
- ✅ **[Fase 3]** Vocabulário customizado
- ✅ **[Fase 3]** Normalização de texto

---

## Melhorias de Produção (v1.0)

Revisão de código realizada em Janeiro 2026:

### Granite Backend (modo precise)
- ✅ Correção de division by zero em áudio silencioso
- ✅ Uso correto de `device_map` para carregamento eficiente de modelo
- ✅ Integração real de diarização (antes ignorava resultados do pyannote)
- ✅ Normalização adequada de áudio com torchaudio

### Segurança
- ✅ Validação de path traversal em diretório de saída
- ✅ Escape completo de AppleScript em notificações
- ✅ Validação de paths de vocabulário customizado

### Arquitetura
- ✅ Factory pattern corrigido para passar configuração aos backends
- ✅ Testes expandidos para cobrir Granite backend (169 testes total)

### Documentação
- ✅ PLAN.md movido para `docs/DEVELOPMENT_HISTORY.md`
- ✅ README.md e CLAUDE.md consolidados como documentação principal

### Progresso em Tempo Real (Fevereiro 2026)
- ✅ Fix: barra de progresso agora visível (antes era engolida pelo SuppressOutput)
- ✅ Spinner animado (`⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏`) via background thread para indicar atividade
- ✅ Timer de tempo decorrido atualizado em tempo real
- ✅ Estimativa de tempo restante (ETA) por etapa
- ✅ Bilíngue (pt/en) para todas as mensagens de progresso
- ✅ Progresso granular na etapa de transcrição (intercepta print_progress do whisperX)
- ✅ Progresso granular para VAD e diarização via monkey-patch do pyannote `Inference.slide`
- ✅ Progresso granular na transcrição MLX via interceptação do tqdm
- ✅ Progresso granular na transcrição Granite via BaseStreamer (per-token)
- ✅ Auto-detecção de GPU (MPS em Apple Silicon, CUDA com NVIDIA, CPU como fallback)
- ✅ Nova etapa "Detectando fala" (VAD) visível na barra de progresso
- ✅ Contagem de stages dinâmica via `backend.total_stages` (WhisperX=6, Granite=4, MLX=3)

### Compatibilidade HuggingFace Hub (Fevereiro 2026)
- ✅ Patch de compatibilidade para `huggingface_hub` 1.x (parâmetro `use_auth_token` → `token`)
- ✅ Corrige erro de diarização com versões recentes do huggingface_hub
- ✅ Patch transparente aplicado antes de operações pyannote (VAD e diarização)
- ✅ Auto-detecta se patch é necessário (seguro para versões antigas e novas)

### Revisão de Código Completa (Fevereiro 2026)

Revisão abrangente cobrindo performance, segurança, qualidade de código e arquitetura:

#### Segurança
- ✅ `torch.load` patch limitado via context manager (antes era global no import)
- ✅ Vocabulário: allowlist substitui denylist (paths restritos ao diretório do projeto)
- ✅ Limite de 1MB para arquivos de vocabulário (prevenção de DoS)
- ✅ Strip de null bytes e control chars no AppleScript
- ✅ Output path hardening: `~/.ssh`, `~/.aws`, `~/.kube` bloqueados
- ✅ Sanitização de exceções (não expõe paths internos sem `--verbose`)
- ✅ Thread-safe patches com `threading.Lock` em `patch_hf_hub_compat`
- ✅ Logging de tentativas de acesso a paths bloqueados

#### Performance
- ✅ Lazy imports: backends carregados sob demanda (startup ~3s mais rápido)
- ✅ Granite: áudio carregado uma única vez (antes era 2x do disco)
- ✅ Binary search O(log N) para speaker matching no Granite (antes O(N×M))
- ✅ Progress rendering com throttle de 100ms (antes: até 448 renders por geração)
- ✅ `inspect.signature()` substituído por dicionário estático no factory
- ✅ `locale.getdefaultlocale()` substituído por `locale.getlocale()` (Python 3.15 ready)

#### Arquitetura
- ✅ `_load_hf_token` centralizado na base class (antes: 4 cópias)
- ✅ ABC `transcribe()` completo com `progress_callback`, `min/max_speakers`
- ✅ `is_available()` no ABC com check no factory
- ✅ `_build_diarize_kwargs()` helper centralizado (antes: 3 cópias)
- ✅ `_save_results()` extraído de `transcribe()` (SRP)
- ✅ `translator.lang` exposto (substitui comparação frágil de string)
- ✅ `SuppressOutput.__exit__` corrigido (`is not None` vs truthy)
- ✅ `ProgressReporter` aceita `file` parameter explícito
- ✅ `ProgressStreamer` protegido contra `max_tokens=0`
- ✅ Código morto removido (`get_compute_type`, `get_batch_size`, `load_hf_token`)

#### Testes (+74 novos, total 289)
- ✅ Testes para `_allow_legacy_torch_load` context manager
- ✅ Testes para `_save_results()` (JSON, TXT, MD, all)
- ✅ Testes para `SuppressOutput` (restore, exception safety)
- ✅ Testes para `format_timestamp` (incluindo horas)
- ✅ Testes para `ProgressReporter.error()`, throttle, file param
- ✅ Testes para `format_duration` edge cases
- ✅ Testes para vocabulary allowlist, size limit, logging
- ✅ Testes para notify control char stripping
- ✅ Testes para `translator.lang`, `detect_system_language()`
- ✅ Testes para `_build_diarize_kwargs`, `_load_hf_token` base class
- ✅ Testes para binary search no Granite speaker matching
- ✅ Testes para lazy imports e `is_available()` factory check

---

*Última atualização: 27 de Fevereiro de 2026 (revisão completa de código)*
