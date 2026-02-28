# Changelog

Registro de funcionalidades implementadas e melhorias do projeto Meeting Transcriber.

---

## Funcionalidades Base

- Transcrição com whisperX
- Identificação de speakers (diarização)
- Múltiplos formatos de saída (JSON, TXT, MD)
- Supressão de warnings de bibliotecas externas
- Tratamento de erros com mensagens úteis
- Otimização de performance (compute_type, batch_size)
- Liberação de memória após cada etapa
- Testes unitários (289 testes)

---

## Fase 3 — Múltiplos Backends e UX

- Múltiplos backends (MLX-Whisper, WhisperX, Granite)
- Interface bilíngue (en/pt)
- Barra de progresso com spinner animado, timer e ETA
- Notificações macOS
- Vocabulário customizado
- Normalização de texto

---

## Melhorias de Produção (v1.0) — Janeiro 2026

Revisão de código realizada em Janeiro 2026.

### Granite Backend (modo precise)
- Correção de division by zero em áudio silencioso
- Uso correto de `device_map` para carregamento eficiente de modelo
- Integração real de diarização (antes ignorava resultados do pyannote)
- Normalização adequada de áudio com torchaudio

### Segurança
- Validação de path traversal em diretório de saída
- Escape completo de AppleScript em notificações
- Validação de paths de vocabulário customizado

### Arquitetura
- Factory pattern corrigido para passar configuração aos backends
- Testes expandidos para cobrir Granite backend (169 testes total)

### Documentação
- PLAN.md movido para `docs/DEVELOPMENT_HISTORY.md`
- README.md e CLAUDE.md consolidados como documentação principal

---

## Progresso em Tempo Real — Fevereiro 2026

- Fix: barra de progresso agora visível (antes era engolida pelo SuppressOutput)
- Spinner animado (`⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏`) via background thread para indicar atividade
- Timer de tempo decorrido atualizado em tempo real
- Estimativa de tempo restante (ETA) por etapa
- Bilíngue (pt/en) para todas as mensagens de progresso
- Progresso granular na etapa de transcrição (intercepta print_progress do whisperX)
- Progresso granular para VAD e diarização via monkey-patch do pyannote `Inference.slide`
- Progresso granular na transcrição MLX via interceptação do tqdm
- Progresso granular na transcrição Granite via BaseStreamer (per-token)
- Auto-detecção de GPU (MPS em Apple Silicon, CUDA com NVIDIA, CPU como fallback)
- Nova etapa "Detectando fala" (VAD) visível na barra de progresso
- Contagem de stages dinâmica via `backend.total_stages` (WhisperX=6, Granite=4, MLX=3)

---

## Compatibilidade HuggingFace Hub — Fevereiro 2026

- Patch de compatibilidade para `huggingface_hub` 1.x (parâmetro `use_auth_token` → `token`)
- Corrige erro de diarização com versões recentes do huggingface_hub
- Patch transparente aplicado antes de operações pyannote (VAD e diarização)
- Auto-detecta se patch é necessário (seguro para versões antigas e novas)

---

## Revisão de Código Completa — Fevereiro 2026

Revisão abrangente cobrindo performance, segurança, qualidade de código e arquitetura.

### Segurança
- `torch.load` patch limitado via context manager (antes era global no import)
- Vocabulário: allowlist substitui denylist (paths restritos ao diretório do projeto)
- Limite de 1MB para arquivos de vocabulário (prevenção de DoS)
- Strip de null bytes e control chars no AppleScript
- Output path hardening: `~/.ssh`, `~/.aws`, `~/.kube` bloqueados
- Sanitização de exceções (não expõe paths internos sem `--verbose`)
- Thread-safe patches com `threading.Lock` em `patch_hf_hub_compat`
- Logging de tentativas de acesso a paths bloqueados

### Performance
- Lazy imports: backends carregados sob demanda (startup ~3s mais rápido)
- Granite: áudio carregado uma única vez (antes era 2x do disco)
- Binary search O(log N) para speaker matching no Granite (antes O(N×M))
- Progress rendering com throttle de 100ms (antes: até 448 renders por geração)
- `inspect.signature()` substituído por dicionário estático no factory
- `locale.getdefaultlocale()` substituído por `locale.getlocale()` (Python 3.15 ready)

### Arquitetura
- `_load_hf_token` centralizado na base class (antes: 4 cópias)
- ABC `transcribe()` completo com `progress_callback`, `min/max_speakers`
- `is_available()` no ABC com check no factory
- `_build_diarize_kwargs()` helper centralizado (antes: 3 cópias)
- `_save_results()` extraído de `transcribe()` (SRP)
- `translator.lang` exposto (substitui comparação frágil de string)
- `SuppressOutput.__exit__` corrigido (`is not None` vs truthy)
- `ProgressReporter` aceita `file` parameter explícito
- `ProgressStreamer` protegido contra `max_tokens=0`
- Código morto removido (`get_compute_type`, `get_batch_size`, `load_hf_token`)

### Testes (+74 novos, total 289)
- Testes para `_allow_legacy_torch_load` context manager
- Testes para `_save_results()` (JSON, TXT, MD, all)
- Testes para `SuppressOutput` (restore, exception safety)
- Testes para `format_timestamp` (incluindo horas)
- Testes para `ProgressReporter.error()`, throttle, file param
- Testes para `format_duration` edge cases
- Testes para vocabulary allowlist, size limit, logging
- Testes para notify control char stripping
- Testes para `translator.lang`, `detect_system_language()`
- Testes para `_build_diarize_kwargs`, `_load_hf_token` base class
- Testes para binary search no Granite speaker matching
- Testes para lazy imports e `is_available()` factory check

---

*Última atualização: 27 de Fevereiro de 2026*
