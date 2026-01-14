# Meeting Transcriber - Plano de Desenvolvimento

> Sistema local de transcrição de reuniões otimizado para Mac Apple Silicon

**Última atualização:** Janeiro 2026

---

## Resumo Executivo

Sistema de transcrição local que:
1. Processa áudio em português e inglês
2. Identifica speakers automaticamente
3. Gera JSONs estruturados para processamento com Claude Code
4. Roda 100% local em Mac Apple Silicon

**Stack final:**
- `whisperX 3.7.4` — Transcrição + diarização integrada
- `faster-whisper` — Backend de transcrição (usado pelo whisperX)
- `pyannote.audio 3.4.0` — Speaker diarization
- Python 3.12

---

## Por Que Esta Stack

### mlx-whisper vs Alternativas

| Ferramenta | Velocidade (M1/M2) | Instalação | Manutenção |
|------------|-------------------|------------|------------|
| **mlx-whisper** | 10-15x real-time | Simples | Apple/MLX team |
| whisper (original) | 2-3x real-time | Simples | OpenAI |
| faster-whisper | 4-6x real-time | Média | Comunidade |

**Decisão:** `mlx-whisper` é a escolha óbvia para Apple Silicon.

### whisperX para Diarização

`whisperX` integra:
- Transcrição (usa faster-whisper internamente)
- Alinhamento fonético (word-level timestamps)
- Speaker diarization (via pyannote embarcado)

**Vantagem:** Um único pipeline ao invés de combinar ferramentas separadas.

**Trade-off:** whisperX usa faster-whisper, não mlx-whisper. Para reuniões, a diferença de velocidade (~2-3x vs 10x) não é crítica.

---

## Estrutura do Projeto

```
meeting-transcriber/
├── PLAN.md                 # Este arquivo
├── README.md               # Documentação de uso
├── requirements.txt        # Dependências Python
├── .env.example            # Template de configuração
├── .gitignore
├── src/
│   ├── transcribe.py       # Script principal de transcrição
│   └── utils.py            # Funções auxiliares (se necessário)
├── data/
│   ├── audio/              # Arquivos .wav/.mp3 de entrada
│   ├── transcripts/        # JSONs com transcritos
│   └── outputs/            # Atas e documentos gerados
└── tests/
    └── test_transcribe.py  # Testes básicos
```

**Filosofia:** Estrutura mínima. Expandir apenas quando necessário.

---

## Roadmap de Desenvolvimento

### Fase 1: MVP com Transcrição + Diarização ✅ CONCLUÍDA

**Objetivo:** Script funcional que transcreve e identifica speakers.

**Tarefas:**
1. [x] Setup ambiente Python 3.12 (via Homebrew)
2. [x] Instalar whisperX (inclui diarização)
3. [x] Obter token HuggingFace para pyannote
4. [x] Criar `transcribe.py` com CLI completa
5. [x] Testar com áudio de exemplo
6. [x] Validar output JSON

**Entregável:** Script que recebe áudio e gera JSON estruturado. ✅

**Código esperado:**

```python
# src/transcribe.py - MVP
import whisperx
import json
from pathlib import Path

def transcribe(audio_path: str, output_dir: str = "data/transcripts"):
    """Transcreve áudio com identificação de speakers."""

    device = "cpu"  # mps para Apple Silicon se suportado
    compute_type = "float32"

    # 1. Carregar modelo
    model = whisperx.load_model("large-v3", device, compute_type=compute_type)

    # 2. Transcrever
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio)

    # 3. Alinhar (word-level timestamps)
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"],
        device=device
    )
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device
    )

    # 4. Diarização (identificar speakers)
    diarize_model = whisperx.DiarizationPipeline(
        use_auth_token="YOUR_HF_TOKEN",  # Do .env
        device=device
    )
    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    # 5. Salvar JSON
    output_path = Path(output_dir) / f"{Path(audio_path).stem}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result

if __name__ == "__main__":
    import sys
    audio_file = sys.argv[1] if len(sys.argv) > 1 else "data/audio/exemplo.wav"
    transcribe(audio_file)
```

**Output JSON esperado:**

```json
{
  "segments": [
    {
      "start": 0.5,
      "end": 5.2,
      "text": "Bom dia a todos, vamos começar a reunião.",
      "speaker": "SPEAKER_00",
      "words": [
        {"word": "Bom", "start": 0.5, "end": 0.7, "speaker": "SPEAKER_00"},
        {"word": "dia", "start": 0.72, "end": 0.95, "speaker": "SPEAKER_00"}
      ]
    },
    {
      "start": 5.5,
      "end": 12.3,
      "text": "Obrigado. Gostaria de apresentar os resultados do trimestre.",
      "speaker": "SPEAKER_01"
    }
  ],
  "language": "pt"
}
```

---

### Fase 2: CLI + Integração Claude Code (Semana 3)

**Objetivo:** Interface de linha de comando e workflow com Claude Code.

**Tarefas:**
1. [ ] Adicionar argparse para CLI
2. [ ] Opções: modelo, idioma, output format
3. [ ] Criar template de prompt para atas
4. [ ] Documentar workflow Claude Code
5. [ ] Processar primeira reunião real

**CLI esperada:**

```bash
# Transcrição básica
python src/transcribe.py data/audio/reuniao.wav

# Com opções
python src/transcribe.py data/audio/reuniao.wav \
    --model large-v3 \
    --language pt \
    --output data/transcripts/

# Forçar número de speakers (se conhecido)
python src/transcribe.py data/audio/reuniao.wav --num-speakers 4
```

**Workflow com Claude Code:**

```bash
# 1. Transcrever
python src/transcribe.py data/audio/reuniao-pmdf-2026-01-15.wav

# 2. Usar Claude Code para gerar ata
# No Claude Code:
# "Leia o arquivo data/transcripts/reuniao-pmdf-2026-01-15.json
#  e gere uma ata estruturada com: participantes, pauta,
#  decisões, action items e próximos passos."
```

---

### Fase 3: Otimizações (Semana 4, opcional)

**Melhorias baseadas em uso real:**

1. **Performance:**
   - Cache de modelos
   - Processamento batch
   - Testar mlx-whisper para transcrição pura (sem diarização)

2. **Qualidade:**
   - Vocabulário customizado (termos médicos, militares)
   - Mapeamento de nomes de speakers
   - Correção de erros comuns

3. **UX:**
   - Progress bar
   - Estimativa de tempo
   - Notificação ao finalizar

---

## Requisitos de Sistema

### Hardware
- Mac com Apple Silicon (M1/M2/M3/M4)
- 8GB RAM mínimo (16GB recomendado para modelos large)
- 10GB espaço em disco (modelos)

### Software
- Python 3.11+
- FFmpeg (para conversão de áudio)
- Homebrew (para instalar dependências)

### Tokens/Accounts
- **HuggingFace:** Token gratuito para pyannote (speaker diarization)
  - Criar conta: https://huggingface.co
  - Aceitar termos: https://huggingface.co/pyannote/speaker-diarization-3.1
  - Gerar token: https://huggingface.co/settings/tokens

---

## Setup Inicial

```bash
# 1. Criar ambiente virtual
cd meeting-transcriber
python3.11 -m venv venv
source venv/bin/activate

# 2. Instalar FFmpeg (se não tiver)
brew install ffmpeg

# 3. Instalar dependências
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configurar token HuggingFace
cp .env.example .env
# Editar .env e adicionar HF_TOKEN

# 5. Testar instalação
python -c "import whisperx; print('whisperX OK')"
```

---

## Conceitos a Aprender

### Fase 1 - Fundamentos

- [ ] **ASR (Automatic Speech Recognition):** Como modelos como Whisper convertem áudio em texto
- [ ] **Sampling rate:** Por que 16kHz é o padrão para ASR
- [ ] **Speaker embeddings:** Como diarização identifica diferentes vozes
- [ ] **VAD (Voice Activity Detection):** Detecção de fala vs silêncio

### Fase 2 - Prático

- [ ] **Alinhamento forçado:** Como mapear palavras a timestamps exatos
- [ ] **Clustering de speakers:** Algoritmos que agrupam vozes similares
- [ ] **Formato JSON para transcrições:** Estruturas padrão da indústria

### Fase 3 - Avançado

- [ ] **Quantização de modelos:** Reduzir tamanho mantendo qualidade
- [ ] **Fine-tuning:** Adaptar modelo para vocabulário específico
- [ ] **Streaming ASR:** Transcrição em tempo real (futuro)

---

## Troubleshooting Esperado

### Problema: "No module named 'whisperx'"
**Solução:** Verificar ambiente virtual ativo: `which python`

### Problema: Erro de memória com modelo large-v3
**Solução:** Usar modelo `medium` ou `small`

### Problema: Diarização não identifica speakers corretamente
**Soluções:**
1. Verificar qualidade do áudio (ruído de fundo)
2. Ajustar `min_speakers` e `max_speakers`
3. Áudio muito curto (<30s) dificulta diarização

### Problema: Transcrição em idioma errado
**Solução:** Forçar idioma com `--language pt` ou `--language en`

---

## Métricas de Sucesso

### MVP (Fase 1) ✅
- [x] Transcreve áudio de 30s em ~40s (modelo small)
- [x] Identifica speakers corretamente
- [x] Gera JSON válido e legível
- [ ] Precisão >85% (avaliação manual em amostra) — pendente teste com reunião real

### Produção (Fase 2-3)
- [x] CLI funcional com help
- [ ] Workflow documentado para Claude Code
- [ ] Primeira ata de reunião real gerada
- [ ] Tempo total (transcrição + ata) < 15min para reunião de 1h

---

## Referências

- whisperX: https://github.com/m-bain/whisperX
- mlx-whisper: https://github.com/ml-explore/mlx-examples/tree/main/whisper
- pyannote (diarization): https://github.com/pyannote/pyannote-audio
- HuggingFace pyannote: https://huggingface.co/pyannote/speaker-diarization-3.1

---

## Log de Progresso

### 14 de Janeiro de 2026 — MVP Concluído
- [x] Projeto iniciado e estrutura criada
- [x] Python 3.12 instalado via Homebrew (3.14 incompatível com whisperX)
- [x] Ambiente virtual configurado
- [x] whisperX 3.7.4 instalado com todas as dependências
- [x] Token HuggingFace configurado
- [x] Termos de uso aceitos (pyannote/speaker-diarization-3.1, pyannote/segmentation-3.0)
- [x] Script `transcribe.py` criado com CLI completa (~240 linhas)
- [x] Fix implementado para PyTorch 2.6+ (weights_only)
- [x] Primeiro teste bem-sucedido com áudio de exemplo
- [x] JSON gerado com 10 segmentos, timestamps e speaker identification

**Problemas resolvidos:**
1. Python 3.14 incompatível → instalado Python 3.12
2. PyTorch 2.6+ `weights_only=True` → patch no torch.load
3. whisperX API mudou → import de `whisperx.diarize.DiarizationPipeline`
4. Modelos gated no HuggingFace → aceitar termos de múltiplos modelos

---

*Autor: Iuri Almeida*
*Projeto educacional para aprendizado de ASR e processamento de áudio*
