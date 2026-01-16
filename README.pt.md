# Meeting Transcriber

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Plataforma](https://img.shields.io/badge/Plataforma-macOS%20Apple%20Silicon-lightgrey.svg)](https://support.apple.com/pt-br/HT211814)
[![Licença](https://img.shields.io/badge/Licen%C3%A7a-MIT-green.svg)](LICENSE)
[![Testes](https://img.shields.io/badge/testes-95%20passaram-brightgreen.svg)](tests/)
[![Offline](https://img.shields.io/badge/funciona-100%25%20offline-blueviolet.svg)]()

**Transforme gravações de reuniões em transcrições com identificação de quem está falando — 100% local, sem nuvem.**

[Read in English](README.md)

---

## O Que Faz

O Meeting Transcriber converte arquivos de áudio em texto, identificando automaticamente cada pessoa que fala. Seu áudio nunca sai do seu computador.

**Ideal para:**
- Reuniões de equipe e entrevistas
- Aulas e apresentações
- Mensagens de voz do WhatsApp
- Podcasts e gravações

**Exemplo de saída:**
```
[00:00] SPEAKER_00: Bom dia a todos, vamos começar a reunião.
[00:05] SPEAKER_01: Obrigado pela presença. Primeiro item da pauta...
[00:12] SPEAKER_00: Antes de começar, alguma atualização da semana passada?
```

---

## Início Rápido

```bash
# 1. Clone e configure
git clone https://github.com/yourusername/meeting-transcriber.git
cd meeting-transcriber
python3.12 -m venv venv && source venv/bin/activate

# 2. Instale
brew install ffmpeg
pip install -r requirements.txt

# 3. Transcreva!
python src/transcribe.py seu-audio.mp3
```

> **Primeira vez?** Você precisará de uma conta gratuita no [HuggingFace](#configuracao-huggingface) para identificação de speakers.

---

## Formatos de Áudio Suportados

| Formato | Extensão | Origem Comum |
|---------|----------|--------------|
| MP3 | `.mp3` | Maioria dos players |
| WAV | `.wav` | Gravações profissionais |
| M4A | `.m4a` | Gravações iPhone/Mac |
| Opus | `.opus` | Mensagens de voz WhatsApp |
| FLAC | `.flac` | Áudio sem perdas |
| OGG | `.ogg` | Gravações web |
| WebM | `.webm` | Gravações de navegador |
| AAC | `.aac` | Transmissões digitais |

---

## Instalação

### Requisitos

| Requisito | Detalhes |
|-----------|----------|
| **Computador** | Mac com Apple Silicon (chip M1, M2, M3 ou M4) |
| **Python** | Versão 3.12 (não 3.14) |
| **Espaço em Disco** | ~10GB para modelos |
| **Internet** | Apenas para configuração inicial |

### Instalação Passo a Passo

<details>
<summary><strong>1. Instalar Python 3.12</strong> (se não tiver)</summary>

```bash
# Usando Homebrew
brew install python@3.12

# Verificar instalação
python3.12 --version
```
</details>

<details>
<summary><strong>2. Instalar FFmpeg</strong> (processamento de áudio)</summary>

```bash
brew install ffmpeg
```

FFmpeg é uma ferramenta gratuita que converte formatos de áudio.
</details>

<details>
<summary><strong>3. Configurar o projeto</strong></summary>

```bash
# Clonar o repositório
git clone https://github.com/yourusername/meeting-transcriber.git
cd meeting-transcriber

# Criar ambiente Python isolado
python3.12 -m venv venv
source venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```
</details>

<details>
<summary><strong>4. Configurar HuggingFace</strong> (necessário para identificação de speakers)</summary>

<a name="configuracao-huggingface"></a>

O HuggingFace fornece os modelos de IA para identificação de speakers. É gratuito.

1. **Criar conta** em [huggingface.co](https://huggingface.co/join)

2. **Obter seu token:**
   - Vá em [Settings → Access Tokens](https://huggingface.co/settings/tokens)
   - Clique em "New token" → Dê qualquer nome → Create
   - Copie o token (começa com `hf_`)

3. **Salvar o token:**
   ```bash
   cp .env.example .env
   # Edite .env e cole seu token após HF_TOKEN=
   ```

4. **Aceitar termos dos modelos** (uma vez só):
   - Acesse [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - Clique em "Agree and access repository"
   - Acesse [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   - Clique em "Agree and access repository"

</details>

### Opcional: Backends Adicionais

A instalação padrão atende a maioria das necessidades. Para casos especializados:

| Modo | Comando de Instalação | Ideal Para |
|------|----------------------|------------|
| `fast` | `pip install mlx-whisper` | Transcrições rápidas (sem ID de speaker) |
| `precise` | `pip install transformers accelerate` | Máxima precisão (IBM Granite) |

```bash
# Instalar todos os backends (recomendado para usuários avançados)
pip install mlx-whisper transformers accelerate
```

> **Nota:** O modo `precise` usa IBM Granite 8B, que requer ~16GB de RAM e GPU para velocidade razoável. Para máquinas com menos memória, use o modo `meeting` (padrão).

---

## Guia de Uso

### Uso Básico

```bash
# Transcrever qualquer arquivo de áudio suportado
python src/transcribe.py reuniao.mp3

# Saída: Cria reuniao.json, reuniao.txt e reuniao.md
```

### Escolha Seu Modo

| Eu quero... | Comando | Observações |
|-------------|---------|-------------|
| **Melhor qualidade** (padrão) | `python src/transcribe.py audio.mp3` | Identificação de speakers incluída |
| **Resultado mais rápido** | `python src/transcribe.py audio.mp3 --mode fast` | Sem identificação de speakers |
| **Máxima precisão** | `python src/transcribe.py audio.mp3 --mode precise` | Mais lento, porém mais preciso |

### Opções Comuns

```bash
# Especificar idioma (melhora a precisão)
python src/transcribe.py reuniao.mp3 --language pt

# Sabe quantas pessoas falam? Informe ao sistema
python src/transcribe.py reuniao.mp3 --num-speakers 3

# Receber notificação quando terminar (útil para arquivos longos)
python src/transcribe.py reuniao.mp3 --notify

# Gerar apenas arquivo texto (mais rápido)
python src/transcribe.py reuniao.mp3 --format txt
```

### Referência Completa de Opções

<details>
<summary>Clique para expandir tabela completa</summary>

| Opção | Curto | Descrição | Padrão |
|-------|-------|-----------|--------|
| `--model` | `-m` | Tamanho do modelo (tiny/base/small/medium/large-v3) | large-v3 |
| `--language` | `-l` | Idioma do áudio (en, pt, es, etc.) | auto-detectar |
| `--num-speakers` | `-n` | Número exato de speakers | auto-detectar |
| `--min-speakers` | | Mínimo de speakers esperado | - |
| `--max-speakers` | | Máximo de speakers esperado | - |
| `--output` | `-o` | Onde salvar arquivos | data/transcripts |
| `--format` | `-f` | Formato de saída (json/txt/md/all) | all |
| `--mode` | | Modo de transcrição (fast/meeting/precise) | meeting |
| `--device` | `-d` | Processador (cpu/cuda/mps) | cpu |
| `--notify` | | Notificação macOS ao terminar | desligado |
| `--vocab` | | Arquivo de vocabulário customizado | - |
| `--ui-lang` | | Idioma da interface (en/pt) | auto |
| `--verbose` | `-v` | Mostrar logs detalhados | desligado |

</details>

---

## Formatos de Saída

### Texto (.txt) — Leitura humana
```
[00:00] SPEAKER_00: Bom dia a todos.
[00:05] SPEAKER_01: Obrigado pela presença.
```

### Markdown (.md) — Formatado para documentos
```markdown
## Transcrição da Reunião

**[00:00] Speaker 1:** Bom dia a todos.

**[00:05] Speaker 2:** Obrigado pela presença.
```

### JSON (.json) — Para desenvolvedores
```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 4.2,
      "text": "Bom dia a todos.",
      "speaker": "SPEAKER_00"
    }
  ],
  "metadata": {
    "language": "pt",
    "num_speakers": 2
  }
}
```

---

## Seleção de Modelo

Modelos maiores são mais precisos, porém mais lentos e usam mais memória.

| Modelo | Precisão | Velocidade | RAM Necessária | Recomendado Para |
|--------|----------|------------|----------------|------------------|
| tiny | Baixa | Muito rápido | 1GB | Apenas testes |
| base | Média | Rápido | 1GB | Rascunhos rápidos |
| small | Boa | Moderado | 2GB | Uso diário |
| medium | Muito boa | Lento | 5GB | Reuniões importantes |
| **large-v3** | Excelente | Mais lento | 10GB | Produção (padrão) |

```bash
# Usar modelo menor para testes
python src/transcribe.py reuniao.mp3 --model small

# Usar modelo maior para gravações importantes
python src/transcribe.py reuniao.mp3 --model large-v3
```

---

## Solução de Problemas

<details>
<summary><strong>Erro: "No module named 'whisperx'"</strong></summary>

Seu ambiente virtual não está ativado.

```bash
source venv/bin/activate
pip install -r requirements.txt
```
</details>

<details>
<summary><strong>Erro de memória</strong></summary>

Use um modelo menor:

```bash
python src/transcribe.py audio.mp3 --model small
```
</details>

<details>
<summary><strong>Speakers identificados incorretamente</strong></summary>

Informe ao sistema quantos speakers há:

```bash
python src/transcribe.py audio.mp3 --num-speakers 3
```
</details>

<details>
<summary><strong>Idioma detectado errado</strong></summary>

Especifique o idioma explicitamente:

```bash
python src/transcribe.py audio.mp3 --language pt
```
</details>

<details>
<summary><strong>Transcrição lenta</strong></summary>

- Use `--mode fast` para velocidade (sem identificação de speakers)
- Use `--model small` para processamento mais rápido
- Feche outros aplicativos para liberar memória
</details>

<details>
<summary><strong>Erro de autenticação HuggingFace</strong></summary>

1. Verifique seu token no arquivo `.env`
2. Confirme que aceitou os termos dos dois modelos pyannote
3. Verifique o token em [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
</details>

---

## Privacidade e Segurança

**Seus dados ficam no seu computador.**

- Todo processamento acontece localmente — nenhum áudio é enviado para lugar nenhum
- Não é necessária conexão com internet após a configuração inicial
- Modelos são baixados uma vez e armazenados localmente
- Sem telemetria, analytics ou coleta de dados

Isso torna o Meeting Transcriber ideal para:
- Reuniões empresariais confidenciais
- Consultas médicas
- Processos jurídicos
- Gravações pessoais

---

## Vocabulário Customizado

Melhore a precisão para termos específicos do seu domínio:

```bash
# Crie vocab/default.txt com seus termos (um por linha):
PMDF
Dr. Silva
Kubernetes
API
```

O sistema carrega automaticamente `vocab/default.txt` se existir, ou especifique um arquivo customizado:

```bash
python src/transcribe.py reuniao.mp3 --vocab meus-termos.txt
```

---

## Integração com Claude

Após a transcrição, use IA para gerar atas de reunião:

```bash
# 1. Transcrever
python src/transcribe.py reuniao.mp3

# 2. Peça ao Claude para processar a transcrição
# Veja a pasta examples/ para templates de prompts:
#   - examples/meeting_minutes.md — Ata formal de reunião
#   - examples/executive_summary.md — Resumo executivo com plano de ação
#   - examples/action_items.md — Extração rápida de tarefas
```

---

## Agradecimentos

Construído com estes excelentes projetos open-source:

- [WhisperX](https://github.com/m-bain/whisperX) — Reconhecimento de fala com timestamps por palavra
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — Inferência Whisper otimizada
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) — Diarização de speakers
- [MLX](https://github.com/ml-explore/mlx) — Framework ML para Apple Silicon

---

## Licença

Licença MIT — livre para uso pessoal e comercial.

---

## Autor

**Iuri Almeida**
Médico | Gestor de Segurança Pública | Estudante de Ciência da Computação

*Janeiro de 2026*
