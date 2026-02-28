# Meeting Transcriber

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Plataforma](https://img.shields.io/badge/Plataforma-macOS%20Apple%20Silicon-lightgrey.svg)](https://support.apple.com/pt-br/HT211814)
[![Licenca](https://img.shields.io/badge/Licen%C3%A7a-MIT-green.svg)](LICENSE)
[![Testes](https://img.shields.io/badge/testes-289%20passaram-brightgreen.svg)](tests/)
[![Offline](https://img.shields.io/badge/funciona-100%25%20offline-blueviolet.svg)]()

**Transforme gravacoes de reunioes em transcricoes com identificacao de quem falou — 100% local, sem nuvem.**

[Read in English](README.md)

---

## O Que Ele Faz

O Meeting Transcriber ouve a gravacao de uma reuniao (ou entrevista, aula, mensagem de voz, podcast...) e produz um documento escrito que mostra **quem falou o que, e quando**. Ele identifica as diferentes pessoas automaticamente e marca cada parte da conversa.

Tudo acontece no seu proprio computador. Seus arquivos de audio nunca sao enviados para lugar nenhum.

**Veja como fica o resultado:**

```
[00:00] SPEAKER_00: Bom dia a todos, vamos comecar a reuniao.
[00:05] SPEAKER_01: Obrigado pela presenca. Primeiro item da pauta...
[00:12] SPEAKER_00: Antes de comecar, alguma atualizacao da semana passada?
[00:18] SPEAKER_02: Sim, o cliente aprovou a proposta ontem.
```

Voce recebe tres arquivos de saida: uma versao em texto simples (facil de ler), uma versao formatada em Markdown (fica bonita em documentos), e um arquivo JSON (util se voce quiser processar os dados depois).

---

## O Que Voce Precisa

Antes de comecar, verifique se voce tem o seguinte:

| O que | Detalhes |
|-------|----------|
| **Um Mac com Apple Silicon** | Ou seja, chip M1, M2, M3 ou M4. Voce pode verificar clicando no menu da Apple e selecionando "Sobre Este Mac". |
| **Cerca de 10 GB de espaco livre em disco** | Os modelos de IA que fazem a transcricao sao arquivos grandes. Eles sao baixados uma unica vez e ficam armazenados no seu computador. |
| **Uma conexao com a internet** | So e necessaria na primeira vez, para baixar o programa e seus modelos. Depois disso, tudo funciona offline. |

---

## Instalacao

Esta secao te guia pela configuracao do Meeting Transcriber passo a passo. Cada passo inclui uma forma de verificar se funcionou. Se algo der errado, consulte a secao [Solucao de Problemas](#solucao-de-problemas).

### Passo 1: Instalar o Homebrew (se voce ainda nao tem)

O Homebrew e uma ferramenta que facilita a instalacao de programas no Mac. Abra o aplicativo **Terminal** (voce pode encontra-lo em Aplicativos > Utilitarios, ou buscar por "Terminal" no Spotlight) e cole este comando:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**Verifique se funcionou:**
```bash
brew --version
```
Voce deve ver um numero de versao como `Homebrew 4.x.x`.

### Passo 2: Instalar o Python 3.12

Python e a linguagem de programacao na qual esta ferramenta foi escrita. Voce precisa especificamente da versao 3.12 (versoes mais novas como 3.14 nao sao compativeis com algumas das bibliotecas que usamos).

```bash
brew install python@3.12
```

**Verifique se funcionou:**
```bash
python3.12 --version
```
Voce deve ver `Python 3.12.x`.

### Passo 3: Instalar o FFmpeg

O FFmpeg e uma ferramenta gratuita que cuida da conversao de arquivos de audio nos bastidores. O Meeting Transcriber usa-o para ler diferentes formatos de audio.

```bash
brew install ffmpeg
```

**Verifique se funcionou:**
```bash
ffmpeg -version
```
Voce deve ver informacoes de versao (a primeira linha ja e suficiente).

### Passo 4: Baixar o Meeting Transcriber

Este comando baixa o programa para o seu computador:

```bash
git clone https://github.com/iurileao-hub/meeting-transcriber.git
cd meeting-transcriber
```

### Passo 5: Criar um ambiente virtual

Um ambiente virtual e como uma pasta separada onde o programa e todos os seus arquivos vivem, sem afetar o resto do seu computador. Isso mantem tudo organizado e evita conflitos com outros programas.

```bash
python3.12 -m venv venv
source venv/bin/activate
```

Apos executar o segundo comando, voce deve ver `(venv)` no inicio da linha do seu terminal. Isso significa que o ambiente virtual esta ativo.

> **Importante:** Toda vez que voce abrir uma nova janela do Terminal para usar o Meeting Transcriber, voce precisa ativar o ambiente virtual novamente:
> ```bash
> cd meeting-transcriber
> source venv/bin/activate
> ```

### Passo 6: Instalar as dependencias do programa

Dependencias sao as bibliotecas e ferramentas que o Meeting Transcriber precisa para funcionar. Este comando baixa e instala todas elas:

```bash
pip install -r requirements.txt
```

Isso pode levar alguns minutos. Voce vera muitas linhas rolando na tela — isso e normal.

**Verifique se funcionou:**
```bash
python -c "import whisperx; print('OK')"
```
Voce deve ver `OK`.

### Passo 7: Configurar o HuggingFace (necessario para identificacao de speakers)

O HuggingFace e um site que hospeda modelos de IA gratuitos. O Meeting Transcriber precisa de acesso a dois desses modelos para distinguir quem esta falando. Isso e completamente gratuito.

**7a. Crie uma conta gratuita:**
- Acesse [huggingface.co/join](https://huggingface.co/join) e faca seu cadastro.

**7b. Obtenha seu token de acesso:**

Um token e uma senha especial que permite ao programa baixar modelos de IA do HuggingFace.

- Va em [Settings > Access Tokens](https://huggingface.co/settings/tokens)
- Clique em "New token", de qualquer nome (como "meeting-transcriber"), e clique em Create.
- Copie o token (ele comeca com `hf_`).

**7c. Salve o token:**
```bash
cp .env.example .env
```

Agora abra o arquivo `.env` em qualquer editor de texto e substitua `hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx` pelo token que voce acabou de copiar. Salve o arquivo.

**7d. Aceite os termos dos modelos de IA (so precisa fazer uma vez):**

Voce precisa visitar duas paginas e clicar em "Agree and access repository" em cada uma:

1. [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) -- clique em "Agree and access repository"
2. [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) -- clique em "Agree and access repository"

Pronto! Voce so precisa fazer isso uma vez.

---

## Sua Primeira Transcricao

Vamos garantir que tudo esta funcionando. Coloque um arquivo de audio (qualquer `.mp3`, `.wav`, `.m4a` ou outro [formato suportado](#formatos-de-audio-suportados)) dentro da pasta `data/audio/` e execute:

```bash
python src/transcribe.py data/audio/seu-arquivo.mp3
```

O programa vai mostrar uma barra de progresso enquanto trabalha em varias etapas: carregando o modelo de IA, detectando fala, transcrevendo e identificando speakers. Dependendo da duracao do seu audio e do tamanho do modelo, isso pode levar de alguns segundos a varios minutos.

Quando terminar, voce vai encontrar suas transcricoes na pasta `data/transcripts/`:
- `seu-arquivo.txt` -- Texto simples, facil de ler
- `seu-arquivo.md` -- Versao formatada, boa para compartilhar
- `seu-arquivo.json` -- Dados estruturados

Abra o arquivo `.txt` para ver sua transcricao com identificacao de speakers e marcacoes de tempo.

### Formatos de Audio Suportados

O Meeting Transcriber funciona com estes tipos de arquivo de audio:

`.wav` `.mp3` `.m4a` `.flac` `.ogg` `.webm` `.aac` `.opus`

O formato `.m4a` e o que iPhones e Macs usam para gravacoes de voz. O formato `.opus` e o que o WhatsApp usa para mensagens de voz.

---

## Modos de Transcricao

O Meeting Transcriber tem tres modos. Pense neles como abordagens diferentes para o mesmo trabalho:

### Qual modo devo usar?

- **"Estou transcrevendo uma reuniao e preciso saber quem falou o que"** -- Use o modo **meeting** (este e o padrao, voce nao precisa adicionar nada):
  ```bash
  python src/transcribe.py reuniao.mp3
  ```

- **"So preciso do texto rapidamente, nao me importo com quem falou"** -- Use o modo **fast**:
  ```bash
  python src/transcribe.py reuniao.mp3 --mode fast
  ```

- **"Preciso do texto rapidamente, mas tambem quero saber quem falou"** -- Use o modo **fast** com a flag `--diarize`:
  ```bash
  python src/transcribe.py reuniao.mp3 --mode fast --diarize
  ```

- **"Precisao e a prioridade, e eu nao me importo de esperar mais"** -- Use o modo **precise**:
  ```bash
  python src/transcribe.py reuniao.mp3 --mode precise
  ```

### Comparacao entre os modos

| | Meeting (padrao) | Fast | Precise |
|---|---|---|---|
| **Velocidade** | Moderada | Muito rapido (10-15x tempo real) | Mais lento |
| **Identifica speakers?** | Sim, sempre | Somente com `--diarize` | Sim, sempre |
| **Precisao** | Muito boa | Boa | A melhor |
| **Memoria necessaria** | ~10 GB | ~4 GB | ~16 GB |
| **Melhor para** | A maioria das reunioes | Rascunhos rapidos, audio com um unico speaker | Gravacoes importantes, juridico/medico |

> **Nota sobre o modo precise:** Ele usa um modelo de IA grande (IBM Granite) que precisa de cerca de 16 GB de RAM. Se o seu Mac tem 8 GB de memoria, fique com o modo **meeting**.

---

## Opcoes Uteis

Aqui estao as opcoes que voce vai usar com mais frequencia:

### Informe o idioma (melhora a precisao)

```bash
python src/transcribe.py reuniao.mp3 --language pt
```

Codigos de idioma mais comuns: `en` (ingles), `pt` (portugues), `es` (espanhol), `fr` (frances), `de` (alemao).

### Informe quantas pessoas estao na gravacao

Se voce sabe que havia exatamente 3 pessoas na reuniao, informar ao programa ajuda a identifica-las com mais precisao:

```bash
python src/transcribe.py reuniao.mp3 --num-speakers 3
```

### Receba uma notificacao quando terminar

Para gravacoes longas, voce pode pedir ao Mac para te avisar quando a transcricao estiver pronta:

```bash
python src/transcribe.py reuniao.mp3 --notify
```

### Escolha quais arquivos de saida voce quer

Por padrao, voce recebe todos os tres formatos (txt, md, json). Se quiser apenas o texto simples:

```bash
python src/transcribe.py reuniao.mp3 --format txt
```

### Veja informacoes detalhadas do progresso

Se algo parecer errado, o modo verbose mostra tudo o que o programa esta fazendo:

```bash
python src/transcribe.py reuniao.mp3 --verbose
```

---

## Formatos de Saida

O Meeting Transcriber produz ate tres arquivos a partir de cada gravacao de audio:

### Texto Simples (.txt)

O formato mais simples. Facil de ler, facil de pesquisar, funciona em qualquer lugar.

```
[00:00] SPEAKER_00: Bom dia a todos, vamos comecar a reuniao.
[00:05] SPEAKER_01: Obrigado pela presenca. Primeiro item da pauta e
        a proposta do cliente.
[00:12] SPEAKER_00: Antes de comecar, alguma atualizacao da semana passada?
```

### Markdown (.md)

Uma versao formatada que fica bonita quando aberta em apps como Notion, Obsidian ou GitHub. Otima para compartilhar.

```markdown
## Transcricao da Reuniao

**[00:00] Speaker 1:** Bom dia a todos, vamos comecar a reuniao.

**[00:05] Speaker 2:** Obrigado pela presenca. Primeiro item da pauta e
a proposta do cliente.

**[00:12] Speaker 1:** Antes de comecar, alguma atualizacao da semana passada?
```

### JSON (.json)

Um formato estruturado que contem todos os detalhes, incluindo horarios exatos de inicio e fim de cada segmento. Util se voce quiser processar os dados com outras ferramentas ou scripts.

```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 4.2,
      "text": "Bom dia a todos, vamos comecar a reuniao.",
      "speaker": "SPEAKER_00"
    }
  ],
  "metadata": {
    "language": "pt",
    "num_speakers": 3
  }
}
```

---

## Vocabulario Customizado

Se suas reunioes usam termos especializados (terminologia medica, jargao juridico, nomes de empresas, siglas), voce pode ensinar o transcritor a reconhece-los. Isso e especialmente util para palavras que a IA pode nao conhecer ou entender errado.

**Como configurar:**

1. Abra o arquivo `vocab/default.txt` (ou crie um baseado no `vocab/default.txt.example`).
2. Adicione seus termos, um por linha. Linhas que comecam com `#` sao ignoradas.

```
# Pessoas
Dr. Silva
Prof. Santos

# Nomes de empresas
Acme Corporation
NovaTech

# Siglas e termos tecnicos
SUS
laparoscopia
amortizacao
```

3. Pronto! O programa carrega automaticamente o `vocab/default.txt` toda vez que roda.

Se voce tem arquivos de vocabulario diferentes para projetos diferentes, pode especificar qual usar:

```bash
python src/transcribe.py reuniao.mp3 --vocab vocab/termos-juridicos.txt
```

---

## Usando IA para Gerar Atas de Reuniao

Uma vez que voce tem uma transcricao, pode usar um assistente de IA (como o Claude) para transforma-la em um documento polido. O Meeting Transcriber inclui templates prontos para os cenarios mais comuns:

### Templates Disponiveis

| Template | O que ele produz | Quando usar |
|----------|-----------------|-------------|
| [Ata de Reuniao](examples/pt/ata_reuniao.md) | Ata formal com pauta, decisoes e itens de acao | Apos qualquer reuniao de equipe |
| [Resumo Executivo](examples/pt/resumo_executivo.md) | Resumo de alto nivel com plano de acao estruturado (metodologia 5W2H) | Para relatorios a lideranca |
| [Itens de Acao](examples/pt/itens_acao.md) | Uma tabela rapida de quem precisa fazer o que, ate quando | Quando voce so precisa da lista de tarefas |

### Como usar

1. **Transcreva sua reuniao:**
   ```bash
   python src/transcribe.py data/audio/reuniao-equipe.mp3
   ```

2. **Abra o Claude Code** (ou qualquer assistente de IA) e faca um pedido como:

   ```
   Leia data/transcripts/reuniao-equipe.txt e gere uma ata formal de reuniao
   seguindo o template em examples/pt/ata_reuniao.md
   ```

3. A IA vai ler sua transcricao e produzir um documento polido e organizado.

### Dicas

- Para melhores resultados, especifique o idioma e o numero de speakers ao transcrever.
- Voce pode customizar os templates para combinar com o formato da sua organizacao -- basta editar os arquivos em `examples/pt/`.
- Se voce criar seus proprios templates, salve-os na pasta `prompts/` (esta pasta nao e compartilhada quando voce atualiza o programa).

---

## Solucao de Problemas

### "No module named 'whisperx'" (ou erros parecidos de modulo nao encontrado)

Nao se preocupe, isso e facil de resolver. Normalmente significa que o ambiente virtual nao esta ativo. Execute estes comandos:

```bash
cd meeting-transcriber
source venv/bin/activate
pip install -r requirements.txt
```

O principal a lembrar: toda vez que voce abre uma nova janela do Terminal, precisa executar `source venv/bin/activate` antes de usar o programa.

### O programa fica sem memoria

Isso significa que o modelo de IA e grande demais para a RAM disponivel. Tente usar um modelo menor:

```bash
python src/transcribe.py reuniao.mp3 --model small
```

O modelo `small` usa cerca de 2 GB de RAM e ainda produz bons resultados. Voce tambem pode fechar outros aplicativos para liberar memoria.

### Speakers nao sao identificados corretamente

A IA faz o melhor para distinguir os speakers, mas nao e perfeita. Voce pode ajuda-la informando quantos speakers havia na gravacao:

```bash
python src/transcribe.py reuniao.mp3 --num-speakers 3
```

Se voce sabe que havia entre 2 e 5 speakers mas nao tem certeza do numero exato:

```bash
python src/transcribe.py reuniao.mp3 --min-speakers 2 --max-speakers 5
```

### O idioma detectado estava errado

Se a transcricao sair no idioma errado, informe ao programa qual idioma esperar:

```bash
python src/transcribe.py reuniao.mp3 --language pt
```

### A transcricao esta demorando muito

Algumas coisas que voce pode tentar:

- Use `--mode fast` se voce nao precisa de identificacao de speakers.
- Use `--model small` para um modelo de IA menor e mais rapido.
- Feche outros aplicativos para liberar memoria e poder de processamento.
- Para gravacoes muito longas (2+ horas), considere dividir o audio em partes menores antes.

### Erro de autenticacao do HuggingFace

Isso significa que o programa nao consegue acessar os modelos de identificacao de speakers. Verifique o seguinte:

1. Abra o arquivo `.env` e certifique-se de que seu token esta la (deve comecar com `hf_`).
2. Certifique-se de que voce visitou **ambas** as paginas abaixo e clicou em "Agree and access repository":
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
3. Verifique se seu token ainda e valido em [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### "command not found: python" ou "command not found: pip"

Certifique-se de que voce ativou o ambiente virtual primeiro:

```bash
cd meeting-transcriber
source venv/bin/activate
```

Se o erro persistir, tente usar `python3.12` em vez de `python`.

---

## Privacidade e Seguranca

**Suas gravacoes e transcricoes nunca saem do seu computador.**

- Todo o processamento acontece localmente no seu Mac. Nenhum audio e enviado para qualquer servidor.
- Apos a configuracao inicial, nenhuma conexao com a internet e necessaria.
- Os modelos de IA sao baixados uma vez e armazenados no seu computador.
- Nao ha telemetria, analytics ou coleta de dados de nenhum tipo.

Isso torna o Meeting Transcriber ideal para:
- **Reunioes empresariais confidenciais** -- discussoes de diretoria, sessoes de estrategia
- **Consultas medicas** -- entrevistas com pacientes, anotacoes clinicas
- **Processos juridicos** -- depoimentos, reunioes com clientes
- **Gravacoes pessoais** -- entrevistas, aulas, notas de voz

---

<details>
<summary><strong>Selecao de Modelo (avancado)</strong></summary>

Um modelo e o cerebro de IA que converte fala em texto. Modelos maiores sao mais precisos, mas mais lentos e usam mais memoria. O padrao (`large-v3`) e a melhor escolha para a maioria das pessoas.

| Modelo | Precisao | Velocidade | RAM Necessaria | Quando usar |
|--------|----------|------------|----------------|-------------|
| tiny | Baixa | Muito rapido | ~1 GB | Apenas para testar se o programa funciona |
| base | Media | Rapido | ~1 GB | Rascunhos rapidos |
| small | Boa | Moderado | ~2 GB | Uso diario quando velocidade importa |
| medium | Muito boa | Lento | ~5 GB | Reunioes importantes onde precisao importa |
| **large-v3** | Excelente | Mais lento | ~10 GB | Melhor qualidade (este e o padrao) |

```bash
# Usar um modelo menor se voce precisa de velocidade ou tem pouca memoria
python src/transcribe.py reuniao.mp3 --model small

# Usar explicitamente o maior modelo para gravacoes importantes
python src/transcribe.py reuniao.mp3 --model large-v3
```

### Modelos adicionais para o modo fast

Ao usar `--mode fast`, voce tem acesso a alguns modelos extras otimizados para Apple Silicon:

| Modelo | Descricao |
|--------|-----------|
| large-v3-turbo | Melhor equilibrio entre velocidade e qualidade |
| distil-large-v3 | Mais rapido, levemente menos preciso |
| large-v3-8bit | Usa menos memoria |

```bash
python src/transcribe.py reuniao.mp3 --mode fast --model large-v3-turbo
```

</details>

<details>
<summary><strong>Referencia Completa de Opcoes</strong></summary>

| Opcao | Forma curta | O que faz | Padrao |
|-------|-------------|-----------|--------|
| `--model` | `-m` | Escolhe o tamanho do modelo de IA (ver Selecao de Modelo acima) | large-v3 |
| `--language` | `-l` | Define o idioma do audio (en, pt, es, fr, de, etc.) | auto-detectar |
| `--num-speakers` | `-n` | Informa o numero exato de speakers | auto-detectar |
| `--min-speakers` | | Numero minimo de speakers esperado | -- |
| `--max-speakers` | | Numero maximo de speakers esperado | -- |
| `--output` | `-o` | Pasta onde as transcricoes sao salvas | data/transcripts |
| `--format` | `-f` | Formato de saida: json, txt, md ou all | all |
| `--mode` | | Abordagem de transcricao: fast, meeting ou precise | meeting |
| `--device` | `-d` | Processador a usar: cpu, cuda ou mps | auto |
| `--notify` | | Mostra uma notificacao do macOS ao terminar | desligado |
| `--vocab` | | Caminho para um arquivo de vocabulario customizado | -- |
| `--ui-lang` | | Idioma da interface: en ou pt | auto |
| `--diarize` | | Identificar speakers no modo fast | desligado |
| `--verbose` | `-v` | Mostra logs detalhados (util para debug) | desligado |

</details>

<details>
<summary><strong>Instalando backends de transcricao adicionais</strong></summary>

A instalacao padrao inclui o backend do modo **meeting**. Se voce quiser usar os outros modos:

**Modo fast** (usa MLX-Whisper, otimizado para Apple Silicon):
```bash
pip install mlx-whisper
```

**Modo precise** (usa IBM Granite, maior precisao):
```bash
pip install transformers accelerate
```

**Instalar tudo de uma vez:**
```bash
pip install mlx-whisper transformers accelerate
```

> **Nota:** O modo precise requer cerca de 16 GB de RAM. Se o seu Mac tem 8 GB de memoria, os modos meeting e fast vao te atender muito bem.

</details>

---

## Agradecimentos

Construido com estes excelentes projetos open-source:

- [WhisperX](https://github.com/m-bain/whisperX) -- Reconhecimento de fala com timestamps por palavra
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) -- Inferencia Whisper otimizada
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) -- Diarizacao de speakers
- [MLX](https://github.com/ml-explore/mlx) -- Framework de machine learning para Apple Silicon
- [IBM Granite Speech](https://huggingface.co/ibm-granite) -- Reconhecimento de fala de alta precisao

---

## Licenca

Licenca MIT -- livre para uso pessoal e comercial.

---

## Autor

**Iuri Almeida**

*Fevereiro de 2026*
