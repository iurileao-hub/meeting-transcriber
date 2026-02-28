# Meeting Transcriber

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Plataforma](https://img.shields.io/badge/Plataforma-macOS%20Apple%20Silicon-lightgrey.svg)](https://support.apple.com/pt-br/HT211814)
[![Licença](https://img.shields.io/badge/Licen%C3%A7a-MIT-green.svg)](LICENSE)
[![Testes](https://img.shields.io/badge/testes-289%20passaram-brightgreen.svg)](tests/)
[![Offline](https://img.shields.io/badge/funciona-100%25%20offline-blueviolet.svg)]()

**Transforme gravações de reuniões em transcrições com identificação de quem falou — 100% local, sem nuvem.**

[Read in English](README.md)

---

## O Que Ele Faz

O Meeting Transcriber ouve a gravação de uma reunião (ou entrevista, aula, mensagem de voz, podcast...) e produz um documento escrito que mostra **quem falou o quê, e quando**. Ele identifica as diferentes pessoas automaticamente e marca cada parte da conversa.

Tudo acontece no seu próprio computador. Seus arquivos de áudio nunca são enviados para lugar nenhum.

**Veja como fica o resultado:**

```
[00:00] SPEAKER_00: Bom dia a todos, vamos começar a reunião.
[00:05] SPEAKER_01: Obrigado pela presença. Primeiro item da pauta...
[00:12] SPEAKER_00: Antes de começar, alguma atualização da semana passada?
[00:18] SPEAKER_02: Sim, o cliente aprovou a proposta ontem.
```

Você recebe três arquivos de saída: uma versão em texto simples (fácil de ler), uma versão formatada em Markdown (fica bonita em documentos), e um arquivo JSON (útil se você quiser processar os dados depois).

---

## O Que Você Precisa

Antes de começar, verifique se você tem o seguinte:

| O que | Detalhes |
|-------|----------|
| **Um Mac com Apple Silicon** | Ou seja, chip M1, M2, M3 ou M4. Você pode verificar clicando no menu da Apple e selecionando "Sobre Este Mac". |
| **Cerca de 10 GB de espaço livre em disco** | Os modelos de IA que fazem a transcrição são arquivos grandes. Eles são baixados uma única vez e ficam armazenados no seu computador. |
| **Uma conexão com a internet** | Só é necessária na primeira vez, para baixar o programa e seus modelos. Depois disso, tudo funciona offline. |

---

## Instalação

Esta seção te guia pela configuração do Meeting Transcriber passo a passo. Cada passo inclui uma forma de verificar se funcionou. Se algo der errado, consulte a seção [Solução de Problemas](#solução-de-problemas).

### Passo 1: Instalar o Homebrew (se você ainda não tem)

O Homebrew é uma ferramenta que facilita a instalação de programas no Mac. Abra o aplicativo **Terminal** (você pode encontrá-lo em Aplicativos > Utilitários, ou buscar por "Terminal" no Spotlight) e cole este comando:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**Verifique se funcionou:**
```bash
brew --version
```
Você deve ver um número de versão como `Homebrew 4.x.x`.

### Passo 2: Instalar o Python 3.12

Python é a linguagem de programação na qual esta ferramenta foi escrita. Você precisa especificamente da versão 3.12 (versões mais novas como 3.14 não são compatíveis com algumas das bibliotecas que usamos).

```bash
brew install python@3.12
```

**Verifique se funcionou:**
```bash
python3.12 --version
```
Você deve ver `Python 3.12.x`.

### Passo 3: Instalar o FFmpeg

O FFmpeg é uma ferramenta gratuita que cuida da conversão de arquivos de áudio nos bastidores. O Meeting Transcriber usa-o para ler diferentes formatos de áudio.

```bash
brew install ffmpeg
```

**Verifique se funcionou:**
```bash
ffmpeg -version
```
Você deve ver informações de versão (a primeira linha já é suficiente).

### Passo 4: Baixar o Meeting Transcriber

Este comando baixa o programa para o seu computador:

```bash
git clone https://github.com/iurileao-hub/meeting-transcriber.git
cd meeting-transcriber
```

### Passo 5: Criar um ambiente virtual

Um ambiente virtual é como uma pasta separada onde o programa e todos os seus arquivos vivem, sem afetar o resto do seu computador. Isso mantém tudo organizado e evita conflitos com outros programas.

```bash
python3.12 -m venv venv
source venv/bin/activate
```

Após executar o segundo comando, você deve ver `(venv)` no início da linha do seu terminal. Isso significa que o ambiente virtual está ativo.

> **Importante:** Toda vez que você abrir uma nova janela do Terminal para usar o Meeting Transcriber, você precisa ativar o ambiente virtual novamente:
> ```bash
> cd meeting-transcriber
> source venv/bin/activate
> ```

### Passo 6: Instalar as dependências do programa

Dependências são as bibliotecas e ferramentas que o Meeting Transcriber precisa para funcionar. Este comando baixa e instala todas elas:

```bash
pip install -r requirements.txt
```

Isso pode levar alguns minutos. Você verá muitas linhas rolando na tela — isso é normal.

**Verifique se funcionou:**
```bash
python -c "import whisperx; print('OK')"
```
Você deve ver `OK`.

### Passo 7: Configurar o HuggingFace (necessário para identificação de speakers)

O HuggingFace é um site que hospeda modelos de IA gratuitos. O Meeting Transcriber precisa de acesso a dois desses modelos para distinguir quem está falando. Isso é completamente gratuito.

**7a. Crie uma conta gratuita:**
- Acesse [huggingface.co/join](https://huggingface.co/join) e faça seu cadastro.

**7b. Obtenha seu token de acesso:**

Um token é uma senha especial que permite ao programa baixar modelos de IA do HuggingFace.

- Vá em [Settings > Access Tokens](https://huggingface.co/settings/tokens)
- Clique em "New token", dê qualquer nome (como "meeting-transcriber"), e clique em Create.
- Copie o token (ele começa com `hf_`).

**7c. Salve o token:**
```bash
cp .env.example .env
```

Agora abra o arquivo `.env` em qualquer editor de texto e substitua `hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx` pelo token que você acabou de copiar. Salve o arquivo.

**7d. Aceite os termos dos modelos de IA (só precisa fazer uma vez):**

Você precisa visitar duas páginas e clicar em "Agree and access repository" em cada uma:

1. [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) -- clique em "Agree and access repository"
2. [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) -- clique em "Agree and access repository"

Pronto! Você só precisa fazer isso uma vez.

---

## Sua Primeira Transcrição

Vamos garantir que tudo está funcionando. Coloque um arquivo de áudio (qualquer `.mp3`, `.wav`, `.m4a` ou outro [formato suportado](#formatos-de-áudio-suportados)) dentro da pasta `data/audio/` e execute:

```bash
python src/transcribe.py data/audio/seu-arquivo.mp3
```

O programa vai mostrar uma barra de progresso enquanto trabalha em várias etapas: carregando o modelo de IA, detectando fala, transcrevendo e identificando speakers. Dependendo da duração do seu áudio e do tamanho do modelo, isso pode levar de alguns segundos a vários minutos.

Quando terminar, você vai encontrar suas transcrições na pasta `data/transcripts/`:
- `seu-arquivo.txt` -- Texto simples, fácil de ler
- `seu-arquivo.md` -- Versão formatada, boa para compartilhar
- `seu-arquivo.json` -- Dados estruturados

Abra o arquivo `.txt` para ver sua transcrição com identificação de speakers e marcações de tempo.

### Formatos de Áudio Suportados

O Meeting Transcriber funciona com estes tipos de arquivo de áudio:

`.wav` `.mp3` `.m4a` `.flac` `.ogg` `.webm` `.aac` `.opus`

O formato `.m4a` é o que iPhones e Macs usam para gravações de voz. O formato `.opus` é o que o WhatsApp usa para mensagens de voz.

---

## Modos de Transcrição

O Meeting Transcriber tem três modos. Pense neles como abordagens diferentes para o mesmo trabalho:

### Qual modo devo usar?

- **"Estou transcrevendo uma reunião e preciso saber quem falou o quê"** -- Use o modo **meeting** (este é o padrão, você não precisa adicionar nada):
  ```bash
  python src/transcribe.py reuniao.mp3
  ```

- **"Só preciso do texto rapidamente, não me importo com quem falou"** -- Use o modo **fast**:
  ```bash
  python src/transcribe.py reuniao.mp3 --mode fast
  ```

- **"Preciso do texto rapidamente, mas também quero saber quem falou"** -- Use o modo **fast** com a flag `--diarize`:
  ```bash
  python src/transcribe.py reuniao.mp3 --mode fast --diarize
  ```

- **"Precisão é a prioridade, e eu não me importo de esperar mais"** -- Use o modo **precise**:
  ```bash
  python src/transcribe.py reuniao.mp3 --mode precise
  ```

### Comparação entre os modos

| | Meeting (padrão) | Fast | Precise |
|---|---|---|---|
| **Velocidade** | Moderada | Muito rápido (10-15x tempo real) | Mais lento |
| **Identifica speakers?** | Sim, sempre | Somente com `--diarize` | Sim, sempre |
| **Precisão** | Muito boa | Boa | A melhor |
| **Memória necessária** | ~10 GB | ~4 GB | ~16 GB |
| **Melhor para** | A maioria das reuniões | Rascunhos rápidos, áudio com um único speaker | Gravações importantes, jurídico/médico |

> **Nota sobre o modo precise:** Ele usa um modelo de IA grande (IBM Granite) que precisa de cerca de 16 GB de RAM. Se o seu Mac tem 8 GB de memória, fique com o modo **meeting**.

---

## Opções Úteis

Aqui estão as opções que você vai usar com mais frequência:

### Informe o idioma (melhora a precisão)

```bash
python src/transcribe.py reuniao.mp3 --language pt
```

Códigos de idioma mais comuns: `en` (inglês), `pt` (português), `es` (espanhol), `fr` (francês), `de` (alemão).

### Informe quantas pessoas estão na gravação

Se você sabe que havia exatamente 3 pessoas na reunião, informar ao programa ajuda a identificá-las com mais precisão:

```bash
python src/transcribe.py reuniao.mp3 --num-speakers 3
```

### Receba uma notificação quando terminar

Para gravações longas, você pode pedir ao Mac para te avisar quando a transcrição estiver pronta:

```bash
python src/transcribe.py reuniao.mp3 --notify
```

### Escolha quais arquivos de saída você quer

Por padrão, você recebe todos os três formatos (txt, md, json). Se quiser apenas o texto simples:

```bash
python src/transcribe.py reuniao.mp3 --format txt
```

### Veja informações detalhadas do progresso

Se algo parecer errado, o modo verbose mostra tudo o que o programa está fazendo:

```bash
python src/transcribe.py reuniao.mp3 --verbose
```

---

## Formatos de Saída

O Meeting Transcriber produz até três arquivos a partir de cada gravação de áudio:

### Texto Simples (.txt)

O formato mais simples. Fácil de ler, fácil de pesquisar, funciona em qualquer lugar.

```
[00:00] SPEAKER_00: Bom dia a todos, vamos começar a reunião.
[00:05] SPEAKER_01: Obrigado pela presença. Primeiro item da pauta é
        a proposta do cliente.
[00:12] SPEAKER_00: Antes de começar, alguma atualização da semana passada?
```

### Markdown (.md)

Uma versão formatada que fica bonita quando aberta em apps como Notion, Obsidian ou GitHub. Ótima para compartilhar.

```markdown
## Transcrição da Reunião

**[00:00] Speaker 1:** Bom dia a todos, vamos começar a reunião.

**[00:05] Speaker 2:** Obrigado pela presença. Primeiro item da pauta é
a proposta do cliente.

**[00:12] Speaker 1:** Antes de começar, alguma atualização da semana passada?
```

### JSON (.json)

Um formato estruturado que contém todos os detalhes, incluindo horários exatos de início e fim de cada segmento. Útil se você quiser processar os dados com outras ferramentas ou scripts.

```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 4.2,
      "text": "Bom dia a todos, vamos começar a reunião.",
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

## Vocabulário Customizado

Se suas reuniões usam termos especializados (terminologia médica, jargão jurídico, nomes de empresas, siglas), você pode ensinar o transcritor a reconhecê-los. Isso é especialmente útil para palavras que a IA pode não conhecer ou entender errado.

**Como configurar:**

1. Abra o arquivo `vocab/default.txt` (ou crie um baseado no `vocab/default.txt.example`).
2. Adicione seus termos, um por linha. Linhas que começam com `#` são ignoradas.

```
# Pessoas
Dr. Silva
Prof. Santos

# Nomes de empresas
Acme Corporation
NovaTech

# Siglas e termos técnicos
SUS
laparoscopia
amortização
```

3. Pronto! O programa carrega automaticamente o `vocab/default.txt` toda vez que roda.

Se você tem arquivos de vocabulário diferentes para projetos diferentes, pode especificar qual usar:

```bash
python src/transcribe.py reuniao.mp3 --vocab vocab/termos-juridicos.txt
```

---

## Usando IA para Gerar Atas de Reunião

Uma vez que você tem uma transcrição, pode usar um assistente de IA (como o Claude) para transformá-la em um documento polido. O Meeting Transcriber inclui templates prontos para os cenários mais comuns:

### Templates Disponíveis

| Template | O que ele produz | Quando usar |
|----------|-----------------|-------------|
| [Ata de Reunião](examples/pt/ata_reuniao.md) | Ata formal com pauta, decisões e itens de ação | Após qualquer reunião de equipe |
| [Resumo Executivo](examples/pt/resumo_executivo.md) | Resumo de alto nível com plano de ação estruturado (metodologia 5W2H) | Para relatórios à liderança |
| [Itens de Ação](examples/pt/itens_acao.md) | Uma tabela rápida de quem precisa fazer o quê, até quando | Quando você só precisa da lista de tarefas |

### Como usar

1. **Transcreva sua reunião:**
   ```bash
   python src/transcribe.py data/audio/reuniao-equipe.mp3
   ```

2. **Abra o Claude Code** (ou qualquer assistente de IA) e faça um pedido como:

   ```
   Leia data/transcripts/reuniao-equipe.txt e gere uma ata formal de reunião
   seguindo o template em examples/pt/ata_reuniao.md
   ```

3. A IA vai ler sua transcrição e produzir um documento polido e organizado.

### Dicas

- Para melhores resultados, especifique o idioma e o número de speakers ao transcrever.
- Você pode customizar os templates para combinar com o formato da sua organização -- basta editar os arquivos em `examples/pt/`.
- Se você criar seus próprios templates, salve-os na pasta `prompts/` (esta pasta não é compartilhada quando você atualiza o programa).

---

## Solução de Problemas

### "No module named 'whisperx'" (ou erros parecidos de módulo não encontrado)

Não se preocupe, isso é fácil de resolver. Normalmente significa que o ambiente virtual não está ativo. Execute estes comandos:

```bash
cd meeting-transcriber
source venv/bin/activate
pip install -r requirements.txt
```

O principal a lembrar: toda vez que você abre uma nova janela do Terminal, precisa executar `source venv/bin/activate` antes de usar o programa.

### O programa fica sem memória

Isso significa que o modelo de IA é grande demais para a RAM disponível. Tente usar um modelo menor:

```bash
python src/transcribe.py reuniao.mp3 --model small
```

O modelo `small` usa cerca de 2 GB de RAM e ainda produz bons resultados. Você também pode fechar outros aplicativos para liberar memória.

### Speakers não são identificados corretamente

A IA faz o melhor para distinguir os speakers, mas não é perfeita. Você pode ajudá-la informando quantos speakers havia na gravação:

```bash
python src/transcribe.py reuniao.mp3 --num-speakers 3
```

Se você sabe que havia entre 2 e 5 speakers mas não tem certeza do número exato:

```bash
python src/transcribe.py reuniao.mp3 --min-speakers 2 --max-speakers 5
```

### O idioma detectado estava errado

Se a transcrição sair no idioma errado, informe ao programa qual idioma esperar:

```bash
python src/transcribe.py reuniao.mp3 --language pt
```

### A transcrição está demorando muito

Algumas coisas que você pode tentar:

- Use `--mode fast` se você não precisa de identificação de speakers.
- Use `--model small` para um modelo de IA menor e mais rápido.
- Feche outros aplicativos para liberar memória e poder de processamento.
- Para gravações muito longas (2+ horas), considere dividir o áudio em partes menores antes.

### Erro de autenticação do HuggingFace

Isso significa que o programa não consegue acessar os modelos de identificação de speakers. Verifique o seguinte:

1. Abra o arquivo `.env` e certifique-se de que seu token está lá (deve começar com `hf_`).
2. Certifique-se de que você visitou **ambas** as páginas abaixo e clicou em "Agree and access repository":
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
3. Verifique se seu token ainda é válido em [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### "command not found: python" ou "command not found: pip"

Certifique-se de que você ativou o ambiente virtual primeiro:

```bash
cd meeting-transcriber
source venv/bin/activate
```

Se o erro persistir, tente usar `python3.12` em vez de `python`.

---

## Privacidade e Segurança

**Suas gravações e transcrições nunca saem do seu computador.**

- Todo o processamento acontece localmente no seu Mac. Nenhum áudio é enviado para qualquer servidor.
- Após a configuração inicial, nenhuma conexão com a internet é necessária.
- Os modelos de IA são baixados uma vez e armazenados no seu computador.
- Não há telemetria, analytics ou coleta de dados de nenhum tipo.

Isso torna o Meeting Transcriber ideal para:
- **Reuniões empresariais confidenciais** -- discussões de diretoria, sessões de estratégia
- **Consultas médicas** -- entrevistas com pacientes, anotações clínicas
- **Processos jurídicos** -- depoimentos, reuniões com clientes
- **Gravações pessoais** -- entrevistas, aulas, notas de voz

---

<details>
<summary><strong>Seleção de Modelo (avançado)</strong></summary>

Um modelo é o cérebro de IA que converte fala em texto. Modelos maiores são mais precisos, mas mais lentos e usam mais memória. O padrão (`large-v3`) é a melhor escolha para a maioria das pessoas.

| Modelo | Precisão | Velocidade | RAM Necessária | Quando usar |
|--------|----------|------------|----------------|-------------|
| tiny | Baixa | Muito rápido | ~1 GB | Apenas para testar se o programa funciona |
| base | Média | Rápido | ~1 GB | Rascunhos rápidos |
| small | Boa | Moderado | ~2 GB | Uso diário quando velocidade importa |
| medium | Muito boa | Lento | ~5 GB | Reuniões importantes onde precisão importa |
| **large-v3** | Excelente | Mais lento | ~10 GB | Melhor qualidade (este é o padrão) |

```bash
# Usar um modelo menor se você precisa de velocidade ou tem pouca memória
python src/transcribe.py reuniao.mp3 --model small

# Usar explicitamente o maior modelo para gravações importantes
python src/transcribe.py reuniao.mp3 --model large-v3
```

### Modelos adicionais para o modo fast

Ao usar `--mode fast`, você tem acesso a alguns modelos extras otimizados para Apple Silicon:

| Modelo | Descrição |
|--------|-----------|
| large-v3-turbo | Melhor equilíbrio entre velocidade e qualidade |
| distil-large-v3 | Mais rápido, levemente menos preciso |
| large-v3-8bit | Usa menos memória |

```bash
python src/transcribe.py reuniao.mp3 --mode fast --model large-v3-turbo
```

</details>

<details>
<summary><strong>Referência Completa de Opções</strong></summary>

| Opção | Forma curta | O que faz | Padrão |
|-------|-------------|-----------|--------|
| `--model` | `-m` | Escolhe o tamanho do modelo de IA (ver Seleção de Modelo acima) | large-v3 |
| `--language` | `-l` | Define o idioma do áudio (en, pt, es, fr, de, etc.) | auto-detectar |
| `--num-speakers` | `-n` | Informa o número exato de speakers | auto-detectar |
| `--min-speakers` | | Número mínimo de speakers esperado | -- |
| `--max-speakers` | | Número máximo de speakers esperado | -- |
| `--output` | `-o` | Pasta onde as transcrições são salvas | data/transcripts |
| `--format` | `-f` | Formato de saída: json, txt, md ou all | all |
| `--mode` | | Abordagem de transcrição: fast, meeting ou precise | meeting |
| `--device` | `-d` | Processador a usar: cpu, cuda ou mps | auto |
| `--notify` | | Mostra uma notificação do macOS ao terminar | desligado |
| `--vocab` | | Caminho para um arquivo de vocabulário customizado | -- |
| `--ui-lang` | | Idioma da interface: en ou pt | auto |
| `--diarize` | | Identificar speakers no modo fast | desligado |
| `--verbose` | `-v` | Mostra logs detalhados (útil para debug) | desligado |

</details>

<details>
<summary><strong>Instalando backends de transcrição adicionais</strong></summary>

A instalação padrão inclui o backend do modo **meeting**. Se você quiser usar os outros modos:

**Modo fast** (usa MLX-Whisper, otimizado para Apple Silicon):
```bash
pip install mlx-whisper
```

**Modo precise** (usa IBM Granite, maior precisão):
```bash
pip install transformers accelerate
```

**Instalar tudo de uma vez:**
```bash
pip install mlx-whisper transformers accelerate
```

> **Nota:** O modo precise requer cerca de 16 GB de RAM. Se o seu Mac tem 8 GB de memória, os modos meeting e fast vão te atender muito bem.

</details>

---

## Agradecimentos

Construído com estes excelentes projetos open-source:

- [WhisperX](https://github.com/m-bain/whisperX) -- Reconhecimento de fala com timestamps por palavra
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) -- Inferência Whisper otimizada
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) -- Diarização de speakers
- [MLX](https://github.com/ml-explore/mlx) -- Framework de machine learning para Apple Silicon
- [IBM Granite Speech](https://huggingface.co/ibm-granite) -- Reconhecimento de fala de alta precisão

---

## Licença

Licença MIT -- livre para uso pessoal e comercial.

---

## Autor

**Iúri Almeida**

*Fevereiro de 2026*
