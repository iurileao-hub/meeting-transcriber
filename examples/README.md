# Prompt Templates / Modelos de Prompt

Templates for post-processing transcriptions with Claude or other LLMs.

Modelos para pós-processamento de transcrições com Claude ou outros LLMs.

---

## Workflow / Fluxo de Trabalho

```bash
# 1. Transcribe / Transcrever
python src/transcribe.py meeting.wav

# 2. Use template with Claude / Usar modelo com Claude
#    Copy the appropriate template and paste with your transcript
#    Copie o modelo apropriado e cole com sua transcrição
```

---

## Available Templates / Modelos Disponíveis

| Template | Purpose / Propósito |
|----------|---------------------|
| [meeting_minutes.md](meeting_minutes.md) | Formal meeting minutes / Ata formal de reunião |
| [executive_summary.md](executive_summary.md) | Summary + Action plan / Resumo + Plano de ação |
| [action_items.md](action_items.md) | Quick action items extraction / Extração rápida de tarefas |

---

## Usage / Uso

### With Claude Code / Com Claude Code

```
Read data/transcripts/meeting.txt and generate formal meeting minutes
following the template in examples/meeting_minutes.md

Leia data/transcripts/reuniao.txt e gere uma ata formal de reunião
seguindo o modelo em examples/meeting_minutes.md
```

### With Claude.ai / Com Claude.ai

1. Copy the transcript content / Copie o conteúdo da transcrição
2. Copy the template prompt / Copie o prompt do modelo
3. Paste both in Claude / Cole ambos no Claude

---

## Customization / Personalização

These templates are generic starting points. Customize for your organization:

Estes modelos são pontos de partida genéricos. Personalize para sua organização:

- Add your company/organization header / Adicione cabeçalho da sua empresa
- Adjust terminology / Ajuste a terminologia
- Add specific sections / Adicione seções específicas
- Save customized versions in `prompts/` (not versioned) / Salve versões customizadas em `prompts/` (não versionado)

---

## Creating Custom Templates / Criando Modelos Personalizados

Save your customized templates in the `prompts/` folder (gitignored).

Salve seus modelos personalizados na pasta `prompts/` (não versionada).

```bash
# Create personal prompts folder / Criar pasta de prompts pessoais
mkdir -p prompts
cp examples/meeting_minutes.md prompts/my_company_minutes.md
# Edit as needed / Edite conforme necessário
```
