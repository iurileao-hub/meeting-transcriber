# Prompt Templates

Templates for processing transcripts with Claude or other LLMs.

## How to Use

```bash
# 1. Transcribe your audio
python src/transcribe.py meeting.mp3

# 2. Ask Claude to process the transcript using a template
#    Example: "Read data/transcripts/meeting.txt and generate formal
#    meeting minutes following the template in examples/en/meeting_minutes.md"
```

## English Templates

| Template | Purpose |
|----------|---------|
| [en/meeting_minutes.md](en/meeting_minutes.md) | Formal meeting minutes |
| [en/executive_summary.md](en/executive_summary.md) | Executive summary + action plan (5W2H/SMART) |
| [en/action_items.md](en/action_items.md) | Quick action items extraction |

## Templates em Portugues

| Modelo | Uso |
|--------|-----|
| [pt/ata_reuniao.md](pt/ata_reuniao.md) | Ata formal de reuniao |
| [pt/resumo_executivo.md](pt/resumo_executivo.md) | Resumo executivo + plano de acao (5W2H/SMART) |
| [pt/itens_acao.md](pt/itens_acao.md) | Extracao rapida de itens de acao |

## Customization

Save personalized versions in `prompts/` (gitignored):

```bash
mkdir -p prompts
cp examples/en/meeting_minutes.md prompts/my_company_minutes.md
# Edit as needed
```
