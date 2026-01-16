# Meeting Minutes Template / Modelo de Ata de Reunião

Generate formal meeting minutes from a transcript.

Gera ata formal de reunião a partir de transcrição.

---

## Prompt (English)

```
Read the file data/transcripts/[FILENAME].txt and generate formal MEETING MINUTES.

## Format

---

[ORGANIZATION NAME]
[DEPARTMENT/TEAM - if applicable]

MEETING MINUTES - [DATE]

**Meeting:** [Infer title/topic from transcript]
**Date:** [Date if available]
**Time:** [Start time - End time]
**Location:** [Physical location or "Virtual/Video call"]
**Attendees:** [List identified participants]

---

## 1. AGENDA

[Infer main topics discussed and number them]

## 2. DISCUSSION

[For each agenda item, summarize:
- Key points discussed
- Different viewpoints presented
- Questions raised]

## 3. DECISIONS

[List decisions made:]
- Decision 1: [Description]
- Decision 2: [Description]

## 4. ACTION ITEMS

| # | Action | Owner | Due Date | Priority |
|---|--------|-------|----------|----------|
| 1 | [Task description] | [Name/SPEAKER_XX] | [Date or TBD] | [High/Medium/Low] |

## 5. NEXT MEETING

- Date: [If mentioned, or "To be scheduled"]
- Topics: [If mentioned]

---

## Instructions:

1. **Formal tone**: Use professional language
2. **Be objective**: Summarize discussions, don't transcribe literally
3. **Clear decisions**: Each decision should have clear ownership when available
4. **Infer context**: If organization isn't mentioned, leave [ORGANIZATION] as placeholder
5. **Speaker names**: Use SPEAKER_XX if real names aren't identified
6. **Action items**: Include any commitments made, even if informal
```

---

## Prompt (Português)

```
Leia o arquivo data/transcripts/[NOME_ARQUIVO].txt e gere uma ATA DE REUNIÃO formal.

## Formato

---

[NOME DA ORGANIZAÇÃO]
[DEPARTAMENTO/EQUIPE - se aplicável]

ATA DE REUNIÃO - [DATA]

**Reunião:** [Inferir título/tema da transcrição]
**Data:** [Data se disponível]
**Horário:** [Início - Término]
**Local:** [Local físico ou "Virtual/Videoconferência"]
**Participantes:** [Listar participantes identificados]

---

## 1. PAUTA

[Inferir principais tópicos discutidos e numerá-los]

## 2. DISCUSSÕES

[Para cada item da pauta, resumir:
- Pontos principais discutidos
- Diferentes opiniões apresentadas
- Questões levantadas]

## 3. DELIBERAÇÕES

[Listar decisões tomadas:]
- Decisão 1: [Descrição]
- Decisão 2: [Descrição]

## 4. ENCAMINHAMENTOS

| # | Ação | Responsável | Prazo | Prioridade |
|---|------|-------------|-------|------------|
| 1 | [Descrição da tarefa] | [Nome/SPEAKER_XX] | [Data ou A definir] | [Alta/Média/Baixa] |

## 5. PRÓXIMA REUNIÃO

- Data: [Se mencionado, ou "A agendar"]
- Tópicos: [Se mencionados]

---

## Instruções:

1. **Tom formal**: Use linguagem profissional
2. **Seja objetivo**: Resuma discussões, não transcreva literalmente
3. **Decisões claras**: Cada decisão deve ter responsável quando disponível
4. **Inferir contexto**: Se a organização não for mencionada, deixe [ORGANIZAÇÃO] como placeholder
5. **Nomes dos speakers**: Use SPEAKER_XX se nomes reais não forem identificados
6. **Encaminhamentos**: Inclua compromissos assumidos, mesmo que informais
```

---

## Example Output / Exemplo de Saída

```
ACME CORPORATION
Engineering Team

MEETING MINUTES - January 15, 2026

**Meeting:** Sprint Planning Q1
**Date:** January 15, 2026
**Time:** 10:00 - 11:30
**Location:** Virtual/Zoom
**Attendees:** SPEAKER_00 (facilitator), SPEAKER_01, SPEAKER_02

---

## 1. AGENDA

1. Review of previous sprint results
2. Q1 priorities discussion
3. Resource allocation
4. Timeline definition

## 2. DISCUSSION

### 1. Review of previous sprint results
The team reviewed the completed tasks from the previous sprint...

[...]
```
