# Action Items Template / Modelo de Itens de Ação

Quick extraction of action items from a transcript.

Extração rápida de itens de ação de uma transcrição.

---

## Prompt (English)

```
Read the file data/transcripts/[FILENAME].txt and extract all ACTION ITEMS.

## Format

# ACTION ITEMS - [Meeting Date/Topic]

| # | Action | Owner | Due Date | Priority | Status |
|---|--------|-------|----------|----------|--------|
| 1 | [Task description] | [Name/SPEAKER_XX] | [Date/TBD] | [H/M/L] | Pending |

## Instructions:

1. List ALL tasks, commitments, or actions mentioned
2. Include informal commitments ("I'll take care of that")
3. Use SPEAKER_XX if names aren't identified
4. Mark all as "Pending" status
5. Infer priority from context (urgency, emphasis)
6. Use "TBD" for undefined deadlines
```

---

## Prompt (Português)

```
Leia o arquivo data/transcripts/[NOME_ARQUIVO].txt e extraia todos os ITENS DE AÇÃO.

## Formato

# ITENS DE AÇÃO - [Data/Tema da Reunião]

| # | Ação | Responsável | Prazo | Prioridade | Status |
|---|------|-------------|-------|------------|--------|
| 1 | [Descrição da tarefa] | [Nome/SPEAKER_XX] | [Data/A definir] | [A/M/B] | Pendente |

## Instruções:

1. Liste TODAS as tarefas, compromissos ou ações mencionadas
2. Inclua compromissos informais ("eu cuido disso")
3. Use SPEAKER_XX se nomes não forem identificados
4. Marque todos como status "Pendente"
5. Infira prioridade do contexto (urgência, ênfase)
6. Use "A definir" para prazos não definidos
```

---

## Variations / Variações

### Follow-up Report / Relatório de Acompanhamento

```
Read the transcript and generate a FOLLOW-UP REPORT:

1. Completed actions (mentioned as finished)
2. In-progress actions (with % progress if mentioned)
3. Pending/overdue actions
4. New items from this meeting
5. Blockers/impediments identified

---

Leia a transcrição e gere um RELATÓRIO DE ACOMPANHAMENTO:

1. Ações concluídas (mencionadas como finalizadas)
2. Ações em andamento (com % de progresso se mencionado)
3. Ações pendentes/atrasadas
4. Novos itens surgidos nesta reunião
5. Bloqueios/impedimentos identificados
```

### Decisions Only / Apenas Decisões

```
Read the transcript and extract only DECISIONS made:

| # | Decision | Context | Impact |
|---|----------|---------|--------|

---

Leia a transcrição e extraia apenas as DECISÕES tomadas:

| # | Decisão | Contexto | Impacto |
|---|---------|----------|---------|
```

---

## Tips / Dicas

- **Long meetings**: Split output by topic / Reuniões longas: divida por tópico
- **Multiple owners**: List all, separated by comma / Múltiplos responsáveis: liste todos, separados por vírgula
- **Implicit actions**: Include commitments even if not formalized / Ações implícitas: inclua compromissos mesmo que não formalizados
