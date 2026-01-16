# Executive Summary Template / Modelo de Resumo Executivo

Generate an executive summary with action plan from a transcript.

Gera resumo executivo com plano de ação a partir de transcrição.

---

## Prompt (English)

```
Read the file data/transcripts/[FILENAME].txt and generate an EXECUTIVE SUMMARY with ACTION PLAN.

## Format

---

# EXECUTIVE SUMMARY

**Meeting:** [Infer title/topic from transcript]
**Date:** [Date if available]
**Participants:** [List identified speakers]

## 1. Context

[1-2 paragraphs explaining the meeting purpose and why the topics are relevant]

## 2. Key Decisions

[List decisions as objective bullets:]
- Decision 1: [One sentence description]
- Decision 2: [One sentence description]

## 3. Points of Attention

[Risks, concerns, or pending items mentioned:]
- [Point 1]
- [Point 2]

---

# ACTION PLAN

## Action Items (5W2H Framework)

For each action identified, fill in:

### Action 1: [Action title]

| Element | Description |
|---------|-------------|
| **WHAT** | [Specific task description] |
| **WHY** | [Justification/objective] |
| **WHO** | [Owner - use SPEAKER_XX if name not identified] |
| **WHERE** | [Location/department/system] |
| **WHEN** | [Specific deadline if mentioned, or "TBD"] |
| **HOW** | [Method/approach if discussed] |
| **HOW MUCH** | [Resources needed if mentioned, or "N/A"] |

[Repeat for each action]

## Action Summary

| # | Action | Owner | Due Date | Priority |
|---|--------|-------|----------|----------|
| 1 | [Summarized action] | [Name] | [Date] | [High/Medium/Low] |

## SMART Validation

For each main goal/action, validate:
- [ ] **S**pecific: Is the action well defined?
- [ ] **M**easurable: Can success be measured?
- [ ] **A**chievable: Is it within scope and resources?
- [ ] **R**ealistic: Is it feasible in the proposed timeline?
- [ ] **T**ime-bound: Is there a defined deadline?

---

# NEXT STEPS

1. [Step 1 with deadline]
2. [Step 2 with deadline]
3. [Next meeting: DATE or "To be scheduled"]

---

## Instructions:

1. **Infer priorities**: Use tone and emphasis to determine High/Medium/Low
2. **Be specific**: Avoid vague actions like "improve process" - detail what needs to be done
3. **Identify dependencies**: If one action depends on another, mention it
4. **Risks**: List concerns mentioned, even without defined solutions
5. **Missing deadlines**: If not mentioned, use "TBD" and highlight as pending
```

---

## Prompt (Português)

```
Leia o arquivo data/transcripts/[NOME_ARQUIVO].txt e gere um RESUMO EXECUTIVO com PLANO DE AÇÃO.

## Formato

---

# RESUMO EXECUTIVO

**Reunião:** [Inferir título/tema da transcrição]
**Data:** [Data se disponível]
**Participantes:** [Listar speakers identificados]

## 1. Contexto

[1-2 parágrafos explicando o propósito da reunião e por que os temas são relevantes]

## 2. Principais Decisões

[Listar decisões em bullets objetivos:]
- Decisão 1: [Descrição em uma frase]
- Decisão 2: [Descrição em uma frase]

## 3. Pontos de Atenção

[Riscos, preocupações ou pendências mencionadas:]
- [Ponto 1]
- [Ponto 2]

---

# PLANO DE AÇÃO

## Itens de Ação (Metodologia 5W2H)

Para cada ação identificada, preencha:

### Ação 1: [Título da ação]

| Elemento | Descrição |
|----------|-----------|
| **O QUÊ** (What) | [Descrição específica da tarefa] |
| **POR QUÊ** (Why) | [Justificativa/objetivo] |
| **QUEM** (Who) | [Responsável - usar SPEAKER_XX se nome não identificado] |
| **ONDE** (Where) | [Local/departamento/sistema] |
| **QUANDO** (When) | [Prazo específico se mencionado, ou "A definir"] |
| **COMO** (How) | [Método/abordagem se discutido] |
| **QUANTO** (How much) | [Recursos necessários se mencionado, ou "N/A"] |

[Repetir para cada ação]

## Resumo das Ações

| # | Ação | Responsável | Prazo | Prioridade |
|---|------|-------------|-------|------------|
| 1 | [Ação resumida] | [Nome] | [Data] | [Alta/Média/Baixa] |

## Validação SMART

Para cada meta/ação principal, validar:
- [ ] E**S**pecífica: A ação está bem definida?
- [ ] **M**ensurável: É possível medir o sucesso?
- [ ] **A**lcançável: Está dentro do escopo e recursos?
- [ ] **R**ealista: É factível no prazo proposto?
- [ ] **T**emporal: Tem prazo definido?

---

# PRÓXIMOS PASSOS

1. [Passo 1 com prazo]
2. [Passo 2 com prazo]
3. [Próxima reunião: DATA ou "A definir"]

---

## Instruções:

1. **Inferir prioridades**: Use tom e ênfase para determinar Alta/Média/Baixa
2. **Ser específico**: Evite ações vagas como "melhorar processo" - detalhe o que deve ser feito
3. **Identificar dependências**: Se uma ação depende de outra, mencione
4. **Riscos**: Liste preocupações mencionadas, mesmo sem solução definida
5. **Prazos ausentes**: Se não mencionado, use "A definir" e destaque como pendência
```

---

## Methodologies Used / Metodologias Utilizadas

### 5W2H

Structures each action with 7 fundamental questions for clarity and completeness.

Estrutura cada ação com 7 perguntas fundamentais para clareza e completude.

### SMART

Validates if goals are Specific, Measurable, Achievable, Realistic, and Time-bound.

Valida se metas são Específicas, Mensuráveis, Alcançáveis, Realistas e Temporais.
