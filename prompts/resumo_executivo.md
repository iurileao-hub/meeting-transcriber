# Prompt: Resumo Executivo + Plano de Trabalho

Use este prompt com Claude Code para gerar um resumo executivo e plano de ação a partir de transcrições de reunião.

---

## Prompt

```
Leia o arquivo data/transcripts/[NOME_DO_ARQUIVO].txt (ou .json) e gere um RESUMO EXECUTIVO com PLANO DE TRABALHO.

## Formato do Documento

---

# RESUMO EXECUTIVO

**Reunião:** [Título/Tema inferido da transcrição]
**Data:** [Data se disponível]
**Participantes:** [Listar speakers identificados]

## 1. Contexto

[1-2 parágrafos explicando o propósito da reunião e por que os temas são relevantes. Inferir do conteúdo discutido.]

## 2. Principais Decisões

[Listar as decisões tomadas em bullets objetivos:]
- Decisão 1: [descrição em uma frase]
- Decisão 2: [descrição em uma frase]
- Decisão 3: [descrição em uma frase]

## 3. Pontos de Atenção

[Riscos, preocupações ou pendências mencionadas:]
- [Ponto 1]
- [Ponto 2]

---

# PLANO DE TRABALHO

## Action Items (Metodologia 5W2H + SMART)

Para cada ação identificada na reunião, preencha:

### Ação 1: [Título da ação]

| Elemento | Descrição |
|----------|-----------|
| **O QUE** (What) | [Descrição específica da tarefa] |
| **POR QUE** (Why) | [Justificativa/objetivo] |
| **QUEM** (Who) | [Responsável - usar SPEAKER_XX se nome não identificado] |
| **ONDE** (Where) | [Local/setor/sistema onde será executado] |
| **QUANDO** (When) | [Prazo específico se mencionado, ou "A definir"] |
| **COMO** (How) | [Método/abordagem se discutido] |
| **QUANTO** (How much) | [Recursos necessários se mencionado, ou "N/A"] |

[Repetir para cada ação identificada]

## Resumo das Ações

| # | Ação | Responsável | Prazo | Prioridade |
|---|------|-------------|-------|------------|
| 1 | [Ação resumida] | [Nome] | [Data] | [Alta/Média/Baixa] |
| 2 | [Ação resumida] | [Nome] | [Data] | [Alta/Média/Baixa] |

## Critérios SMART Aplicados

Para cada meta/ação principal, validar:
- [ ] **S**pecífica: A ação está bem definida?
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

## Instruções adicionais:

1. **Inferir prioridades**: Use o tom e ênfase da discussão para determinar Alta/Média/Baixa
2. **Ser específico**: Evite ações vagas como "melhorar processo" - detalhe o que deve ser feito
3. **Identificar dependências**: Se uma ação depende de outra, mencione na descrição
4. **Riscos**: Liste preocupações mencionadas, mesmo que não tenham solução definida
5. **Prazos ausentes**: Se não mencionado, use "A definir" e destaque como pendência
```

---

## Exemplo de Uso

No Claude Code, após transcrever `planejamento_trimestral.wav`:

```
Leia o arquivo data/transcripts/planejamento_trimestral.txt e gere um RESUMO EXECUTIVO com PLANO DE TRABALHO.

[Cole o restante do prompt acima]
```

---

## Variações do Prompt

### Versão Curta (só action items)

```
Leia o arquivo data/transcripts/[ARQUIVO].txt e extraia apenas os ACTION ITEMS no formato:

| Ação | Responsável | Prazo | Status |
|------|-------------|-------|--------|

Liste todas as tarefas, compromissos ou ações mencionadas na reunião.
Marque Status como "Pendente" para todos.
```

### Versão Follow-up (para reunião de acompanhamento)

```
Leia o arquivo data/transcripts/[ARQUIVO].txt e gere um RELATÓRIO DE FOLLOW-UP:

1. Ações concluídas (mencionadas como finalizadas)
2. Ações em andamento (com % de progresso se mencionado)
3. Ações pendentes/atrasadas
4. Novos itens surgidos nesta reunião
5. Impedimentos/bloqueios identificados
```

---

## Metodologias Incluídas

### 5W2H
Estrutura cada ação com 7 perguntas fundamentais para garantir clareza e completude.

### SMART
Valida se as metas são Específicas, Mensuráveis, Alcançáveis, Realistas e Temporais.

### Priorização
Classifica ações em Alta/Média/Baixa baseado no contexto da discussão.

---

## Dicas de Uso

1. **Reuniões longas**: Divida em seções se a transcrição tiver muitos tópicos
2. **Múltiplos responsáveis**: Liste todos separados por vírgula
3. **Ações implícitas**: Inclua compromissos que foram assumidos mas não formalizados
4. **Revisão**: Sempre revise o output antes de distribuir - a IA pode inferir incorretamente
