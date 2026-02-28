# Modelo de Itens de Ação

Extração rápida de itens de ação de uma transcrição.

## Prompt

```
Leia o arquivo data/transcripts/[NOME_ARQUIVO].txt e extraia todos os ITENS DE AÇÃO.

Formate como tabela:

| # | Ação | Responsável | Prazo | Prioridade | Status |
|---|------|-------------|-------|------------|--------|

Instruções:
- Liste TODAS as tarefas, compromissos ou ações mencionadas
- Inclua compromissos informais ("eu cuido disso")
- Use SPEAKER_XX se nomes não forem identificados
- Marque todos como "Pendente"
- Infira prioridade do contexto (urgência, ênfase)
- Use "A definir" para prazos não definidos
```

## Variações

### Relatório de Acompanhamento

```
Leia a transcrição e gere um RELATÓRIO DE ACOMPANHAMENTO:

1. Ações concluídas (mencionadas como finalizadas)
2. Ações em andamento (com % de progresso se mencionado)
3. Ações pendentes/atrasadas
4. Novos itens surgidos nesta reunião
5. Bloqueios/impedimentos identificados
```

### Apenas Decisões

```
Leia a transcrição e extraia apenas as DECISÕES tomadas:

| # | Decisão | Contexto | Impacto |
|---|---------|----------|---------|
```

## Dicas

- **Reuniões longas**: Peça para dividir a saída por tópico
- **Múltiplos responsáveis**: Liste todos, separados por vírgula
- **Ações implícitas**: Inclua compromissos mesmo que não formalizados
