# Modelo de Itens de Acao

Extracao rapida de itens de acao de uma transcricao.

## Prompt

```
Leia o arquivo data/transcripts/[NOME_ARQUIVO].txt e extraia todos os ITENS DE ACAO.

Formate como tabela:

| # | Acao | Responsavel | Prazo | Prioridade | Status |
|---|------|-------------|-------|------------|--------|

Instrucoes:
- Liste TODAS as tarefas, compromissos ou acoes mencionadas
- Inclua compromissos informais ("eu cuido disso")
- Use SPEAKER_XX se nomes nao forem identificados
- Marque todos como "Pendente"
- Infira prioridade do contexto (urgencia, enfase)
- Use "A definir" para prazos nao definidos
```

## Variacoes

### Relatorio de Acompanhamento

```
Leia a transcricao e gere um RELATORIO DE ACOMPANHAMENTO:

1. Acoes concluidas (mencionadas como finalizadas)
2. Acoes em andamento (com % de progresso se mencionado)
3. Acoes pendentes/atrasadas
4. Novos itens surgidos nesta reuniao
5. Bloqueios/impedimentos identificados
```

### Apenas Decisoes

```
Leia a transcricao e extraia apenas as DECISOES tomadas:

| # | Decisao | Contexto | Impacto |
|---|---------|----------|---------|
```

## Dicas

- **Reunioes longas**: Peca para dividir a saida por topico
- **Multiplos responsaveis**: Liste todos, separados por virgula
- **Acoes implicitas**: Inclua compromissos mesmo que nao formalizados
