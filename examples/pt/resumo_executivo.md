# Modelo de Resumo Executivo

Gera resumo executivo com plano de acao a partir de uma transcricao.

## Prompt

```
Leia o arquivo data/transcripts/[NOME_ARQUIVO].txt e gere um RESUMO EXECUTIVO com PLANO DE ACAO.

## Resumo Executivo

Inclua:
- Contexto (1-2 paragrafos: proposito da reuniao, relevancia dos temas)
- Principais decisoes (lista objetiva, uma frase cada)
- Pontos de atencao (riscos, preocupacoes, pendencias)

## Plano de Acao (5W2H)

Para cada acao, preencha uma tabela:
- O QUE (What): Tarefa especifica
- POR QUE (Why): Justificativa
- QUEM (Who): Responsavel (SPEAKER_XX se desconhecido)
- ONDE (Where): Departamento/sistema
- QUANDO (When): Prazo ou "A definir"
- COMO (How): Abordagem (se discutida)
- QUANTO (How much): Recursos necessarios ou "N/A"

Depois forneca:
- Tabela resumo: #, Acao, Responsavel, Prazo, Prioridade
- Checklist de validacao SMART para metas principais
- Proximos passos com prazos

Instrucoes:
- Infira prioridades pelo tom e enfase
- Seja especifico — evite acoes vagas como "melhorar processo"
- Identifique dependencias entre acoes
- Use "A definir" para prazos indefinidos e destaque como pendencia
```

## Metodologias

**5W2H** — Estrutura cada acao com 7 perguntas para clareza e completude.

**SMART** — Valida se metas sao Especificas, Mensuraveis, Alcancaveis, Realistas e Temporais.
