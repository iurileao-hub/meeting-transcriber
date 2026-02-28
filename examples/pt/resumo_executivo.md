# Modelo de Resumo Executivo

Gera resumo executivo com plano de ação a partir de uma transcrição.

## Prompt

```
Leia o arquivo data/transcripts/[NOME_ARQUIVO].txt e gere um RESUMO EXECUTIVO com PLANO DE AÇÃO.

## Resumo Executivo

Inclua:
- Contexto (1-2 parágrafos: propósito da reunião, relevância dos temas)
- Principais decisões (lista objetiva, uma frase cada)
- Pontos de atenção (riscos, preocupações, pendências)

## Plano de Ação (5W2H)

Para cada ação, preencha uma tabela:
- O QUE (What): Tarefa específica
- POR QUE (Why): Justificativa
- QUEM (Who): Responsável (SPEAKER_XX se desconhecido)
- ONDE (Where): Departamento/sistema
- QUANDO (When): Prazo ou "A definir"
- COMO (How): Abordagem (se discutida)
- QUANTO (How much): Recursos necessários ou "N/A"

Depois forneça:
- Tabela resumo: #, Ação, Responsável, Prazo, Prioridade
- Checklist de validação SMART para metas principais
- Próximos passos com prazos

Instruções:
- Infira prioridades pelo tom e ênfase
- Seja específico — evite ações vagas como "melhorar processo"
- Identifique dependências entre ações
- Use "A definir" para prazos indefinidos e destaque como pendência
```

## Metodologias

**5W2H** — Estrutura cada ação com 7 perguntas para clareza e completude.

**SMART** — Valida se metas são Específicas, Mensuráveis, Alcançáveis, Realistas e Temporais.
