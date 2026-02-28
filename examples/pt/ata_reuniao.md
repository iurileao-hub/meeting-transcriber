# Modelo de Ata de Reunião

Gera ata formal de reunião a partir de uma transcrição.

## Prompt

```
Leia o arquivo data/transcripts/[NOME_ARQUIVO].txt e gere uma ATA DE REUNIÃO formal.

Inclua:
- Título da reunião (inferido do conteúdo), data, horário, local, participantes
- Itens de pauta numerados (inferidos dos tópicos discutidos)
- Resumo das discussões por item (pontos principais, opiniões, questões)
- Deliberações (com responsável quando disponível)
- Tabela de encaminhamentos: #, Ação, Responsável, Prazo, Prioridade
- Próxima reunião (se mencionada)

Instruções:
- Use tom profissional e formal
- Resuma discussões — não transcreva literalmente
- Use SPEAKER_XX se nomes reais não forem identificados
- Inclua compromissos informais como encaminhamentos
- Deixe [PLACEHOLDER] para dados desconhecidos da organização
```

## Exemplo de Saída

```
ACME CORPORATION
Equipe de Engenharia

ATA DE REUNIÃO - 15 de Janeiro de 2026

Reunião: Planejamento do Sprint Q1
Data: 15 de Janeiro de 2026
Horário: 10:00 - 11:30
Local: Virtual/Zoom
Participantes: SPEAKER_00 (facilitador), SPEAKER_01, SPEAKER_02

## 1. PAUTA

1. Revisão dos resultados do sprint anterior
2. Discussão de prioridades Q1
3. Alocação de recursos

## 2. DISCUSSÕES

### 1. Revisão dos resultados do sprint anterior
A equipe revisou as tarefas concluídas do sprint anterior...

## 3. DELIBERAÇÕES

- Decisão 1: Adotar novo pipeline CI/CD até o final do mês

## 4. ENCAMINHAMENTOS

| # | Ação                   | Responsável | Prazo    | Prioridade |
|---|------------------------|-------------|----------|------------|
| 1 | Configurar pipeline CI | SPEAKER_01  | 30/Jan   | Alta       |

## 5. PRÓXIMA REUNIÃO

- Data: 22 de Janeiro de 2026
- Tópicos: Revisão do progresso CI/CD
```
