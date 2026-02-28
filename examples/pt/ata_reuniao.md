# Modelo de Ata de Reuniao

Gera ata formal de reuniao a partir de uma transcricao.

## Prompt

```
Leia o arquivo data/transcripts/[NOME_ARQUIVO].txt e gere uma ATA DE REUNIAO formal.

Inclua:
- Titulo da reuniao (inferido do conteudo), data, horario, local, participantes
- Itens de pauta numerados (inferidos dos topicos discutidos)
- Resumo das discussoes por item (pontos principais, opinioes, questoes)
- Deliberacoes (com responsavel quando disponivel)
- Tabela de encaminhamentos: #, Acao, Responsavel, Prazo, Prioridade
- Proxima reuniao (se mencionada)

Instrucoes:
- Use tom profissional e formal
- Resuma discussoes — nao transcreva literalmente
- Use SPEAKER_XX se nomes reais nao forem identificados
- Inclua compromissos informais como encaminhamentos
- Deixe [PLACEHOLDER] para dados desconhecidos da organizacao
```

## Exemplo de Saida

```
ACME CORPORATION
Equipe de Engenharia

ATA DE REUNIAO - 15 de Janeiro de 2026

Reuniao: Planejamento do Sprint Q1
Data: 15 de Janeiro de 2026
Horario: 10:00 - 11:30
Local: Virtual/Zoom
Participantes: SPEAKER_00 (facilitador), SPEAKER_01, SPEAKER_02

## 1. PAUTA

1. Revisao dos resultados do sprint anterior
2. Discussao de prioridades Q1
3. Alocacao de recursos

## 2. DISCUSSOES

### 1. Revisao dos resultados do sprint anterior
A equipe revisou as tarefas concluidas do sprint anterior...

## 3. DELIBERACOES

- Decisao 1: Adotar novo pipeline CI/CD ate o final do mes

## 4. ENCAMINHAMENTOS

| # | Acao                   | Responsavel | Prazo    | Prioridade |
|---|------------------------|-------------|----------|------------|
| 1 | Configurar pipeline CI | SPEAKER_01  | 30/Jan   | Alta       |

## 5. PROXIMA REUNIAO

- Data: 22 de Janeiro de 2026
- Topicos: Revisao do progresso CI/CD
```
