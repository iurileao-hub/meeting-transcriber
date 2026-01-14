# Prompts para Geração de Documentos

Este diretório contém prompts padronizados para usar com Claude Code após a transcrição de reuniões.

## Fluxo de Trabalho

```
1. Transcrever áudio
   python src/transcribe.py reuniao.wav

2. Escolher o prompt adequado
   - ata_sei.md → Ata formal para SEI!
   - resumo_executivo.md → Resumo + Plano de trabalho

3. Usar no Claude Code
   - Abra o arquivo .txt ou .json da transcrição
   - Cole o prompt escolhido
   - Ajuste conforme necessário
```

## Prompts Disponíveis

### `ata_sei.md` - Ata de Reunião (Formato SEI!)

Gera ata formal compatível com o Sistema Eletrônico de Informações.

**Características:**
- Formato institucional brasileiro
- Estrutura numerada (pauta, deliberações, encaminhamentos)
- Pronto para copiar/colar no SEI!
- Linguagem formal em terceira pessoa

**Ideal para:**
- Reuniões de comitês e comissões
- Deliberações que precisam de registro oficial
- Documentação para processos SEI!

---

### `resumo_executivo.md` - Resumo + Plano de Trabalho

Gera resumo executivo com action items estruturados.

**Características:**
- Resumo em 1-2 páginas
- Plano de trabalho com metodologia 5W2H
- Validação SMART para metas
- Tabela de ações com responsáveis e prazos

**Ideal para:**
- Reuniões de planejamento
- Alinhamentos de equipe
- Follow-up de projetos

---

## Exemplo Rápido

Após transcrever `reuniao_gestao.wav`:

```
# No terminal
python src/transcribe.py data/audio/reuniao_gestao.wav --format txt

# No Claude Code
Leia data/transcripts/reuniao_gestao.txt e gere uma ATA DE REUNIÃO
formal seguindo o padrão institucional brasileiro para o SEI!.

[... restante do prompt de ata_sei.md ...]
```

---

## Personalização

### Para a PMDF

Adicione ao cabeçalho dos prompts:

```
POLÍCIA MILITAR DO DISTRITO FEDERAL
DEPARTAMENTO/SEÇÃO: [Seu departamento]
```

### Para Secretarias do GDF

```
GOVERNO DO DISTRITO FEDERAL
SECRETARIA DE ESTADO DE [NOME]
```

### Para outros órgãos

Adapte o cabeçalho conforme o padrão institucional do seu órgão.

---

## Dicas

1. **Use o formato .txt** para prompts simples (mais leve)
2. **Use o formato .json** quando precisar de timestamps precisos
3. **Revise sempre** o output antes de publicar oficialmente
4. **Adapte os prompts** às necessidades específicas do seu órgão
5. **Salve versões customizadas** para tipos recorrentes de reunião

---

## Contribuindo

Para adicionar novos prompts:

1. Crie um arquivo `.md` neste diretório
2. Inclua: descrição, prompt completo, exemplo de uso
3. Documente variações úteis
4. Atualize este README
