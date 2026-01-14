# Prompt: Ata de Reunião (Formato SEI!)

Use este prompt com Claude Code para gerar uma ata formal compatível com o Sistema Eletrônico de Informações (SEI!).

---

## Prompt

```
Leia o arquivo data/transcripts/[NOME_DO_ARQUIVO].txt (ou .json) e gere uma ATA DE REUNIÃO formal seguindo o padrão institucional brasileiro para o SEI!.

## Formato da Ata

Use a seguinte estrutura:

---

GOVERNO DO DISTRITO FEDERAL
[ÓRGÃO/SECRETARIA - preencher conforme contexto]
[COMITÊ/COMISSÃO - se aplicável]

ATA DA [Nº]ª REUNIÃO [ORDINÁRIA/EXTRAORDINÁRIA]

Aos [DATA POR EXTENSO], às [HORÁRIO], reuniram-se [presencialmente/por videoconferência], no [LOCAL/PLATAFORMA], os participantes abaixo identificados, para deliberar sobre os assuntos constantes da pauta.

1. PARTICIPANTES

1.1 Presentes:
[Listar nomes identificados na transcrição, com cargos se mencionados]

1.2 Ausentes:
[Se mencionado, listar. Caso contrário: "Não houve ausências registradas."]

2. PAUTA

[Inferir da transcrição os principais tópicos discutidos, numerando-os]

3. DELIBERAÇÕES

[Para cada item da pauta, descrever objetivamente:
- O que foi discutido
- O que foi decidido
- Quem ficou responsável (se mencionado)
- Prazo definido (se mencionado)]

4. ENCAMINHAMENTOS

[Listar ações definidas no formato:]
| Ação | Responsável | Prazo |
|------|-------------|-------|

5. ENCERRAMENTO

Nada mais havendo a tratar, a reunião foi encerrada às [HORÁRIO], lavrando-se a presente ata.

[LOCAL], [DATA].

---

## Instruções adicionais:

1. **Linguagem formal**: Use terceira pessoa e tom institucional
2. **Objetividade**: Resuma discussões, não transcreva literalmente
3. **Números**: Escreva por extenso quando possível (ex: "três participantes")
4. **Deliberações claras**: Cada decisão deve ter ação, responsável e prazo quando disponíveis
5. **Inferir contexto**: Se o órgão não for mencionado, deixe [ÓRGÃO] para preenchimento posterior
6. **Participantes**: Use os nomes dos speakers identificados (SPEAKER_00, etc.) como placeholders se não houver nomes reais
7. **Formatação SEI!**: Evite tabelas complexas, use texto estruturado com numeração
```

---

## Exemplo de Uso

No Claude Code, após transcrever `reuniao_gestao.wav`:

```
Leia o arquivo data/transcripts/reuniao_gestao.txt e gere uma ATA DE REUNIÃO formal seguindo o padrão institucional brasileiro para o SEI!.

[Cole o restante do prompt acima]
```

---

## Personalização

Você pode adaptar este prompt para seu contexto específico:

- **PMDF**: Adicione "POLÍCIA MILITAR DO DISTRITO FEDERAL" no cabeçalho
- **Secretaria de Saúde**: Adicione "SECRETARIA DE ESTADO DE SAÚDE DO DF"
- **Reunião específica**: Mencione o comitê/comissão (ex: "Comitê de Gestão de Riscos")

---

## Notas sobre o SEI!

- O texto gerado pode ser copiado diretamente para o SEI!
- Use Ctrl+Shift+V para colar sem formatação
- Após colar, aplique os estilos padrão do sistema
- As assinaturas eletrônicas são adicionadas pelo próprio SEI!
