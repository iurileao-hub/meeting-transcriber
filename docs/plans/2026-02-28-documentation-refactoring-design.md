# Documentation Refactoring Design

> Date: 2026-02-28
> Status: Approved

## Problem

- CLAUDE.md has 366 lines; ~60% is user documentation or changelog, not agent instructions
- README.md and README.pt.md are well-structured but need tone adjustment for non-technical users
- Prompt templates are bilingual within same file (confusing, verbose)
- No separation between changelog, architecture docs, and agent instructions

## Solution: Approach A — Radical Separation

### CLAUDE.md (~100 lines)

Focused exclusively on what Claude Code needs:
1. Project purpose (3 lines)
2. Stack (7 lines)
3. Project structure (tree, ~25 lines)
4. Code conventions (5 lines)
5. Development commands (10 lines — tests, venv, common dev tasks)
6. Required configuration (5 lines — HF_TOKEN + pyannote terms)
7. Architectural patterns (15 lines — factory, ABC, progress callbacks)
8. Development notes (8 lines)
9. Known issues (5 lines)

**Removed:** audio formats, CLI flags, backend installation, user workflow, changelog, feature checklist.
**Added:** test patterns, architectural patterns, dev tips.

### READMEs — Non-technical tone

Target audience: professionals (doctors, lawyers, managers) who need to transcribe meetings.
- Explain technical concepts in simple language
- Step-by-step with verification at each step
- Empathetic troubleshooting
- Structure: What it does → What you need → Install → First use → Modes → Options → Formats → Vocab → AI integration → Troubleshooting → Privacy → Full reference

### Templates — Separated by language

```
examples/
  README.md           → Simple index
  en/
    meeting_minutes.md
    executive_summary.md
    action_items.md
  pt/
    ata_reuniao.md
    resumo_executivo.md
    itens_acao.md
```

Prompts simplified ~50%, no bilingual duplication within files.

### docs/

```
docs/
  CHANGELOG.md          → Changelog extracted from CLAUDE.md
  ARCHITECTURE.md       → Technical decisions, patterns, contribution guide
  DEVELOPMENT_HISTORY.md  → Already exists (keep)
  plans/                  → Already exists (keep)
```

## Success Criteria

- CLAUDE.md under 120 lines
- READMEs equivalent in EN/PT with non-technical tone
- Each template file is monolingual and under 80 lines
- Changelog preserved in docs/CHANGELOG.md
- All information preserved (nothing lost, just reorganized)
