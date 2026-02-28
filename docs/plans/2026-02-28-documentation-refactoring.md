# Documentation Refactoring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restructure all project documentation — slim CLAUDE.md to ~100 lines, rewrite READMEs for non-technical users, separate templates by language, extract changelog and architecture docs.

**Architecture:** Move content from CLAUDE.md to purpose-specific files (CHANGELOG.md, ARCHITECTURE.md). Rewrite READMEs with simple language targeting professionals (doctors, lawyers, managers). Split bilingual templates into per-language folders.

**Tech Stack:** Markdown only. No code changes.

---

### Task 1: Create docs/CHANGELOG.md

Extract the changelog content from CLAUDE.md (lines 254-363: "Funcionalidades Implementadas" + "Melhorias de Produção") into a dedicated changelog file.

**Files:**
- Create: `docs/CHANGELOG.md`

**Step 1: Write docs/CHANGELOG.md**

Content: All changelog entries from CLAUDE.md, organized chronologically with proper headings. Keep in Portuguese (matches original).

**Step 2: Verify the file**

Run: `cat docs/CHANGELOG.md | head -5`
Expected: Shows the changelog header.

**Step 3: Commit**

```bash
git add docs/CHANGELOG.md
git commit -m "docs: extract changelog from CLAUDE.md into docs/CHANGELOG.md"
```

---

### Task 2: Create docs/ARCHITECTURE.md

Document architectural decisions, patterns, and contribution guide for developers.

**Files:**
- Create: `docs/ARCHITECTURE.md`

**Step 1: Write docs/ARCHITECTURE.md**

Sections:
1. Architecture Overview (backend factory, ABC base class, progress system)
2. Key Patterns (factory in backends/__init__.py, monkey-patching for progress, lazy imports)
3. Security Decisions (path validation, allowlist vocab, torch.load context manager)
4. Performance Decisions (binary search speaker matching, throttled rendering, lazy imports)
5. Testing Patterns (how to mock backends, common fixtures, running specific tests)

Source material: Technical details currently in CLAUDE.md "Melhorias de Produção" section + DEVELOPMENT_HISTORY.md.

**Step 2: Commit**

```bash
git add docs/ARCHITECTURE.md
git commit -m "docs: create ARCHITECTURE.md with technical patterns and decisions"
```

---

### Task 3: Rewrite CLAUDE.md (~100 lines)

Slim down to only what Claude Code needs to work effectively in this repo.

**Files:**
- Modify: `CLAUDE.md` (full rewrite)

**Step 1: Rewrite CLAUDE.md**

Target structure (~100 lines):
```
1. Propósito (3 lines)
2. Stack (7 lines)
3. Estrutura do Projeto (tree, ~25 lines)
4. Convenções de Código (5 lines)
5. Comandos de Desenvolvimento (10 lines)
6. Configuração Necessária (5 lines)
7. Padrões Arquiteturais (15 lines)
8. Notas para Desenvolvimento (8 lines)
9. Problemas Conhecidos (5 lines)
```

Remove: audio formats, CLI flags, backend installation, user workflow, changelog, feature checklist.
Add: test patterns, architectural patterns key summary, dev tips.

**Step 2: Verify line count**

Run: `wc -l CLAUDE.md`
Expected: Under 120 lines.

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: slim CLAUDE.md to ~100 lines focused on agent instructions"
```

---

### Task 4: Create template files separated by language

Split bilingual templates into per-language folders with simplified prompts.

**Files:**
- Create: `examples/en/meeting_minutes.md`
- Create: `examples/en/executive_summary.md`
- Create: `examples/en/action_items.md`
- Create: `examples/pt/ata_reuniao.md`
- Create: `examples/pt/resumo_executivo.md`
- Create: `examples/pt/itens_acao.md`

**Step 1: Create directories**

Run: `mkdir -p examples/en examples/pt`

**Step 2: Write EN templates**

Each template: ~50-60 lines max. Simplified prompt, one example output, clear instructions. Monolingual.

**Step 3: Write PT templates**

Same structure, Portuguese only.

**Step 4: Commit**

```bash
git add examples/en/ examples/pt/
git commit -m "docs: create per-language template files with simplified prompts"
```

---

### Task 5: Update examples/README.md and remove old templates

Update the examples index to point to new per-language structure. Remove old bilingual files.

**Files:**
- Modify: `examples/README.md` (rewrite)
- Delete: `examples/meeting_minutes.md`
- Delete: `examples/executive_summary.md`
- Delete: `examples/action_items.md`

**Step 1: Rewrite examples/README.md**

Simple index pointing to en/ and pt/ folders. Brief explanation of each template.

**Step 2: Remove old bilingual templates**

Run: `git rm examples/meeting_minutes.md examples/executive_summary.md examples/action_items.md`

**Step 3: Commit**

```bash
git add examples/README.md
git commit -m "docs: update examples index and remove old bilingual templates"
```

---

### Task 6: Rewrite README.md (English) for non-technical users

Full rewrite targeting professionals who need to transcribe meetings.

**Files:**
- Modify: `README.md` (full rewrite)

**Step 1: Rewrite README.md**

Tone: Simple, welcoming, explains technical concepts.
Structure:
1. What It Does (with visual example)
2. What You Need (requirements in plain language)
3. Installation (step-by-step with verification at each step, explain each concept)
4. Your First Transcription
5. Transcription Modes (when to use each, decision tree style)
6. Useful Options (most common ones, not exhaustive)
7. Output Formats (with examples)
8. Custom Vocabulary
9. Using AI to Generate Meeting Minutes (expanded, references examples/en/)
10. Troubleshooting (empathetic tone)
11. Privacy & Security
12. Model Selection (in <details>)
13. Full Options Reference (in <details>)
14. Acknowledgments + License + Author

Key changes:
- Explain "virtual environment" = "isolated folder for the program"
- Explain "token" = "password for accessing AI models"
- Add verification step after each install command
- Expand "Integration with Claude" section
- Empathetic troubleshooting ("Don't worry, this is easy to fix")

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README.md for non-technical users"
```

---

### Task 7: Rewrite README.pt.md (Portuguese) equivalent to English

Full rewrite matching README.md structure and tone, in Portuguese.

**Files:**
- Modify: `README.pt.md` (full rewrite)

**Step 1: Rewrite README.pt.md**

Same structure, tone, and content as README.md but in natural Portuguese (not a mechanical translation). Same non-technical explanations adapted for PT context.

**Step 2: Verify equivalence**

Manually compare section headers and content coverage between README.md and README.pt.md. Same number of sections, same information.

**Step 3: Commit**

```bash
git add README.pt.md
git commit -m "docs: rewrite README.pt.md equivalent to English version for non-technical users"
```

---

### Task 8: Final review and verification

Cross-check all files for consistency, broken links, and completeness.

**Step 1: Verify no information was lost**

Check that all content from old CLAUDE.md exists somewhere:
- Changelog → docs/CHANGELOG.md
- Architecture details → docs/ARCHITECTURE.md
- User instructions → README.md / README.pt.md
- Template content → examples/en/ and examples/pt/

**Step 2: Verify line counts**

Run: `wc -l CLAUDE.md README.md README.pt.md docs/CHANGELOG.md docs/ARCHITECTURE.md examples/en/*.md examples/pt/*.md examples/README.md`

Expected:
- CLAUDE.md: under 120 lines
- README.md and README.pt.md: roughly equivalent line counts
- Each template: under 80 lines

**Step 3: Verify no broken references**

Check that CLAUDE.md doesn't reference removed sections.
Check that READMEs point to correct example paths (examples/en/, examples/pt/).

**Step 4: Final commit if any fixes needed**

```bash
git add -A
git commit -m "docs: final review fixes for documentation refactoring"
```
