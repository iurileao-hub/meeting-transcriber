# Executive Summary Template

Generate an executive summary with action plan from a transcript.

## Prompt

```
Read the file data/transcripts/[FILENAME].txt and generate an EXECUTIVE SUMMARY with ACTION PLAN.

## Executive Summary

Include:
- Context (1-2 paragraphs: meeting purpose, why topics matter)
- Key decisions (bullet list, one sentence each)
- Points of attention (risks, concerns, pending items)

## Action Plan (5W2H)

For each action, fill in a table:
- WHAT: Specific task
- WHY: Justification
- WHO: Owner (SPEAKER_XX if unknown)
- WHERE: Department/system
- WHEN: Deadline or "TBD"
- HOW: Approach (if discussed)
- HOW MUCH: Resources needed or "N/A"

Then provide:
- Summary table: #, Action, Owner, Due Date, Priority
- SMART validation checklist for main goals
- Next steps with deadlines

Instructions:
- Infer priorities from tone and emphasis
- Be specific — avoid vague actions like "improve process"
- Note dependencies between actions
- Use "TBD" for undefined deadlines and flag as pending
```

## Methodologies

**5W2H** — Structures each action with 7 questions for clarity and completeness.

**SMART** — Validates goals are Specific, Measurable, Achievable, Realistic, and Time-bound.
