# Action Items Template

Quick extraction of action items from a transcript.

## Prompt

```
Read the file data/transcripts/[FILENAME].txt and extract all ACTION ITEMS.

Format as a table:

| # | Action | Owner | Due Date | Priority | Status |
|---|--------|-------|----------|----------|--------|

Instructions:
- List ALL tasks, commitments, or actions mentioned
- Include informal commitments ("I'll take care of that")
- Use SPEAKER_XX if names aren't identified
- Mark all as "Pending"
- Infer priority from context (urgency, emphasis)
- Use "TBD" for undefined deadlines
```

## Variations

### Follow-up Report

```
Read the transcript and generate a FOLLOW-UP REPORT:

1. Completed actions (mentioned as finished)
2. In-progress actions (with % progress if mentioned)
3. Pending/overdue actions
4. New items from this meeting
5. Blockers/impediments identified
```

### Decisions Only

```
Read the transcript and extract only DECISIONS made:

| # | Decision | Context | Impact |
|---|----------|---------|--------|
```

## Tips

- **Long meetings**: Ask to split output by topic
- **Multiple owners**: List all, separated by comma
- **Implicit actions**: Include commitments even if not formalized
