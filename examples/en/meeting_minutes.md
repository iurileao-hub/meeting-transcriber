# Meeting Minutes Template

Generate formal meeting minutes from a transcript.

## Prompt

```
Read the file data/transcripts/[FILENAME].txt and generate formal MEETING MINUTES.

Include:
- Meeting title (inferred from content), date, time, location, attendees
- Numbered agenda items (inferred from discussion topics)
- Discussion summary per agenda item (key points, viewpoints, questions raised)
- Decisions made (with clear ownership when available)
- Action items table: #, Action, Owner, Due Date, Priority
- Next meeting info (if mentioned)

Instructions:
- Use professional, formal tone
- Summarize discussions — don't transcribe literally
- Use SPEAKER_XX if real names aren't identified
- Include informal commitments as action items
- Leave [PLACEHOLDER] for unknown organization details
```

## Example Output

```
ACME CORPORATION
Engineering Team

MEETING MINUTES - January 15, 2026

Meeting: Sprint Planning Q1
Date: January 15, 2026
Time: 10:00 - 11:30
Location: Virtual/Zoom
Attendees: SPEAKER_00 (facilitator), SPEAKER_01, SPEAKER_02

## 1. AGENDA

1. Review of previous sprint results
2. Q1 priorities discussion
3. Resource allocation

## 2. DISCUSSION

### 1. Review of previous sprint results
The team reviewed completed tasks from the previous sprint...

## 3. DECISIONS

- Decision 1: Adopt new CI/CD pipeline by end of month

## 4. ACTION ITEMS

| # | Action             | Owner      | Due Date | Priority |
|---|--------------------|------------|----------|----------|
| 1 | Set up CI pipeline | SPEAKER_01 | Jan 30   | High     |

## 5. NEXT MEETING

- Date: January 22, 2026
- Topics: CI/CD progress review
```
