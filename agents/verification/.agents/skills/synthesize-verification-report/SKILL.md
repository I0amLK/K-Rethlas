---
name: synthesize-verification-report
description: Aggregate all detected errors and gaps into the final verification report, apply strict accept/reject logic, and produce repair hints when rejected.
---

# Synthesize Verification Report

Produce the final verification output JSON and verdict.

## Input Contract

Read all findings from:

- `statement_checks`
- `reference_checks`

Each issue must include `location` and `issue`.

## Procedure

1. Collect all critical errors and all gaps from previous checks.
2. Build a complete `verification_report` object with:
   - `summary`
   - `critical_errors`
   - `gaps`
3. Apply strict verdict rule:
   - `correct` iff `critical_errors=[]` and `gaps=[]`.
   - otherwise `wrong`.
4. If verdict is `wrong`, produce concrete non-empty `repair_hints`.
5. Validate the output via the shared `validate_verification_output` contract. A legitimate `wrong` report remains a valid business result.
6. Persist a **candidate** via `write_verification_output`; it is not a publication receipt.
7. The trusted controller independently compares all accepted, privately registered findings, exact frozen input revisions and review configuration. Deleting a worker JSONL file cannot unregister a finding. This protects coverage of received findings, not discovery of every mathematical error.
8. Never issue a capability or receipt from model-side MCP. Fixture output is only a protocol test; live LLM review is unavailable in M1 until its isolated broker exists.

## Output Contract

Final output JSON:

```json
{
  "verification_report": {
    "summary": "string",
    "critical_errors": [],
    "gaps": []
  },
  "verdict": "correct",
  "repair_hints": ""
}
```

If there is any error or gap, verdict must be `"wrong"` and `repair_hints` must be non-empty.

## MCP Tools

- `memory_query`
- `memory_append`
- `validate_verification_output`
- `write_verification_output`
