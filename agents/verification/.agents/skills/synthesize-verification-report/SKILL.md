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

1. Collect all critical errors and all gaps from previous checks using `memory_query(run_id, channel, limit=..., reverse=false)` for both `statement_checks` and `reference_checks`, with a limit large enough to include every record in this run. Findings are in `items[].record`; filters match envelope keys, not nested fields. Do not substitute semantic top hits for complete collection. For targeted context, `memory_search` returns flat `results` IDs; use `memory_get(run_id, node_ids=[...], include_dependencies=true)` to read full content, conditions, sources, and direct dependencies before relying on them.
2. Build a complete `verification_report` object with:
   - `summary`
   - `critical_errors`
   - `gaps`
3. Apply strict verdict rule:
   - `correct` iff `critical_errors=[]` and `gaps=[]`.
   - otherwise `wrong`.
4. If verdict is `wrong`, produce concrete non-empty `repair_hints`.
5. Validate the output via `validate_verification_output`.
6. Persist output via `write_verification_output`.

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
- `memory_search`
- `memory_get`
- `memory_append`
- `validate_verification_output`
- `write_verification_output`
