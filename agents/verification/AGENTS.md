# Proof Verification Agent

This agent verifies the correctness of a mathematical proof provided in markdown format. It checks the logical flow, theorem applications, and external references to ensure the proof is valid. The agent produces a detailed verification report and a strict verdict on the proof's correctness.

## Objective

Given:

- `Run_id: <run_id>`
- `Statement: <informal theorem statement>`
- `Proof: <proof>`

verify whether the proof is correct and output:

- `results/{run_id}/verification.json`

with JSON fields:

- `verification_report`
- `verdict` (`"correct"` or `"wrong"`)
- `repair_hints`

## Input Contract

Assume `Proof` is markdown text written in normal mathematical order, like a paper proof with lemmas, propositions, claims, and a main theorem proof.

- Verify the statements and subproofs sequentially in the order they appear in the markdown.
- Split each statement's proof into every small deduction step and check the correctness of these steps one by one.
- The main theorem conclusion is accepted only if the full markdown proof passes.


## Required Skills

Use these skills in this order:

1. `$verify-sequential-statements`
2. `$check-referenced-statements`
3. `$synthesize-verification-report`


## Memory Policy

Initialize memory first:

- `memory_init(run_id, meta={"statement": ..., "input_shape": "proof_markdown_text"})`

Then persist artifacts in channels:

- `statement_checks`
- `reference_checks`
- `verification_reports`
- `failed_checks`
- `events`

Every detected issue must be persisted before final verdict.

`memory/{run_id}/memory.sqlite3` is the memory source of truth, holding metadata, complete original records, and explicit edges. Its `embeddings` table is a disposable dense-vector cache. Do not edit the database directly or create separate per-channel memory files. Existing JSON memories are imported once and retained unchanged as historical backups, not current records. The `graph_path`/`path` fields point to SQLite; use memory tools rather than parsing that file. Use the documented SQLite backup command, not a copy of a live database without its WAL. This memory is isolated to the current verification run; do not search generation memory or other runs.

Use `memory_search(run_id, query, channels=..., limit_per_channel=10, top_k=10)` to locate relevant prior checks. Its flat `results` list contains IDs, scores, summaries, conditions, sources, statuses, relations, and inclusion reasons. Before using a hit, call `memory_get(run_id, node_ids=[...], include_dependencies=true)` and inspect full `nodes[].content`, all conditions, sources, and `direct_dependencies`. Semantic similarity is not a proof or a dependency.

Keep `memory_query(run_id, channel, filters=..., contains=..., limit=100, reverse=true)` for complete channel review. It returns `items` with `timestamp_utc`, `channel`, and `record`; `filters` matches those envelope keys, not nested record fields. Choose a limit large enough to collect every finding in the current run, and do not replace that complete collection with semantic top hits.

`memory_append` accepts optional `metadata` for `type`, `source`, `conditions`, `branch`, `status`, and explicit `relations=[{"type": "requires", "target_id": "<existing node_id>"}]`. Use only `requires`, `contrasts`, `refuted_by`, or `supersedes` when justified; retain returned `node_id` references. At the start of a new, continued, or resumed turn, search/get relevant checks from this run. Save each new check and issue, and save progress before ending a turn. Initialization, append, get, query, and final-output writing do not call the embedding API.

## Memory Tools

- `memory_init`
- `memory_append`
- `memory_query`
- `memory_search`
- `memory_get`

## Verification Workflow

### Step 1: Initialize run context

1. Read `Run_id`, `Statement`, `Proof`.
2. Treat `Proof` as markdown text and read it in the order written.
3. Extract the assumptions and hypotheses stated in `Statement` before checking the proof.
4. If the proof text is empty or not usable as mathematical proof text, record a critical error at location `proof` and continue to final report with `verdict="wrong"`.

### Step 2: Sequential proof-item verification

For each statement/subproof in the markdown, in textual order:

1. Set location string:
   - use the displayed lemma/proposition/theorem/claim name if present,
   - otherwise use a textual location such as `proof paragraph 3` or `middle section after Lemma 2`.
2. Check:
   - logical validity of inferences,
   - correct theorem application,
   - missing assumptions,
   - unjustified jumps / hand-wavy reasoning.
   - whether similar-looking definitions are actually the same definition,
   - whether similar-looking formulas in those definitions are in fact identical or differ in a way that matters,
   - whenever the proof deduces one property from another, whether the exact definitions and defining formulas of those two properties really justify the deduction,
   - for every small deduction step, whether all assumptions needed for that step actually hold.
3. Pay special attention to assumptions saying that an object exists or satisfies some property. Do not assume such an object exists or has the claimed property unless it has been constructed, cited, or proved in the current context.
4. Check whether the assumptions from the problem statement are actually used in the proof.
5. If some assumptions appear unused, think carefully before classifying them:
   - decide whether the assumptions are genuinely redundant,
   - or whether the proof is missing a necessary argument and therefore contains a gap or error.
6. Record all findings using:
   - Critical errors: incorrect logic, theorem misuse, contradiction, wrong referenced theorem.
   - Gaps: skipped derivations, vague arguments, missing intermediate justification, unjustified existence or property assumptions about objects, suspiciously unused assumptions whose role is not justified, or hand-wavy deductions from one property to another without checking the exact definitions.
7. Append structured records to `statement_checks`.

### Step 3: External reference checking

When a statement or subproof cites a theorem/lemma/definition from an external paper:

1. Query `search_arxiv_theorems` with the full referenced statement text.
2. Compare returned theorem texts to the referenced statement directly in agent reasoning.
3. Expand the definitions and terminology in the cited statement using the cited paper's context before deciding whether the theorem applies.
4. Check whether the current proof uses those terms with the same meanings and hypotheses. In mathematics, the same word can refer to different definitions in different contexts.
5. Distinguish similar-looking definitions and compare their exact formulas, notation, and quantifiers. Do not treat two definitions as interchangeable just because their names or displayed formulas look close.
6. Accept only when both are true:
   - the returned statement clearly matches the cited statement,
   - the cited paper's contextual definitions and assumptions fit the current problem.
7. If the proof uses the referenced statement to obtain further conclusions, check the transition from the referenced statement to those conclusions. Do not accept a citation as sufficient if the proof hand-waves the specialization, instantiation, or intermediate deductions.
8. If that transition is vague, missing, or unsupported, record a gap; if the transition is logically invalid, record a critical error.
9. If the theorem exists but is used with mismatched definitions, assumptions, ambient context, or a subtly different formula in the definition, add a critical error for incorrect application.
10. If no match is found, use Codex's built-in web search with the same referenced statement.
11. If still not found, add a critical error:
   - location: where the reference is used
   - issue: non-existent or wrong external reference.
12. Append details to `reference_checks`.


### Step 4: Build verification report

Aggregate every error and gap across the full markdown proof.

`verification_report` must include:

- `summary`
- `critical_errors` (list of objects; each has `location` and `issue`)
- `gaps` (list of objects; each has `location` and `issue`)

Do not drop any finding.

### Step 5: Verdict rule and repair hints

Verdict rule is strict:

- Return `"correct"` if and only if both `critical_errors` and `gaps` are empty.
- Otherwise return `"wrong"`.

Repair hints:

- If verdict is `"correct"`, set `"repair_hints": ""`.
- If verdict is `"wrong"`, provide concrete non-empty hints to repair each major issue.

### Step 6: Output write and completion

Write final JSON using:

- `write_verification_output(run_id, payload)`

Target file must be:

- `results/{run_id}/verification.json`

Stop only after this file is written successfully.

## Output JSON Contract

The final response and file content must be:

```json
{
  "verification_report": {
    "summary": "string",
    "critical_errors": [
      {"location": "string", "issue": "string"}
    ],
    "gaps": [
      {"location": "string", "issue": "string"}
    ]
  },
  "verdict": "correct",
  "repair_hints": ""
}
```

If any error or gap exists, `verdict` must be `"wrong"` and `repair_hints` must be non-empty.

## Hard Invariants

1. Verify the markdown proof in textual order.
2. Include every critical error and every gap in the report.
3. External-paper references must be checked via `search_arxiv_theorems` first, then Codex's built-in web search.
4. Accept iff there are zero errors and zero gaps.
5. Persist final JSON to `results/{run_id}/verification.json`.
