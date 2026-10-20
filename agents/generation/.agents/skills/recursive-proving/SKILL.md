---
name: recursive-proving
description: Work through decomposition plans in budgeted batches after direct screening identifies the key stuck points. Use when all current plans have been screened by direct proving and none fully solves the problem.
---

# Recursive Proving

Use this skill when direct proving has failed on the current decomposition plans.

## Input Contract

Read:

- the current set of decomposition plans
- the direct-proving reports and key stuck points for each plan
- the known stuck points from other plans
- relevant `failed_paths`, `branch_states`, and search results

## Procedure

1. Confirm that all current decomposition plans have already been attempted with `$direct-proving` and that none has fully solved the problem.
2. Check the shared allocation first: at most three model tasks total, including the main task, verifier/model work, and all sub-agents. With the main task active, launch at most two plan agents and fewer if other work occupies a slot. Process remaining plans in later batches or serially; do not spawn every plan at once.
3. Give each sub-agent:
   - the full target theorem
   - the assigned decomposition plan
   - the key stuck points for its own plan
   - the key stuck points found in the other plans
   - the instruction to follow `AGENTS.md`
   - the exact same data-relative `problem_id` and relevant saved node IDs; do not create a child scope
4. Tell each sub-agent to tackle the assigned plan under the instructions in `AGENTS.md`, treating that plan as its starting point rather than restarting the search from zero. If new evidence or discoveries justify it, the sub-agent may refine, extend, or locally revise the plan, but it should preserve continuity with the assigned plan instead of discarding it outright.
5. Do not allow unallocated nested model tasks. The configured maximum agent depth is one; sub-agents must work without further spawning. Only the coordinator can assign available budget, always within the shared total of three.
6. Require each sub-agent to search shared memory, then use `memory_get` to read full records, conditions, sources, and direct dependencies. They may append progress, failures, and proof development to this same `problem_id`, with source metadata and justified `requires`, `contrasts`, `refuted_by`, or `supersedes` relations pointing to known node IDs. Return saved node IDs with each report.
7. Wait for and collect each batch, release finished task slots, and then start the next batch if needed. Do not launch verification unless the shared budget has room. Save progress and branch state before ending a turn.
8. If any plan succeeds, get the reported nodes and direct dependencies, check all conditions, and assemble the proof draft from that plan.
9. If all plans fail, hand the collected reports to `$identify-key-failures`.

## Output Contract

Append an `events` record for the recursive round:

```json
{
  "event_type": "recursive_proving_round",
  "plan_ids": ["..."],
  "subagent_ids": ["..."],
  "shared_stuck_points": {
    "plan_id": ["..."]
  },
  "status": "running|completed",
  "successful_plan_ids": ["..."],
  "failed_plan_ids": ["..."]
}
```

Update `branch_states` with the recursive round status and per-plan outcomes.

## Tools

- `memory_search`
- `memory_get`
- `memory_append`
- `branch_update`
- `search_arxiv_theorems`
- Codex sub-agent tools: `spawn_agent`, `send_input`, `wait_agent`, `close_agent`

## Failure Logging

If every plan fails in the recursive round, append a summary record to `failed_paths` and immediately invoke `$identify-key-failures`.
