---
name: recursive-proving
description: Revisit decomposition plans after direct screening; M1 runs plans sequentially and prohibits nested model agents.
---

# Recursive Proving

Use this skill when direct proving has failed on the current decomposition plans.

## M2c fixture boundary

The legacy sequential workflow below is not a managed T08 branch and does not
enable live generation. An explicitly controller-managed fixture branch receives
only its frozen approved public baseline and its own write domain; it must not
inherit the sibling stuck points, history, summaries or shared caches listed in
the legacy contract. All required branches must be sealed before the controller
exports a join candidate. The export is not a shared-source commit or a math
review. These instructions explain the modes; the context controller and actual
worker sandbox enforce the boundary, not this skill. Model concurrency remains
at most three including the main session, with no recursive delegation.

## Input Contract

Read:

- the current set of decomposition plans
- the direct-proving reports and key stuck points for each plan
- the known stuck points from other plans
- relevant `failed_paths`, `branch_states`, and search results

## Procedure

1. Confirm that all current decomposition plans have already been attempted with `$direct-proving` and that none has fully solved the problem.
2. In M1, revisit plans sequentially; do not spawn sub-agents. A future trusted scheduler must reserve shared concurrency slots before any parallel work. The total ceiling is three model tasks including the main session.
3. For each sequential attempt retain:
   - the full target theorem
   - the assigned decomposition plan
   - the key stuck points for its own plan
   - the key stuck points found in the other plans
   - the instruction to follow `AGENTS.md`
4. Tackle each plan under `AGENTS.md`, preserving its recorded stuck points. Refine a plan when evidence warrants it, without discarding the history.
5. No recursive spawning is allowed in M1. Tool visibility and prompt instructions are not substitutes for the controller's execution boundary.
6. Preserve progress, failures and candidate developments under the same data-relative `problem_id`; these are research history, not verified facts. Future branches require frozen public baselines plus separate write domains, not unrestricted shared-directory access.
7. Gather the sequential reports. For future managed parallelism, a trusted join barrier must close writes before merging.
8. If any plan succeeds, assemble the proof draft from that plan.
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
- `memory_append`
- `branch_update`
- `search_arxiv_theorems`
- No Codex spawning tools in M1. Any later scheduler must enforce the shared three-task ceiling and forbid nested delegation.

## Failure Logging

If every plan fails in the recursive round, append a summary record to `failed_paths` and immediately invoke `$identify-key-failures`.
