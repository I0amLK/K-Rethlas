---
name: query-memory
description: Retrieve previously saved immediate conclusions, toy examples, counterexamples, failed paths, or branch states from memory. Use when you want to check whether earlier conclusions, examples, counterexamples, failed paths, or brach states can bring insight to the current question, claim, subgoal, or branch decision, or when you want to test a claim against previously saved counterexamples.
---


# Query Memory

Use this skill when you want to check whether earlier conclusions, examples, counterexamples, failed paths, or brach states can bring insight to the current question, claim, subgoal, or branch decision, or when you want to test a claim against previously saved counterexamples.

## Input Contract

Read:

- the current question, claim, subgoal, or branch decision
- the specific type of prior artifact you want to recover
- the most relevant channel list, chosen from:
  - `immediate_conclusions`
  - `toy_examples`
  - `counterexamples`
  - `failed_paths`
  - `branch_states`

## Procedure

1. Form a concrete natural-language query describing the information you want to recover.
2. Choose the smallest relevant list of channels instead of searching everything by default.
3. Call `memory_search(problem_id, query, channels=..., limit_per_channel=10, top_k=10)` using the current shared, data-relative `problem_id`.
4. Inspect its flat `results` list: `id`, `channel`, `score`, `summary`, `conditions`, `source`, `status`, `relations`, and `include_reason`. Semantic seeds follow the channel limits; explicit linked context may cross channels inside this problem, expanding at most two hops and 20 nodes total. There is no cross-problem or verification-memory search.
5. Call `memory_get(problem_id, node_ids=[...], include_dependencies=true)` for useful result IDs. Read full `nodes[].content` together with their complete conditions, sources, statuses, relations, and `direct_dependencies`. If further context is required, get those known IDs explicitly. Do not infer a mathematical dependency or validity from semantic score or summary.
6. Check that the exact hypotheses apply and distinguish active conclusions, failed paths, refutations, and superseded revisions. Summarize useful items with node IDs and explain how they affect the current proof state.
7. If no useful item is found, say that clearly and then switch to another appropriate skill. Save new conclusions with source/conditions and only justified explicit relations.

## Output Contract

Append a summary record to `events`:

```json
{
  "event_type": "query_memory",
  "query": "...",
  "channels": ["counterexamples", "failed_paths"],
  "limit_per_channel": 10,
  "top_k": 10,
  "results_summary": ["..."],
  "useful_hits": [
    {
      "node_id": "ID returned by memory_search",
      "channel": "counterexamples",
      "score": 0.0,
      "why_relevant": "Conditions and full content checked with memory_get",
      "direct_dependency_ids": ["..."]
    }
  ],
  "branch_id": "optional",
  "subgoal_id": "optional"
}
```

## MCP Tools

- `memory_search`
- `memory_get`
- `memory_append`

## Failure Logging

If the retrieval is not useful, append an `events` record with:

- `event_type="query_memory_stalled"`
- the attempted query
- the channels searched
- the reason the retrieved items were not useful
