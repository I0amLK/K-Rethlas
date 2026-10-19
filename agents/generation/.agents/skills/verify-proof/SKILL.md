---
name: verify-proof
description: Submit a complete candidate to the trusted local verifier; only a current production receipt may establish review status.
---

# Verify Proof

## Input Contract

Read the full original target statement, complete candidate `results/{problem_id}/blueprint.md`, and relevant prior failure reports. Do not submit partial branches or isolated lemmas as complete solutions.

## Procedure

1. Confirm the draft includes a full proof of the target, with expanded definitions, assumptions and exact external references.
2. Call `verify_proof_service(statement=..., proof=...)` only through a controller-issued run identity and capability. The model cannot choose another endpoint, mint a token, change domains, or obtain approval by writing a file.
3. Preserve the **complete** returned response: the original report fields plus binding envelope, verification kind, status and receipt. Never project away provenance.
4. Any `wrong`, critical error or gap requires repair. A contract-valid `wrong` is an ordinary negative business result.
5. A `correct` report alone does not establish completion. Only the controller may verify the signed current-input production receipt, recheck dependencies and publish frozen bytes.
6. Do not rename `blueprint.md` to `blueprint_verified.md`. Legacy files have no review authority.
7. `verification_kind=fixture` / `fixture_completed` is a synthetic protocol test, **never** LLM-reviewed, verified or solved. M1's live provider broker is unavailable; record this limitation instead of bypassing it.
8. Stop at cancellation, finite deadline, unavailable isolation or exhausted budget. Do not open an alternative network/CLI path.

## Output Contract

Append the unchanged full service response to `verification_reports`. Keep the working draft for repair. Preserve failed approaches in `failed_paths`; maintain branch state without promoting research `solved` into mathematical trust.

## MCP Tools

- `verify_proof_service`
- `memory_append`, `memory_search`, `branch_update`
- External search only under a separately authorized trusted egress policy
