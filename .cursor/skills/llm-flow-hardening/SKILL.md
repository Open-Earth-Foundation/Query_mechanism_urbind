---
name: llm-flow-hardening
description: Review LLM pipeline changes that add or modify retry logic, batching, multi-pass aggregation, fallback flows, or persisted diagnostics/artifacts. Use when an agent, orchestrator, writer, or API path changes how LLM calls are retried, split, combined, failed closed, or written to/read from run artifacts.
---

# llm-flow-hardening

Use this skill after changing any LLM execution flow that can fail in more than one stage.

## When to use

- Add or change retry or backoff behavior around LLM calls
- Add or change batching, chunk splitting, map-reduce, or multi-pass combine flows
- Add or change fallback paths between direct and degraded execution modes
- Add or change persisted diagnostics, run artifacts, or artifact-loading APIs
- Add or change token-budget enforcement or input-size gating

Do not use this for prompt-only wording changes with no runtime flow impact.

## Goal

Catch the failure modes that usually appear one commit after the main feature lands:

- retries that never prove a second attempt can succeed
- batching that does not fail closed when one unit still exceeds limits
- fallback paths that work only on the happy path
- diagnostics readers that assume artifacts always exist or are well-formed
- artifact paths that can escape the intended run directory

## Workflow

1. Identify the flow boundary.

- Find the entrypoint that starts the LLM flow.
- Find every retry wrapper, batching helper, fallback branch, and artifact writer/reader touched by the change.
- Trace which values are runtime inputs, which are derived, and which are persisted.

2. Enumerate the stages that can fail independently.

- initial LLM call
- retry attempt
- split/batch planning
- per-batch execution
- combine/reduce execution
- artifact persistence
- artifact reload or API exposure

3. Check each stage against the hardening checklist below.

4. Add or update focused regression tests in the same change.

5. Report which hardening checks passed, which are still missing, and what exact test or code change is needed.

## Hardening Checklist

### Retry behavior

- Prove a transient failure on attempt 1 can succeed on attempt 2.
- Prove the retry path logs or records the retry event in a structured way.
- Prove retry exhaustion surfaces a clear failure instead of silently degrading.
- Verify retry configuration is read from the intended config surface, not hard-coded ad hoc.

### Batching and multi-pass behavior

- Prove oversized input triggers the split or multi-pass path when expected.
- Prove one remaining oversized unit fails closed with a clear error after batching.
- Prove batch order and combine inputs stay deterministic.
- Prove the final combine step preserves the contract expected by downstream code.
- Verify the non-batched path still works when the payload stays under the threshold.

### Fallback behavior

- Prove the fallback path is entered only for the intended failure or threshold condition.
- Prove the fallback result still satisfies the same public output contract.
- Verify fallback-specific diagnostics are persisted so later debugging can explain why fallback happened.
- Reject silent fallback when the system should stop instead of hiding a broken state.

### Diagnostics and artifact persistence

- Prove missing artifacts return `None`, an empty structure, or a clear error as intended.
- Prove malformed JSON or malformed log payloads fail safely.
- Prove artifact labels or URLs exposed to callers do not leak host-only filesystem paths unless that is explicitly intended.
- Prove structured diagnostics include enough context to debug the degraded path: counts, thresholds, retry summary, or coverage summary.

### Path confinement and safety

- Prove configured artifact paths cannot escape the run directory through absolute paths, drive prefixes, or `..`.
- Prove fallback filename resolution stays inside the run directory.
- Verify only existing in-run files are exposed back to callers.

## Required Test Coverage

Add or update tests that cover, at minimum, the parts of the checklist touched by the change.

Prefer focused tests over one large integration test. Typical coverage should include:

- one transient retry success case
- one retry exhausted case when exhaustion behavior changed
- one threshold-crossing case that enters batching or fallback
- one post-batching oversize case that fails closed
- one missing-artifact or malformed-artifact read case
- one path-confinement case when artifact paths are configurable or persisted

## Review Output

When using this skill in review mode, report:

- `Flow changed`: the exact runtime path that changed
- `Risk`: retry, batching, fallback, diagnostics, or path safety
- `Evidence`: file and test references
- `Missing hardening`: the concrete gap
- `Fix`: the exact test or code adjustment to add
- `Verification`: what was run, or what still needs to be run

## Do Not Do

- Do not accept a new fallback path without a test that proves when it activates.
- Do not accept batching changes that only test the happy path.
- Do not expose raw persisted artifact paths to callers unless the product contract requires it.
- Do not rely on manual reasoning alone when the failure mode can be reproduced with a focused test.
