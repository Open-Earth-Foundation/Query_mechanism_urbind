---
name: pr-thread-analysis
description: Analyze GitHub pull request review threads against the current local branch. Use when the user asks to list unresolved PR comments, quote the exact review text, explain the code under review, determine whether the issue still exists, and state whether a fix or only a reply is needed.
---

# Pr Thread Analysis

Review PR threads against the live working tree, not only GitHub thread state.

Prefer exact reviewer quotes, small local code snippets, and a direct verdict for each thread.

## Workflow

1. Identify the PR for the current branch.
- Prefer `gh` when it is installed and authenticated.
- If `gh` is unavailable, use the GitHub REST API and, if needed, the public PR HTML to inspect thread state.

2. Collect the review threads.
- Capture the thread link, file, line, exact comment text, and any visible resolved/unresolved marker.
- Treat GitHub's unresolved state as advisory. Always compare against the current branch before concluding the issue still exists.

3. Inspect the current code for each thread.
- Open the referenced file and extract the smallest relevant snippet.
- If the commented line moved, find the same logic by symbol name, nearby text, or diff context.
- Distinguish behavior bugs from style or maintainability feedback.

4. Verify when practical.
- Run focused tests or checks around the affected area.
- If tooling is missing or tests are not practical, say that explicitly.

5. Produce the answer thread by thread in a fixed structure.

## Required Output Shape

For each thread, present:

- `Comment`: exact quoted review text and the GitHub discussion link.
- `Relevant code`: a fenced snippet from the current local file. If a snippet is not practical, provide the file path instead.
- `What the comment is pointing at`: explain the code path and why the reviewer raised it.
- `Is it valid now?`: yes, no, or partially, based on the current branch.
- `Does a fix still need to be made?`: yes or no.
- `Fix`: the exact code change to make, or say that only a PR reply / thread resolution is needed.
- `Verification`: tests or checks run, or a clear statement that verification was not run.

## Examples

### Example 1: Comment already addressed on the current branch

Comment:
> "Do we need all the inputs here?"

Relevant code:

```python
return {
    "user_message": user_message,
    "original_question": original_question,
    "history": bounded_history,
    "contexts": [...],
}
```

What the comment is pointing at:
- The reviewer is questioning whether the router payload is larger than needed.

Is it valid now?
- No. The current payload is already reduced and does not include the extra ID fields the reviewer was worried about.

Does a fix still need to be made?
- No.

Fix:
- Reply on the PR that the payload was slimmed down in a later commit and resolve the thread.

Verification:
- Inspect the current file and, when practical, run the focused router tests.

### Example 2: Comment is still directionally valid, but it is a docs/prompt issue

Comment:
> "What exactly is in context?"

Relevant code:

```python
capped_excerpts = excerpts[: max(1, max_excerpts_per_source)]
return {
    "excerpt_count": len(excerpts),
    "excerpts": [...],
}
```

What the comment is pointing at:
- The reviewer is worried the model sees only a capped summary, not the full evidence set.

Is it valid now?
- Partially. The concern is real, but it is about the routing design, not a broken implementation.

Does a fix still need to be made?
- Usually yes, but as a prompt/documentation clarification rather than a functional code fix.

Fix:
- Clarify that the router receives lossy summaries and should choose a new search whenever the summary is insufficient or uncertain.

Verification:
- Inspect the prompt plus the payload builder. Run targeted tests only if the clarification changes behavior.

### Example 3: Style-only comment with no required code change

Comment:
> "this init is a great example of why I dislike having those simplified reexports."

Relevant code:

```python
from backend.api.services import (
    ChatJobExecutor,
    ChatJobStore,
    ChatMemoryStore,
    RunExecutor,
    RunStore,
)
```

What the comment is pointing at:
- The reviewer dislikes the large barrel-style `__init__.py` re-export surface.

Is it valid now?
- Yes, as style feedback.

Does a fix still need to be made?
- No, unless the team wants a cleanup refactor.

Fix:
- If desired, switch callers to direct module imports and shrink the package `__init__.py`. Otherwise answer the thread and resolve it as non-blocking feedback.

Verification:
- Inspect imports only. No runtime test is required unless imports are refactored.

## Style Rules

- Quote the comment when the user asks to see the comment first. Do not paraphrase it away.
- Keep code snippets short and local to the issue.
- Separate "still open in GitHub" from "still a real issue in code".
- Call out clearly when a thread is already addressed by newer commits on the branch.
- If a comment is style-only, say so directly instead of framing it as a defect.
- Prefer primary evidence: GitHub thread text, local code, local tests.

## Useful Fallbacks

- If `gh` is missing, use GitHub REST endpoints for pull requests and review comments.
- If REST does not expose thread resolution, inspect the PR HTML and say when any resolved-state inference is uncertain.
- If comment text is long, quote only the essential portion and link the thread.

## Do Not Do

- Do not rely only on the comment timestamp or GitHub unresolved badge.
- Do not answer with only a high-level summary when the user asked for thread-by-thread detail.
- Do not claim an issue still exists unless the current branch evidence supports it.
