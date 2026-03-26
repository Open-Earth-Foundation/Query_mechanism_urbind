---
name: expect
description: Run adversarial browser tests with expect-cli and use Expect's native replay output when available.
license: FSL-1.1-MIT
metadata:
  author: millionco
  version: "2.0.0"
---

# Expect

Use `expect-cli` for browser testing in this repo. Prefer Expect's native replay flow instead of repo-local recording scripts.

## Setup Check

Confirm `expect-cli` is available:

```bash
expect-cli --version
```

If it is missing, install it globally:

```bash
npm install -g expect-cli
```

## Repo Defaults

- Backend from repo root:
  - `uv sync --group dev`
  - `python -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000`
- Frontend from `frontend/`:
  - `npm install`
  - `npm run dev`
- Frontend URL: `http://127.0.0.1:3000`
- Backend URL: `http://127.0.0.1:8000`

## Command

```bash
EXPECT_BASE_URL=http://127.0.0.1:3000 npx -y expect-cli@latest -a codex -m "INSTRUCTION" -t changes
```

Use `EXPECT_BASE_URL` instead of `--base-url`; current `expect-cli 0.0.10` help does not expose `--base-url`.

Use `-t branch` when you want the full branch instead of the current changes.

If the user explicitly wants replay or video output, prefer an interactive Expect run so the CLI can surface its native replay viewer details. When Expect prints a replay or local replay URL, include that URL in your summary.

## Writing Instructions

Think like a user trying to break the feature, not like a checklist that only confirms rendering.

Bad:

```bash
EXPECT_BASE_URL=http://127.0.0.1:3000 npx -y expect-cli@latest -a codex -m "Check that the answer page renders." -t changes
```

Good:

```bash
EXPECT_BASE_URL=http://127.0.0.1:3000 npx -y expect-cli@latest -a codex -m "Open the app, load the latest saved Munich answer, verify the document renders, click a citation chip and confirm the quote popover opens, then open Chat About the Answer and confirm Context Chat Workspace appears without console errors." -t changes
```

## Guardrails

- This app can spend real LLM tokens. Confirm before generating a fresh answer or any flow that depends on live provider behavior.
- Keep prompts focused on the changed or high-risk behavior.
- Do not add `expect-cli` to repo dependencies or CI by default.
- Do not add repo-local recording helpers just to work around Expect limitations.

## After Failures

Read the failure output, fix the issue, and rerun `expect-cli`. If replay details are missing or the CLI hangs after browser actions complete, report that as an upstream Expect limitation instead of introducing a repo-specific workaround.
