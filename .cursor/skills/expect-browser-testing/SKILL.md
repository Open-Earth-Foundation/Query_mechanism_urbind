---
name: expect-browser-testing
description: Use when a user asks to run Expect or expect-cli, validate the frontend in a real browser, or surface Expect's native replay output for this repository.
---

# expect-browser-testing

## Overview

Use `expect-cli` for browser testing against the running Query Mechanism Urbind frontend. Prefer Expect's native replay flow instead of repo-local recording helpers.

## When to use

- Frontend or full-stack changes where browser behavior matters
- End-to-end UI smoke tests for document generation, citations, chat, or dev-mode controls
- Diff-based validation of current changes or the branch
- Requests for replay or video output from Expect itself

Do not use this as a replacement for `pytest`, backend smoke scripts, or linting.

## Repo defaults

- Backend from repo root:
  - `uv sync --group dev`
  - `python -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000`
- Frontend from `frontend/`:
  - `npm install`
  - `npm run dev`
- Optional frontend env:
  - `NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000`
- Local URLs:
  - frontend `http://127.0.0.1:3000`
  - backend `http://127.0.0.1:8000`

## Workflow

1. Make sure the local app or deployed target is reachable.
2. Confirm with the user before running flows that will spend LLM tokens or depend on live provider behavior.
3. Run Expect with a concrete instruction.
   - Current changes:
     `EXPECT_BASE_URL=http://127.0.0.1:3000 npx -y expect-cli@latest -a codex -m "<instruction>" -t changes`
   - Whole branch:
     `EXPECT_BASE_URL=http://127.0.0.1:3000 npx -y expect-cli@latest -a codex -m "<instruction>" -t branch`
4. If the user wants replay or video details, prefer an interactive Expect run and surface any replay URL or local replay URL that Expect prints.
5. Report pass or fail, repro steps, replay details if available, and anything left untested.

## Prompt examples

- `Open http://127.0.0.1:3000, load the latest saved Munich answer, verify the answer renders, click a citation chip so the quote popover opens, and open Chat About the Answer without UI or console errors.`
- `Open http://127.0.0.1:3000 in dev mode, generate an answer for Munich and Leipzig, switch to chat, ask a follow-up, and verify citation chips remain clickable.`
- Weak: `test the app`

## Guardrails

- Keep prompts scoped to the behavior that changed.
- Prefer `-t changes` during active development.
- Use `EXPECT_BASE_URL`; current `expect-cli 0.0.10` help does not expose `--base-url`.
- Do not add `expect-cli` to `frontend/package.json` or CI by default.
- Do not add repo-local recording helpers just to work around Expect limitations.
- Expect is `FSL-1.1-MIT`, not plain MIT today. Do not vendor it into the repo without explicit approval.
