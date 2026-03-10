# Frontend Dev Toggle Notes

## Summary

The frontend now uses one route tree with two visibility modes:

- `standard`: normal end-user surface
- `dev`: reveals internal tools and diagnostics

The selected mode is stored in browser `localStorage` and remains active until the user changes it.

## Mode controls

- `NEXT_PUBLIC_FRONTEND_MODE=standard|dev` sets the default mode.
- `NEXT_PUBLIC_ENABLE_DEV_MODE_TOGGLE=true|false` controls whether the runtime toggle is visible.
- The runtime toggle only changes frontend visibility. It is not a security boundary.

## Dev-only features

- `Assumptions Review` button from the generated document view
- `Manage Contexts` button in the chat workspace
- chat token metrics (`total_tokens`, `token_cap`, per-context token counts)
- read-only `run_id` display with copy action
- user-owned OpenRouter API key controls (`Use This Key`, `Clear`)

## Run ID behavior

- `run_id` is display-only in dev mode.
- The value comes from the active loaded or generated run.
- The frontend does not send a user-defined `run_id` when creating runs.

## Persistence rules

- frontend mode persists in browser storage until changed by the user
- the selected run already persists through the existing `last_run_id` behavior
- the OpenRouter API key override is session-only and is not stored in `localStorage`

## Files involved

- `frontend/src/lib/frontend-mode.ts`
- `frontend/src/app/page.tsx`
- `frontend/src/components/context-chat-workspace.tsx`
- `frontend/src/components/dev-mode-toggle.tsx`
- `frontend/src/components/dev-tools-panel.tsx`
