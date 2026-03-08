# Frontend Dev Toggle Plan

## Purpose

This document maps the frontend features currently described as hidden or partially hidden, then proposes a concrete plan to expose them through a frontend dev-mode toggle instead of keeping them unreachable.

The goal is to support two UI surfaces from one codebase:

- `standard` mode: current end-user experience
- `dev` mode: exposes advanced tools, diagnostics, and in-progress controls

The recommended approach is one frontend with mode-gated controls, not two separate frontend apps.

## Current state summary

The README currently documents a `Hidden but implemented features` section. That section is only partially accurate.

- `Assumptions Review` is implemented in the frontend, but there is no visible entry point that opens it.
- `Manage Contexts` is implemented inside the chat workspace, but there is no visible button that opens the dialog.
- user-owned OpenRouter key override is supported in the frontend API client and backend headers, but the actual frontend controls are not present.
- chat token metrics exist in API models and are used in the chat workspace, but they are not exposed as a deliberate user-facing feature set.

The current UI effectively exposes only one advanced post-run action:

- `Chat About the Answer`

## Feature map

| Feature | README claim | Current implementation | Current UI exposure | Assessment |
| --- | --- | --- | --- | --- |
| Assumptions Review workspace | Hidden but implemented | `frontend/src/components/assumptions-workspace.tsx` exists and is mounted conditionally from `frontend/src/app/page.tsx` via `assumptionsOpen` | No visible button or trigger sets `assumptionsOpen` to `true` | Implemented but unreachable |
| Manage Contexts in chat | Hidden but implemented | `frontend/src/components/context-chat-workspace.tsx` contains `Context Manager` dialog and context selection logic | No visible button or trigger sets `isContextManagerOpen` to `true` | Implemented but unreachable |
| User-owned OpenRouter key controls | Hidden but implemented | `frontend/src/lib/api.ts` supports `setUserApiKey()` and sends `X-OpenRouter-Api-Key`; backend accepts that header on runs, chat, and assumptions routes | No visible input, apply, or clear controls found in the frontend | Backend/API support exists, UI controls are not implemented |
| Chat token metrics | Hidden but implemented for regular users | `total_tokens` and `token_cap` exist in frontend API types and chat workspace logic; per-context token counts render inside the hidden context manager dialog | Partially visible only inside hidden dialog; no deliberate dev metrics panel | Implemented in data flow, not fully surfaced |

## Code evidence

### 1. Assumptions Review

- README claim: `README.md`
- Frontend component: `frontend/src/components/assumptions-workspace.tsx`
- Host page state and conditional mount: `frontend/src/app/page.tsx`

Observed behavior:

- `assumptionsOpen` state exists.
- `AssumptionsWorkspace` is conditionally rendered.
- no visible control currently switches the page into that workspace.

Conclusion:

- this is a real hidden feature, not just documentation drift.

### 2. Manage Contexts

- README claim: `README.md`
- Chat workspace implementation: `frontend/src/components/context-chat-workspace.tsx`
- Backend support: `backend/api/routes/chat.py`

Observed behavior:

- the chat workspace contains the full `Context Manager` dialog.
- token cap calculations and context selection logic are active.
- no visible button opens the dialog.

Conclusion:

- this is a real hidden feature, but only because the entry point is missing.

### 3. User-owned OpenRouter key controls

- README claim: `README.md`
- Frontend header support: `frontend/src/lib/api.ts`
- Backend header support:
  - `backend/api/routes/runs.py`
  - `backend/api/routes/chat.py`
  - `backend/api/routes/assumptions.py`

Observed behavior:

- the frontend API client can send `X-OpenRouter-Api-Key`.
- the backend accepts and routes that override.
- no actual frontend control for entering, applying, or clearing a key was found.

Conclusion:

- the README overstates this item.
- the capability exists at API-client level, not as a hidden UI feature.

### 4. Chat token metrics

- README claim: `README.md`
- Frontend models: `frontend/src/lib/api.ts`
- Chat workspace display and selection logic: `frontend/src/components/context-chat-workspace.tsx`
- Backend response fields:
  - `backend/api/models.py`
  - `backend/api/routes/chat.py`

Observed behavior:

- token counts are part of the response contract.
- token counts are already used to enforce context selection limits.
- some token counts are rendered inside the hidden context manager.
- there is no top-level dev metrics panel or explicit standard-vs-dev visibility model.

Conclusion:

- this is partially surfaced internally, but not presented as a deliberate user-facing feature.

## What the frontend is doing today

The current app is not split into a user version and a dev version. Instead, it has one page with a few latent advanced features:

- `frontend/src/app/page.tsx` is the main orchestration surface.
- advanced workspaces are mounted conditionally.
- some advanced controls already exist deeper in the tree.
- missing triggers are what make several features effectively hidden.

This means the lowest-risk design is to add a mode layer on top of the existing app rather than building a second frontend.

## Recommended target design

### Guiding rule

Use one codebase and one route structure, with a small mode system that controls feature visibility.

Do not fork the app into separate pages like "normal page" and "dev page". That would create drift quickly, especially because the current UI is already centralized in `frontend/src/app/page.tsx`.

### Target modes

- `standard`
  - current end-user surface
  - only polished, supported actions are visible
- `dev`
  - reveals advanced tools and diagnostics
  - intended for internal/debug workflows

### Features to gate behind dev mode

- `Assumptions Review` entry point
- `Manage Contexts` button in chat
- chat token metrics summary
- user-owned OpenRouter key controls

### Features that should remain available in both modes

- build report flow
- load previous answer
- city scope selection
- answer mode selection
- generated document
- chat workspace itself

## Proposed implementation shape

### 1. Add a single frontend mode source of truth

Create a small config module, for example:

- `frontend/src/lib/frontend-mode.ts`

Responsibilities:

- read a build-time default from environment
- optionally allow a runtime toggle
- expose helpers such as:
  - `getDefaultFrontendMode()`
  - `isDevModeEnabled()`
  - `getDevFeatureFlags()`

Recommended environment variables:

- `NEXT_PUBLIC_FRONTEND_MODE=standard|dev`
- `NEXT_PUBLIC_ENABLE_DEV_MODE_TOGGLE=true|false`

Default recommendation:

- production default: `standard`
- local/dev deployments: allow runtime toggle

### 2. Keep the runtime toggle explicit and local

Recommended runtime behavior:

- show a small toggle only when `NEXT_PUBLIC_ENABLE_DEV_MODE_TOGGLE=true`
- persist the selected mode in `localStorage`
- allow URL override for debugging if useful, for example `?dev=1`

Important note:

- this is a UX gate, not a security boundary.
- anything sensitive must still be protected by backend auth and safe defaults.

### 3. Replace ad hoc hidden states with feature-gated entry points

Instead of relying on unreachable state variables, add visible controls whose visibility depends on mode.

Recommended changes:

- in `frontend/src/app/page.tsx`
  - keep `Chat About the Answer` in all modes
  - add `Assumptions Review` button only in `dev` mode
- in `frontend/src/components/context-chat-workspace.tsx`
  - add `Manage Contexts` button only in `dev` mode
  - add token summary row only in `dev` mode
- add a small dev settings panel for user API key override only in `dev` mode

### 4. Treat dev-only controls as feature flags, not hardcoded checks

Recommended shape:

- `showAssumptionsWorkspace`
- `showContextManager`
- `showChatTokenMetrics`
- `showUserApiKeyControls`

These flags can all currently map to `mode === "dev"`, but a flag object keeps later rollout more flexible.

## Suggested file-level plan

### Phase 1: mode plumbing

Files:

- new: `frontend/src/lib/frontend-mode.ts`
- update: `frontend/.env.example`

Work:

- define frontend mode type and resolver
- add env-documented defaults
- optionally add localStorage persistence helper

### Phase 2: top-level dev toggle

Files:

- update: `frontend/src/app/page.tsx`
- optional new: `frontend/src/components/dev-mode-toggle.tsx`

Work:

- add mode badge or toggle in the main header
- keep the control subtle and internal-facing
- avoid changing the standard mode layout unless needed

### Phase 3: expose hidden workspaces behind dev mode

Files:

- update: `frontend/src/app/page.tsx`
- update: `frontend/src/components/context-chat-workspace.tsx`

Work:

- add `Assumptions Review` button near `Chat About the Answer`
- add `Manage Contexts` button in chat header
- reveal token metrics only in dev mode

### Phase 4: add API key controls

Files:

- update: `frontend/src/lib/api.ts`
- new or update: a small settings/control component under `frontend/src/components/`

Work:

- add input field, apply button, and clear button
- wire them to `setUserApiKey()`
- avoid storing the raw key in long-lived browser storage by default

Recommendation:

- keep the key in memory for the session unless there is a strong reason to persist it
- if persistence is needed, make it explicit and dev-only

### Phase 5: polish and guardrails

Files:

- update: relevant frontend components
- update: `README.md` after implementation

Work:

- label dev-only features clearly
- add helper text where actions are expensive or experimental
- make standard mode the default in production

## Recommended rollout order

1. Add frontend mode config and runtime toggle.
2. Expose `Assumptions Review` and `Manage Contexts`.
3. Expose token metrics in dev mode.
4. Add user-owned OpenRouter key controls.
5. Update README so it distinguishes:
   - implemented and hidden
   - implemented only at API level
   - dev-mode-only

## Why this is better than a second frontend

- less duplication
- less drift between standard and dev experiences
- easier testing because both modes use the same route tree
- lower documentation burden
- easier rollback because dev-only controls can be hidden without deleting code

## Risks and constraints

### 1. Client-side dev mode is not security

If a feature should not be available to some users, frontend hiding is not enough. The proposed toggle is for product surfacing, not access control.

### 2. The current README should not be copied forward as-is

At least one item is currently overstated:

- user-owned OpenRouter key controls are not actually implemented in the UI today

That should be corrected when the actual dev-mode rollout happens.

### 3. `page.tsx` is already large

The file currently owns a lot of orchestration. The dev-mode work should avoid making it much larger.

Practical implication:

- move mode logic into a small helper/module
- prefer adding small components for new control groups rather than more inline conditionals

## Test plan for the future implementation

### Manual checks

- standard mode does not show:
  - assumptions entry point
  - manage contexts button
  - token metrics
  - API key controls
- dev mode shows all of the above
- toggling modes does not reset the active run unexpectedly
- chat still works in both modes
- assumptions workflow still works when opened from dev mode
- context manager enforces token cap correctly
- API key override applies to runs, chat, and assumptions calls

### Code-level checks

- unit-test mode resolution logic if a test setup exists
- verify no standard-mode code path references missing dev-only state
- verify env defaults keep production in `standard`

## Recommended acceptance criteria

- one frontend build supports both `standard` and `dev` mode
- `standard` mode preserves current user flow
- hidden advanced features become reachable in `dev` mode
- API key controls are either truly implemented in the UI or removed from README claims
- mode logic is centralized instead of scattered across components

## Final recommendation

Implement dev mode as a thin feature-visibility layer over the existing frontend, not as a parallel frontend version.

That keeps the change local, makes the hidden features reachable with minimal churn, and gives the README a cleaner model:

- `standard` mode for normal users
- `dev` mode for internal tools, diagnostics, and advanced workflows
