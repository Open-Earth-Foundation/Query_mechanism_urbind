# Frontend (Document Builder UI)

Next.js frontend using shadcn-style components for a document-first workflow:

1. Select question and city scope.
2. Trigger backend run and wait for completion.
3. Read generated document.
4. Optionally open the dedicated Context Chat workspace, which keeps the writer document available in the left rail while chat memory stays persisted per run.
5. In the docked left rail, switch between the generated writer document, raw CCC markdown by city, and build controls.

## Run locally

From `frontend/`:

```bash
npm install
npm run dev
```

Default UI URL: `http://127.0.0.1:3000`

## Environment

Create `.env.local` or use `.env.example`:

```bash
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
NEXT_PUBLIC_LOCAL_API_PORT=8000
NEXT_PUBLIC_FRONTEND_MODE=standard
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_test_your_key_here
CLERK_SECRET_KEY=sk_test_your_key_here
NEXT_PUBLIC_CLERK_SIGN_IN_URL=/sign-in
NEXT_PUBLIC_CLERK_SIGN_UP_URL=/sign-up
NEXT_PUBLIC_CLERK_SIGN_IN_FALLBACK_REDIRECT_URL=/
NEXT_PUBLIC_CLERK_SIGN_UP_FALLBACK_REDIRECT_URL=/
```

`NEXT_PUBLIC_API_BASE_URL` should be set in deployed environments.
When it is omitted, the frontend falls back to a local backend URL built from
`NEXT_PUBLIC_LOCAL_API_PORT`.

`NEXT_PUBLIC_FRONTEND_MODE` sets the default surface (`standard` or `dev`).
The page header always shows a persistent browser toggle that lets users switch between modes without reloading or changing routes.
`NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` and `CLERK_SECRET_KEY` are required because the full app is protected by Clerk.

## Clerk setup

Configure both the Clerk development instance and the Clerk production instance the same way:

1. Enable Google social sign-in.
2. Enable restricted mode.
3. Invite users manually from the Clerk dashboard.

Notes:

- Development and production Clerk instances have separate user and invitation lists.
- Production Google OAuth credentials are stored in the Clerk dashboard, not in repo env files.
- The frontend exposes `/healthz` publicly for readiness checks; the app UI itself is protected.

## Dev mode

`dev` mode keeps the same route and workflow, but reveals internal tooling:

- `Assumptions Review` entry point from the generated document view
- `Manage Contexts` button and token metrics inside the chat workspace
- read-only `run_id` display with copy action
- session-only OpenRouter API key override controls

The selected mode is stored in browser `localStorage` until the user changes it.
The API key override is not persisted in browser storage.
