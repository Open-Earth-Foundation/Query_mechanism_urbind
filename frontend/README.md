# Frontend (Document Builder UI)

Next.js frontend using shadcn-style components for a document-first workflow:

1. Select question and city scope.
2. Trigger backend run and wait for completion.
3. Read generated document.
4. Optionally open the dedicated Context Chat workspace (backend memory persisted per run).

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
NEXT_PUBLIC_FRONTEND_MODE=standard
```

Backend API must be running at the configured base URL.

`NEXT_PUBLIC_FRONTEND_MODE` sets the default surface (`standard` or `dev`).
The page header always shows a persistent browser toggle that lets users switch between modes without reloading or changing routes.

## Dev mode

`dev` mode keeps the same route and workflow, but reveals internal tooling:

- `Assumptions Review` entry point from the generated document view
- `Manage Contexts` button and token metrics inside the chat workspace
- read-only `run_id` display with copy action
- session-only OpenRouter API key override controls

The selected mode is stored in browser `localStorage` until the user changes it.
The API key override is not persisted in browser storage.
