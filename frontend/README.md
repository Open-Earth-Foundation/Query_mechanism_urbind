# Frontend (Document Builder UI)

Next.js frontend using shadcn-style components for a document-first workflow:

1. Select question and city scope.
2. Trigger backend run and wait for completion.
3. Read generated document.
4. Optionally enable popup context chat (backend memory persisted per run).

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
```

Backend API must be running at the configured base URL.
