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

Default UI URL: `http://localhost:3000`

## Environment

For local no-Docker frontend runs, create `frontend/.env.local` or use `.env.example`:

```bash
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
NEXT_PUBLIC_LOCAL_API_PORT=8000
NEXT_PUBLIC_FRONTEND_MODE=standard
APP_SHARED_PASSWORD_HASH=replace_with_bcrypt_hash_of_shared_password
APP_SESSION_SECRET=change_me_to_a_32_char_min_random_secret
APP_SESSION_COOKIE_DOMAIN=
APP_SESSION_TTL_SECONDS=604800
APP_LOGIN_RATE_LIMIT_MAX_ATTEMPTS=5
APP_LOGIN_RATE_LIMIT_WINDOW_SECONDS=900
```

`NEXT_PUBLIC_API_BASE_URL` should be set in deployed environments.
When it is omitted, the frontend falls back to a local backend URL built from
`NEXT_PUBLIC_LOCAL_API_PORT`.
For local auth to work, keep frontend and backend on the same host label:
`localhost` with `localhost`, or `127.0.0.1` with `127.0.0.1`.

`NEXT_PUBLIC_FRONTEND_MODE` sets the default surface (`standard` or `dev`).
The page header always shows a persistent browser toggle that lets users switch between modes without reloading or changing routes.
`APP_SHARED_PASSWORD_HASH` and `APP_SESSION_SECRET` are required in the frontend runtime because the full app is protected by the shared password gate. This `frontend/.env.local` file is only needed for local no-Docker runs; Docker Compose and deployed dev/prod inject these values through container/Kubernetes environment variables.

Supported modes:

- Local: `NEXT_PUBLIC_API_BASE_URL=http://localhost:8000` and `APP_SESSION_COOKIE_DOMAIN` unset. The frontend now auto-aligns `localhost` and `127.0.0.1` for local API calls so the shared session cookie is still sent if your browser uses the other loopback hostname.
- Dev deployment: `NEXT_PUBLIC_API_BASE_URL=https://urbind-query-mechanism-api.openearth.dev` and `APP_SESSION_COOKIE_DOMAIN=.openearth.dev`.

## Shared password gate

Configure the frontend runtime like this:

1. Hash the shared password your team will use with bcrypt and store the result in `APP_SHARED_PASSWORD_HASH`.
2. Set `APP_SESSION_SECRET` to the same value used by the backend. In local no-Docker runs this means duplicating the value from the root `.env`; deployed dev/prod gets the value from GitHub Secrets/Kubernetes secrets.
3. Leave `APP_SESSION_COOKIE_DOMAIN` empty locally.
4. Set `APP_SESSION_COOKIE_DOMAIN=.openearth.dev` in production so the cookie reaches both subdomains.

Users type the shared password into the login page, but the frontend runtime stores only its bcrypt hash in `APP_SHARED_PASSWORD_HASH`. `APP_SESSION_SECRET` is a generated signing secret that users never type. The session secret must be at least 32 characters; use a 64-character hex value from the commands below.

Generate a bcrypt hash from `frontend/` after `npm install`:

```bash
node -e "const bcrypt = require('bcryptjs'); bcrypt.hash(process.argv[1], 10).then((hash) => console.log(hash));" "your-shared-password"
```

If you paste that hash into `frontend/.env.local` or another dotenv-loaded file, escape each `$` as `\$`. GitHub Secrets and Kubernetes secrets should store the raw hash.

Generate a session secret on Windows PowerShell:

```powershell
-join ((1..64) | ForEach-Object { "{0:x}" -f (Get-Random -Maximum 16) })
```

Generate a session secret on macOS/Linux:

```bash
openssl rand -hex 32
```

Notes:

- The login page lives at `/login`.
- The frontend exposes `/healthz` publicly for readiness checks; the app UI itself is protected.
- Login and logout POSTs require a same-origin `Origin` or `Referer` header.
- Failed login attempts are throttled per client address and with a fixed per-pod global bucket of 50 attempts per window. Use ingress or WAF rate limiting on `/api/auth/login` in production.
- Rotating `APP_SESSION_SECRET` logs out every active browser session immediately.

## Dev mode

`dev` mode keeps the same route and workflow, but reveals internal tooling:

- `Assumptions Review` entry point from the generated document view
- `Manage Contexts` button and token metrics inside the chat workspace
- read-only `run_id` display with copy action
- session-only OpenRouter API key override controls

The selected mode is stored in browser `localStorage` until the user changes it.
The API key override is not persisted in browser storage.
