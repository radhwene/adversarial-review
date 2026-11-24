# Adversarial Reasoning: web UI

Local web interface where one agent **proposes** an answer, another **attacks** it
round after round, and an impartial **judge** writes the verdict.

```
question ─► PROPOSER opening ─► CRITIC r1 ──ACCEPT──► JUDGE ─► verdict
                                   │CONTEST
                                   ▼
                              PROPOSER rebuttal r1 ─► CRITIC r2 ─► … (≤ max_rounds) ─► JUDGE
```

## Run

```bash
python3 web/server.py                       # http://127.0.0.1:8765
python3 web/server.py --codex-model <model> # if your Codex account needs a specific model
python3 web/server.py --timeout 300         # seconds per agent turn (default 600)

# Remote access with login/password (HTTP Basic auth, required for a non-loopback --host)
ADVR_USER=admin ADVR_PASSWORD='<12+ chars>' python3 web/server.py --host 0.0.0.0

# DeepSeek (API, no CLI): enabled only when the key is set
DEEPSEEK_API_KEY=sk-... python3 web/server.py --deepseek-model deepseek-reasoner  # default deepseek-chat

# Where projects live (default ~/lldebat)
python3 web/server.py --projects-dir /data/lldebat
```

## Run in Docker (recommended)

```bash
cd web
printf 'ADVR_USER=...\nADVR_PASSWORD=...\n' > .env && chmod 600 .env   # gitignored
chown -R 1000:1000 ~/lldebat        # the container runs as uid 1000 (user "node")
docker compose up -d --build
```

- Listens on `127.0.0.1:8765` only; TLS and the public port belong to the reverse proxy in front.
- Volume `llm-auth` (`/data/auth`) keeps the claude login (`CLAUDE_CONFIG_DIR`), the codex login
  (`CODEX_HOME`) and the DeepSeek key across rebuilds. The image itself holds no credential.
- `~/lldebat` is bind-mounted on `/data/lldebat`.
- Root filesystem read-only, all capabilities dropped, `no-new-privileges`.

### HTTPS through the lab Caddy

The shared lab proxy (`/srv/lab-proxy`, host network) serves the app at
**`https://<lab host>/llmarena/`** (port 443, next to the lab front and Grafana) and forwards to
`127.0.0.1:8765`. Inside the `{$LAB_PUBLIC_HOST}` site block:

```
redir /llmarena /llmarena/ 308
handle_path /llmarena/* {          # strips the prefix: the server still sees /api/...
	reverse_proxy {$LAB_LLDEBAT_UPSTREAM:127.0.0.1:8765}
}
```

- The page only calls relative URLs (`fetch("api/...")`), so it works under any prefix and on
  `127.0.0.1:8765` directly. A test fails if an absolute `/api` URL comes back.
- `encode` moved from the site level into the Grafana and front `handle` blocks: compression would
  buffer the SSE stream of a debate.
- The old dedicated port **8444** block is still there (same app, same auth).
- The Caddyfile is a single-file bind mount: edit it in place, then `docker exec lab-proxy-caddy-1 caddy reload --config /etc/caddy/Caddyfile`.
- Caddy reaches the container through the compose gateway `10.203.65.1`. That address is the only
  `--trusted-proxy`, so the login lockout keys on the real client from `X-Forwarded-For`.
- Verified 2026-11-12 on `/llmarena/`: 308 from `/llmarena`, 401 without credentials, 200 with them, no
  `Content-Encoding` on the app, two mock debates streamed to the verdict over SSE.
- Verified 2026-11-11 (port 8444): 401 without credentials, 200 with them, a valid certificate, a full SSE debate
  streamed through the proxy, and port 8765 unreachable from outside.

## LLM accounts (sign in from the UI)

Rail section **Comptes LLM**: one card per CLI with its status (Connecté / Non connecté / ⚠️ API payante).

- **Claude**: "Connecter" spawns `claude auth login --claudeai`. Open the link, sign in, copy the code shown, paste it, then "Valider le code". The code goes to the CLI's stdin and is never logged.
- **Codex (ChatGPT)**: "Connecter" spawns `codex login --device-auth`. Open the link and enter the one-time code shown in the card (valid 15 min). The card turns "Connecté" on its own.
- "Annuler" kills a pending login; "Déconnecter" runs the CLI logout. A login left unanswered for 15 min expires.
- Only subscription flows exist: `--console` and `--with-api-key` (pay-per-use API billing) are never run.
- Sign-in actions require HTTPS: through the trusted proxy with `X-Forwarded-Proto: https`, or plain loopback. Anything else gets 403.
- Only `https://` links whose host is `claude.com`/`claude.ai` or `auth.openai.com` are shown, so a CLI line pointing elsewhere is never offered as a link.
- Logins land in the `llm-auth` volume and survive rebuilds.
- **Gemini (Google)**: "Connecter" runs `gemini_auth.py login`, which drives `gemini` under a pseudo-terminal (the CLI refuses its manual Google flow without a TTY). Open the link, sign in with Google, paste the code. Only "Login with Google" is configured; `GEMINI_API_KEY` is never used. Note: on the free personal tier Google may use prompts to improve its models. Gemini records every headless session on disk with no off switch; retention is forced to the minimum, 1 day.
- ⚠️ **Gemini is disabled** (`--disable-agent gemini` in `compose.yaml`) since 2026-11-12: sign-in works, but every
  turn fails with `IneligibleTierError ... UNSUPPORTED_CLIENT` (free "Gemini Code Assist for individuals" tier,
  Google points to Antigravity). Remove the flag once the account is on an eligible tier.
- **DeepSeek** has no subscription CLI, only a pay-per-use HTTP API. Its card takes an API key (`sk-…`, password field, HTTPS only), stored at `/data/auth/deepseek/api_key` mode 0600 on the `llm-auth` volume. The key is never logged nor sent back: the card shows its last 4 characters. "Supprimer la clé" deletes the file; a `DEEPSEEK_API_KEY` env var stays a fallback the UI cannot remove.
- ⚠️ **Every DeepSeek turn is billed to the key owner and there is no cost cap yet.** DeepSeek is never a default seat.

## Model and reasoning effort per seat

Under each party's agent select, two small selects: **modèle** and **effort de réflexion**. "par défaut" leaves the CLI's own choice.

| Agent | Models | Efforts | Passed as |
|---|---|---|---|
| claude | fable, opus, sonnet, haiku | low, medium, high, xhigh, max | `--model X --effort Y` |
| codex | none (the ChatGPT account only accepts `codex-auto-review`) | low, medium, high | `-c model_reasoning_effort=Y` |
| gemini | auto, pro, flash, flash-lite | none | `-m X` |
| deepseek | deepseek-chat, deepseek-reasoner | none | `"model"` in the API body |

- API: `POST /api/debates` takes `options: {"proposer": {"model": "opus", "effort": "high"}, ...}`. `GET /api/backends` lists `models`, `efforts` and `images` (whether the agent reads images) per agent.
- Allowlist only: an unknown seat, key or value gets a 400 before any CLI runs, so a value never reaches an argv unchecked.
- The choice is stored per seat and agent in the browser, shown in the case header, and kept in the debate JSON. "Relancer" reuses it.
- Verified 2026-11-11 over HTTPS: two debates, claude(haiku, low) / codex(low) / claude(sonnet, medium), 31 s and 34 s, critic ACCEPT, verdict. The logs show the flags on every turn.

## Projects and the trace-debat.csv trace

Each project is a sub-folder of `~/lldebat/` (or `--projects-dir`), created from the UI's "+" button (name: letters, digits, `-`, `_`; max 64).

```
~/lldebat/<project>/
├── trace-debat.csv     # append-only, unbounded: one row per event of every debate
└── debates/<id>.json   # full replayable transcript (reload /#<id> rebuilds it)
```

`trace-debat.csv` is never truncated or rotated: it is the project's local memory. The `default` project exists out of the box.

## UI ("court record" layout)

- Left rail: project selector, new-project inline form, the project's debates (date, question, parties, state). On phones it folds behind a "Projet · N débat(s)" toggle.
- New-debate sheet: question, examples, proposer ⇄ critic, judge, rounds 1–5.
- Case view: debate number, question as title, parties, live register (round, proposer confidence, open objections), progress steps. The **verdict is stamped at the top** as soon as the judge finishes (winner, judge confidence, unresolved points). Below it the transcript, grouped into Ouverture / Round N / Délibéré.

No dependencies beyond Python ≥ 3.9 and the agent CLIs you want to use
(`claude`, `codex`). DeepSeek uses its HTTP API. The `mock` agent works offline for demos and tests.

## Framing (Phase 1, filled in by assumption, since `/goal` asked to proceed without pausing)

| | |
|---|---|
| **Problem** | The existing `adversarial_review.sh` only works in a terminal, only reviews code, and shows its result only after all phases finish. |
| **Need** | Ask any question or claim in a browser, then watch the proposer/critic exchange live and get a judged verdict. |
| **Users** | Developers running locally, with their own Claude/Codex CLI quotas. |
| **Alternatives** | (a) a web wrapper around the shell script: rejected, because the script is code-review-specific and batch-only. (b) a Node/React app: rejected, because Node 16 is old, npm adds a supply chain, and none of it is needed. **Chosen:** Python stdlib + SSE + one HTML file. |
| **Success criterion** | The server serves on 127.0.0.1. Each turn streams in as soon as it finishes, and the run ends with a verdict. Mock mode works offline. Tests pass. One real Claude debate ran end to end at least twice. |

## Design notes

- **Engine** (`engine.py`): each debate runs in its own thread and appends events to a list. SSE replays from any index (`?from=N` / `Last-Event-ID`), so reloading the page (`/#<id>`) rebuilds the whole debate.
- **Convergence**: every turn ends with a `---X_STATUS---` block (same convention as the shell tool). The critic's `VERDICT: ACCEPT` stops the debate early. A missing block counts as `CONTEST`, so the debate keeps going and no prose string-matching is needed.
- **Agent isolation**: `claude` runs in an empty temp dir with `--setting-sources project --tools "" --strict-mcp-config`. Without that, the operator's global `~/.claude/CLAUDE.md` leaked into the debate: the first real E2E judge opened with "Phase 5 (Tests)…". `codex` runs `exec --json --sandbox read-only -o <file>`. The answer comes from the final-message file, and errors come from the typed `turn.failed` JSONL events.
- **Choose sides freely**: any agent can be proposer, critic or judge. `⇄` swaps proposer and critic.

## Security

- Binds `127.0.0.1` by default. Every request spawns paid agent CLIs, so the server **refuses** a non-loopback `--host` unless `ADVR_USER` / `ADVR_PASSWORD` are set (password ≥ 12 chars).
- **Basic auth** (when enabled): every route requires credentials (401 + `WWW-Authenticate` otherwise), compared with `hmac.compare_digest`. After 10 failures an IP is locked out for 300 s (429), even with the right password. Requests without a header do not count as failures. With auth on, the `Host` allowlist is skipped (the credentials are the gate); the CSRF JSON check stays.
- ⚠️ **Plain HTTP sends the password base64-encoded, i.e. readable by anyone on the path.** Put a TLS reverse proxy (Caddy, nginx) in front, or use an SSH tunnel, for anything beyond a trusted network.
- Binding to localhost does not protect you from **your own browser**, so two checks are added:
  - **CSRF:** `POST` only accepts `Content-Type: application/json` (415 otherwise). A cross-origin JSON request needs a CORS preflight, and the server never answers one, so another website cannot start debates on your account through a `text/plain` form or `no-cors` fetch.
  - **DNS rebinding:** every request needs a `Host` of `127.0.0.1`, `localhost`, `::1` or the `--host` you passed (403 otherwise). A rebound attacker page therefore cannot read your transcripts or SSE streams.
- At most 2 debates run at once (HTTP 429 beyond that). Bodies are capped at 16 KB, questions at 4000 chars, and rounds at 1–5.
- Chat images: `POST /api/chats` and `/api/chats/<id>/messages` take `images: [base64, ...]`, at most 4 per message and 5 MB each (these two routes accept a bigger body for that). The type comes from the file's magic bytes (PNG, JPEG, GIF, WebP), never from the client. Only claude and codex read images; other agents get a 400. Files are stored as `chats/<id>.files/<sha256>.<ext>`, served from `GET /api/chats/<id>/files/<sha256>.<ext>` with `nosniff` and a sandbox CSP, and deleted with the chat. The browser re-encodes each photo to JPEG (long edge 1568 px) before upload, which drops EXIF/GPS. A reply sends the 4 newest images; older ones stay in the history as `[image N]`.
- Model output is HTML-escaped before the mini-markdown renderer runs. A CSP restricts the page to same-origin.

## Tests

```bash
cd web && python3 -m unittest discover -s tests -v
```

- `test_engine.py`: seat options (allowlist errors, exact argv with `--model`/`--effort`/`model_reasoning_effort`, options reaching each turn, old snapshots without options), status-block parsing, exact turn order, exact prompt contents per turn, early convergence, missing block, typed backend errors and timeouts, exact CLI argv, codex JSONL error extraction.
- `test_projects.py`: project name validation, trace-debat.csv append-only writes, debate JSON persistence and listing.
- `test_deepseek.py`: DeepSeek backend against a local fake API server (exact request, HTTP error message, malformed response, timeout, missing key).
- `test_auth.py`: `AuthManager` against fake claude/codex scripts run as real subprocesses: exact code piped to stdin, lookalike URLs skipped, a wrong code redacted from errors, invalid codes rejected before the CLI, ANSI-wrapped device code, session reuse, expiry kills the CLI, cancel, status parsers (subscription vs API billing).
- `test_server.py`: a real HTTP server on an ephemeral port with real SSE parsing. It runs two full debates back to back (anti-flake) and also covers replay from `Last-Event-ID`, 400/404/429, 415 on non-JSON POST, 403 on a foreign `Host`. With auth: 401 challenge, wrong user/password/malformed header, full debate with credentials and any `Host`, CSRF still 415, lockout 429, header-less probes not counted, short password rejected, public bind without auth refused. `/api/auth` routes: 403 without HTTPS, exact manager calls, typed errors mapped to 400/404/409/503.

### Real end-to-end runs (2026-11-10, Claude Code 2.1.322)

| Run | Agents | Rounds | Result | Per-turn latency | Total |
|---|---|---|---|---|---|
| 1 | claude ×3 | 2 max | critic ACCEPT at r2, judge verdict | 21–28 s | 2 min 01 |
| 2 | claude ×3 (after isolation fix) | 2 max | critic ACCEPT at r2, no instruction leak | 16–27 s | ~1 min 45 |
| 3 | claude / **codex** / claude | 1 | `debate_failed`: "The 'gpt-5.5' model is not supported when using Codex with a ChatGPT account." | n/a | 24 s |

**Codex model:** this machine's ChatGPT-linked Codex account rejects `gpt-5.5`, `gpt-5.4`, `gpt-5.2` and `gpt-5`; it only accepts `--codex-model codex-auto-review`.
