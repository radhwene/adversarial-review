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
- Model output is HTML-escaped before the mini-markdown renderer runs. A CSP restricts the page to same-origin.

## Tests

```bash
cd web && python3 -m unittest discover -s tests -v
```

- `test_engine.py`: status-block parsing, exact turn order, exact prompt contents per turn, early convergence, missing block, typed backend errors and timeouts, exact CLI argv, codex JSONL error extraction.
- `test_projects.py`: project name validation, trace-debat.csv append-only writes, debate JSON persistence and listing.
- `test_deepseek.py`: DeepSeek backend against a local fake API server (exact request, HTTP error message, malformed response, timeout, missing key).
- `test_server.py`: a real HTTP server on an ephemeral port with real SSE parsing. It runs two full debates back to back (anti-flake) and also covers replay from `Last-Event-ID`, 400/404/429, 415 on non-JSON POST, 403 on a foreign `Host`. With auth: 401 challenge, wrong user/password/malformed header, full debate with credentials and any `Host`, CSRF still 415, lockout 429, header-less probes not counted, short password rejected, public bind without auth refused.

### Real end-to-end runs (2026-11-10, Claude Code 2.1.322)

| Run | Agents | Rounds | Result | Per-turn latency | Total |
|---|---|---|---|---|---|
| 1 | claude ×3 | 2 max | critic ACCEPT at r2, judge verdict | 21–28 s | 2 min 01 |
| 2 | claude ×3 (after isolation fix) | 2 max | critic ACCEPT at r2, no instruction leak | 16–27 s | ~1 min 45 |
| 3 | claude / **codex** / claude | 1 | `debate_failed`: "The 'gpt-5.5' model is not supported when using Codex with a ChatGPT account." | n/a | 24 s |

**Codex model:** this machine's ChatGPT-linked Codex account rejects `gpt-5.5`, `gpt-5.4`, `gpt-5.2` and `gpt-5`; it only accepts `--codex-model codex-auto-review`.
