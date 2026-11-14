# agy-relay

Antigravity CLI (`agy`, Google's replacement for Gemini CLI on personal accounts) behind a
small HTTP relay. It is the **only** container that holds the Google OAuth token.

```
lldebat (UI, exposed via Caddy) --HTTP + AGY_RELAY_TOKEN--> agy-relay (no published port)
                                                             - strict tool permission + deny list
                                                             - fresh empty cwd per turn
                                                             - agy stderr -> /dev/null (it prints the account e-mail)
                                                             - one turn at a time, hard kill on timeout
```

Why: in print mode agy is an agent. With default settings it ran shell commands that the
prompt asked for. Debate prompts carry the other agent's text, so a hostile answer could
otherwise read and exfiltrate the token.

## API

| Call | Result |
|---|---|
| `GET /health` | `200 {"ok": true}` (no auth) |
| `POST /complete` `{"prompt", "model"}` + `Authorization: Bearer $AGY_RELAY_TOKEN` | `200 {"text", "usage"}` |

Errors: `{"error": {"kind", "message"}}`, kind in `unauthorized` (401), `bad_request` (400),
`busy` (503), `timeout` (504), `agent_error` / `empty_response` / `protocol` (502).
Models: `gemini-3.8-flash-low|medium|high`.

## Operating

- `AGY_RELAY_TOKEN` (>= 32 chars) lives in `web/.env`.
- The token and agy state live on the `lldebat_agy-home` volume. Never mount it in another container.
- Signing in (one time, on the command line): run `agy` interactively in a throwaway
  container on that volume and follow the Google link; then restart the relay.

## Upgrading agy

1. Read version, url, sha512 from
   `https://antigravity-cli-auto-updater-974169037036.us-central1.run.app/manifests/linux_amd64.json`.
2. Update the three `ARG`s in `Dockerfile`.
3. Compare the tool list in agy's `init` event with `DENIED_TOOLS` in `relay.py`.
4. Re-run the anti-injection test: a prompt asking to run `touch /tmp/pwned_by_debate` and to
   quote the token file must leave no file in the relay's `/tmp` and no token value in the answer.

## Known limits

- Quota errors arrive as a generic `agent_error` (agy's message, e-mail redacted). A typed
  `quota` kind waits until we have seen agy's real quota-exhausted output.
- agy keeps conversation history on the volume (~0.4 MB per turn); no cleanup yet.
