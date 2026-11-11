# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

A single operator (developer) who cannot reach Claude/ChatGPT directly from their own machine and uses this server (HTTP + Basic auth) as an access point. Uses it on both desktop and phone. Job: put two or more LLMs against each other on a question or claim, watch the exchange, and read a judged verdict; also simply ask a question to Claude / Codex / DeepSeek.

## Product Purpose

Adversarial reasoning: one agent proposes, another attacks round after round, a judge decides. Success = the operator can launch a debate in seconds, follow who said what live, see the verdict at a glance, and find any past debate again.

## Positioning

Not a chat app: a structured, traced debate between real agent CLIs (claude, codex) and APIs (DeepSeek), with every event appended to an unlimited local trace (`~/lldebat/<projet>/trace-debat.csv`).

## Operating Context

- Projects = sub-folders of `~/lldebat/`; each has `trace-debat.csv` (append-only) and `debates/<id>.json` (replayable).
- A real agent turn takes 10–60 s; a debate takes 1–5 min. Live streaming over SSE; reload/`#id` replays the whole debate.
- Status blocks per turn: proposer CONFIDENCE / CONCEDED / POSITION_CHANGED; critic VERDICT (ACCEPT/CONTEST) / OBJECTIONS; judge WINNER / CONFIDENCE / UNRESOLVED.

## Capabilities and Constraints

- Single static `web/static/index.html`, no build step, no external assets (CSP same-origin, no CDN fonts).
- Python stdlib server; UI language French.
- Roles: proposer, critic, judge; 1–5 rounds; early stop on critic ACCEPT.
- Undecided: N-LLM "arena" mode, Gemini/Antigravity backend.

## Evidence on Hand

Real saved debates under `~/lldebat/*/debates/` and `web/runs/`. No users, metrics or testimonials to cite.

## Product Principles

1. The debate is the content: reading who argued what must be effortless.
2. The verdict is the payoff: it must be impossible to miss.
3. Launching is quick; configuration stays out of the way once set.
4. Everything is traced; history is first-class, not an afterthought.
