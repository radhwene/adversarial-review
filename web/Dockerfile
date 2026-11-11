# Debate server + the two agent CLIs. Auth state lives on the llm-auth volume, never in the image.
FROM node:22.11.0-bookworm-slim

ARG CLAUDE_CODE_VERSION=2.1.322
ARG CODEX_VERSION=0.142.5

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 ca-certificates tini \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g --no-fund --no-audit \
         "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" "@openai/codex@${CODEX_VERSION}" \
    && npm cache clean --force

# node:*-slim ships user "node" (uid 1000); reuse it instead of running as root.
ENV HOME=/home/node \
    CLAUDE_CONFIG_DIR=/data/auth/claude \
    CODEX_HOME=/data/auth/codex \
    PYTHONUNBUFFERED=1
RUN mkdir -p /data/auth/claude /data/auth/codex /data/lldebat \
    && chown -R node:node /data && chmod 700 /data/auth

COPY --chown=node:node . /app/web
WORKDIR /app
USER node

EXPOSE 8765
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python3", "web/server.py", "--host", "0.0.0.0", "--port", "8765", "--projects-dir", "/data/lldebat"]
