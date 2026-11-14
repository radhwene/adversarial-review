# agy-relay: Antigravity CLI behind a tiny HTTP relay. Holds the Google token; publishes no port.
# Version, URL and SHA-512 come from Google's manifest:
#   https://antigravity-cli-auto-updater-974169037036.us-central1.run.app/manifests/linux_amd64.json
# Bump all three together, then re-run the anti-injection test (web/agy_relay/README.md).
FROM debian:bookworm-slim AS fetch
ARG AGY_VERSION=1.5.7
ARG AGY_URL=https://storage.googleapis.com/antigravity-public/antigravity-cli/1.5.7-6190022064472064/linux-x64/cli_linux_x64.tar.gz
ARG AGY_SHA512=60bca280414ed11723f58b3b671962ae394cc2d0c25e3f58e32e1c6847c978cbc77ec6f40e7f3992001c399d80aeb8b64c9ec892abddd591b12413c5ee6d9fce
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL "$AGY_URL" -o /agy.tgz \
    && echo "$AGY_SHA512  /agy.tgz" | sha512sum -c - \
    && mkdir /out && tar -xzf /agy.tgz -C /out \
    && test -x /out/antigravity

FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 ca-certificates tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 1000 --create-home --home-dir /home/agy --shell /usr/sbin/nologin agy
COPY --from=fetch /out/antigravity /usr/local/bin/agy
COPY relay.py /app/relay.py
ENV HOME=/home/agy PYTHONUNBUFFERED=1
USER agy
WORKDIR /home/agy
EXPOSE 8790
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python3", "/app/relay.py", "--host", "0.0.0.0", "--port", "8790"]
