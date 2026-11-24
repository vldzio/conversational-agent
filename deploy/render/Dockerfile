# Render Docker web service: private FreeLLMAPI gateway + VoiceFlow demo API in one container.
#
#   public  $PORT (Render sets it, default 10000) -> demo_api (FastAPI: rate limits, BYO key, LangSmith tracing)
#   private 127.0.0.1:3001                        -> FreeLLMAPI (never exposed)
#
# Build context is deploy/render/ (see render.yaml). The project code is cloned at build
# time; redeploy on Render to pick up new commits.
FROM ghcr.io/tashfeenahmed/freellmapi:latest

USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

ARG REPO=https://github.com/vldzio/conversational-agent.git
ARG REF=demo-ready
ARG CACHEBUST=1
# Fetching the branch's current commit id (a tiny JSON file) changes whenever the branch gets a new commit,
# which invalidates the cache of the clone below. Without it Docker would reuse the OLD clone on every
# redeploy because the Dockerfile text itself has not changed.
ADD https://api.github.com/repos/vldzio/conversational-agent/git/refs/heads/demo-ready /tmp/branch-head.json
RUN git clone --depth 1 --branch "${REF}" "${REPO}" /srv/app \
 && python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir -r /srv/app/demo_api/requirements.txt \
 && chown -R node:node /srv/app

COPY start.sh /srv/start.sh
RUN tr -d '\015' < /srv/start.sh > /tmp/start.sh && install -m 755 /tmp/start.sh /srv/start.sh && rm /tmp/start.sh

USER node
ENV HOME=/home/node \
    FREEAPI_DB_PATH=/app/server/data/freeapi.db \
    DEMO_PROJECT=voiceflow \
    LANGSMITH_PROJECT=voiceflow-demo
CMD ["/srv/start.sh"]
