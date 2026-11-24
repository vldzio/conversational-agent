#!/bin/sh
# Starts the private gateway, then the public demo API. Runs as the unprivileged `node` user.
set -e

# Gateway: localhost only. Provider keys / admin account come from FREEAPI_CONFIG_JSON (Render secret env var).
( cd /app && HOST=127.0.0.1 PORT=3001 exec node server/dist/index.js ) &

# Demo API: Render injects $PORT (default 10000).
cd /srv/app
exec /opt/venv/bin/python -m uvicorn demo_api.app:app --host 0.0.0.0 --port "${PORT:-10000}" --proxy-headers --forwarded-allow-ips='*'
