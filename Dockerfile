# One image, four roles — the service command selects the role in Compose.
FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY mcp_servers ./mcp_servers
COPY agent ./agent
COPY gateway ./gateway
COPY scripts ./scripts
COPY configs ./configs
# the derived StatsBomb player-shares artifact (~48 KB) — without it the
# deployed API silently falls back to role-level players instead of names
COPY data/artifacts/player_shares.json ./data/artifacts/

# [durable] adds the SQLite checkpointer, so CHECKPOINT_URL=sqlite:///...
# (paused approvals that survive a restart) works in the image
RUN pip install --no-cache-dir -e ".[durable]" \
    langgraph langchain-mcp-adapters mcp fastapi uvicorn httpx

# demo artifacts baked in so `docker compose up` works with zero setup;
# mount /app/data/artifacts to serve real trained bundles instead
RUN python -m scripts.build_demo_artifacts

# One native thread per model call: single-row inference gains nothing from
# more, and an unbounded OpenMP pool per request thread can exhaust a pids
# limit and kill the process (src/runtime.py). Raise for batch workloads.
ENV MODEL_THREADS=1

EXPOSE 8000 8101 8102 8103 8104
# honor $PORT when a host (Render/Fly/Cloud Run) assigns one; default 8000
# for local/compose (where the service `command:` sets the port explicitly)
CMD ["sh", "-c", "uvicorn gateway.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
