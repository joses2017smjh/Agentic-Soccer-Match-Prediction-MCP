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

RUN pip install --no-cache-dir -e . \
    langgraph langchain-mcp-adapters mcp fastapi uvicorn httpx

# demo artifacts baked in so `docker compose up` works with zero setup;
# mount /app/data/artifacts to serve real trained bundles instead
RUN python -m scripts.build_demo_artifacts

EXPOSE 8000 8101 8102 8103
# honor $PORT when a host (Render/Fly/Cloud Run) assigns one; default 8000
# for local/compose (where the service `command:` sets the port explicitly)
CMD ["sh", "-c", "uvicorn gateway.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
