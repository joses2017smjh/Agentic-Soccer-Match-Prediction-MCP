# Deployment guide

## The one thing to understand first

The gateway has two modes, set by `AGENT_RUNNER`:

- **`inprocess` (default)** — the gateway calls the three MCP servers' logic as
  in-process Python. You deploy **one backend container** (gateway) + the UI.
- **`mcp`** — the gateway is an MCP *client* talking to three separate server
  containers over Streamable HTTP. This is `docker-compose.yml` (5 services),
  for demonstrating the distributed topology.

For a live portfolio URL, use `inprocess`. Nothing about the prediction output
changes — the same tools run, just as function calls instead of HTTP.

**No CORS setup is ever needed:** the browser only talks to the Next.js app's
own `/api/*` Route Handlers, which proxy to the gateway server-side and attach
`GATEWAY_API_KEY`. The gateway is never exposed to the browser.

## Environment variables

| var | where | purpose |
|---|---|---|
| `GATEWAY_URL` | UI | server-side URL of the gateway (never shipped to browser) |
| `GATEWAY_API_KEY` | UI **and** gateway | shared secret; UI attaches it, gateway requires it |
| `AGENT_RUNNER` | gateway | `inprocess` (default) or `mcp` |
| `PREDICT_RATE_LIMIT` | gateway | e.g. `5/minute` per IP on the agent routes |
| `DATA_BACKEND` | gateway | `demo` (default) or `football_data` (real stats) |
| `ANTHROPIC_API_KEY` | gateway | optional; enables ReAct mode + LLM judge |
| `REDIS_URL` | gateway | optional; shared rate-limit store across replicas |

Generate a key with `openssl rand -hex 16`.

---

## Option A — Local (see it run on your machine)

**With Docker** (needs Docker Desktop running):

```bash
GATEWAY_API_KEY=$(openssl rand -hex 16) \
  docker compose -f docker-compose.simple.yml up --build
# UI → http://localhost:3000   ·   gateway → http://localhost:8000
```

**Without Docker** (verified path — this is exactly how it was tested), two
terminals:

```bash
# terminal 1 — backend
pip install -e .                                  # first time only
python -m scripts.build_demo_artifacts            # first time only
GATEWAY_API_KEY=devkey uvicorn gateway.app:app --port 8000

# terminal 2 — frontend
cd ui && npm install                              # first time only
GATEWAY_URL=http://localhost:8000 GATEWAY_API_KEY=devkey npm run dev
```

Then verify the whole thing from a third terminal:

```bash
BASE=http://localhost:8000 KEY=devkey ./scripts/smoke_test.sh   # 14 checks
```

and open http://localhost:3000 in a browser — try "Predict Arsenal vs Man
City" and "Arsenal vs Man City — any value bets?" (the second triggers the
human-approval panel).

## Option B — One cheap VM (shows the full distributed architecture)

Any small VM works (DigitalOcean $6/mo droplet, GCP `e2-micro`, Oracle Cloud
always-free). ~1 GB RAM is enough for the demo model.

```bash
# on the VM, with Docker + compose plugin installed
git clone https://github.com/joses2017smjh/Agentic-Soccer-Match-Prediction-MCP.git
cd Agentic-Soccer-Match-Prediction-MCP
export GATEWAY_API_KEY=$(openssl rand -hex 16)
docker compose up -d --build          # all 5 services (distributed MCP)
```

Open port 3000 in the firewall (and 8000 only if you want the raw API public).
For a domain + HTTPS, put Caddy in front of the UI — a two-line Caddyfile
(`your-domain { reverse_proxy localhost:3000 }`) gets you automatic TLS.

## Option C — Vercel (UI) + Render (backend) — the recommended portfolio path

This repo ships `render.yaml` (backend Blueprint) and `ui/vercel.json` so both
sides are close to push-to-deploy. Pick a key first and use the **same value**
in both places: `openssl rand -hex 16`.

### Step 1 — backend on Render (do this first; you need its URL)

1. Push the repo to GitHub (already done).
2. render.com → **New → Blueprint** → select the repo. Render reads
   `render.yaml` and creates the `matchintel-gateway` Docker service.
3. It will prompt for `GATEWAY_API_KEY` (marked `sync: false`) — paste your key.
4. Deploy. First build takes a few minutes (installs xgboost/langgraph, bakes
   the demo artifacts). When live you get `https://matchintel-gateway.onrender.com`.
5. Sanity-check it: `BASE=https://matchintel-gateway.onrender.com KEY=<key>
   ./scripts/smoke_test.sh`.

Note: the free plan has 512 MB RAM and sleeps after inactivity (first request
after idle is slow, ~30 s cold start). Both are fine for a portfolio; bump the
plan if you see out-of-memory in the logs.

### Step 2 — frontend on Vercel

1. vercel.com → **Add New → Project** → import the repo.
2. Set **Root Directory = `ui`** (Vercel then auto-detects Next.js via
   `ui/vercel.json`).
3. Add two Environment Variables (Production):
   - `GATEWAY_URL` = `https://matchintel-gateway.onrender.com` (from step 1)
   - `GATEWAY_API_KEY` = the same key you gave Render
4. Deploy → `https://<project>.vercel.app` is your live site.

The two deploys are independent and both auto-redeploy on `git push`. The UI
reaches the backend only through `GATEWAY_URL`, and the key is attached
server-side in the Route Handlers — so rotating the key means updating it in
Render and Vercel, nothing in code.

> Swap Render for Fly.io if you want no idle-sleep: `fly launch` detects the
> Dockerfile, `fly secrets set GATEWAY_API_KEY=…`, `fly deploy`.

---

## Verify any deployment

`scripts/smoke_test.sh` drives every capability against a running gateway:

```bash
BASE=https://<your-backend> KEY=<your-key> ./scripts/smoke_test.sh
```

It checks: health, full layered prediction (outcome + conformal set +
scoreline grid + headline scenario + evidence trail), the HITL interrupt on a
value-bet request, approval resume, the 422 on unparseable input, and the
reflection + calibration loop. 14 checks; exit code is nonzero on any failure,
so it works as a post-deploy gate.

## Notes for a public deploy

- **Set `GATEWAY_API_KEY`.** The rate limit slows abuse; the key stops it.
- **Served model is the synthetic demo bundle.** To serve a real trained
  bundle, build it (`scripts/build_demo_artifacts.py` is the template) and
  mount it at `ARTIFACT_ROOT`.
- **`ANTHROPIC_API_KEY` is optional.** Without it the deterministic workflow
  runs (keyless); with it the ReAct follow-up mode and LLM judge activate.
