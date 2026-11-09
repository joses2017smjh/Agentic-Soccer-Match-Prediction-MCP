#!/usr/bin/env bash
# Deploy-level check that a run paused for human approval survives a restart.
#
# The gateway must be started with CHECKPOINT_URL=sqlite:///... pointing at a
# disk that outlives the process (see agent/checkpoint.py for the scope).
#
#   BASE=https://your-gateway KEY=... ./scripts/smoke_restart.sh pause
#   ... restart the gateway and wait for /health ...
#   BASE=https://your-gateway KEY=... ./scripts/smoke_restart.sh resume
#
# With the default in-memory checkpointer the resume step fails with a 404,
# which is the behaviour this script exists to catch.
set -uo pipefail

BASE="${BASE:-http://localhost:8000}"
KEY="${KEY:-}"
STATE="${STATE_FILE:-.smoke_restart_thread}"
HDR=(-H "Content-Type: application/json")
[[ -n "$KEY" ]] && HDR+=(-H "X-API-Key: $KEY")

case "${1:-}" in
  pause)
    r=$(curl -s -X POST "${HDR[@]}" -d '{"text":"Arsenal vs Man City — any value bets?"}' "$BASE/predict")
    if ! grep -q '"status":"pending_approval"' <<<"$r"; then
      echo "  FAIL  expected pending_approval, got: ${r:0:200}"; exit 1
    fi
    tid=$(grep -o '"thread_id":"[^"]*"' <<<"$r" | head -1 | cut -d'"' -f4)
    echo "$tid" > "$STATE"
    echo "  PASS  paused for approval (thread_id=$tid)"
    ;;
  resume)
    if [[ ! -s "$STATE" ]]; then echo "  FAIL  no saved thread id; run '$0 pause' first"; exit 1; fi
    tid=$(cat "$STATE")
    r=$(curl -s -X POST "${HDR[@]}" -d "{\"thread_id\":\"$tid\",\"action\":\"approve\"}" "$BASE/approve")
    if grep -q '"status":"complete"' <<<"$r" && grep -q 'Approved value' <<<"$r"; then
      echo "  PASS  approval resumed after the restart (thread_id=$tid)"
    else
      echo "  FAIL  approval did not resume: ${r:0:300}"; exit 1
    fi
    ;;
  *)
    echo "usage: $0 pause|resume"; exit 2
    ;;
esac
