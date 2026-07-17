"use client";

/**
 * The prediction console: request input → agent run → layered result.
 *
 * Traffic discipline: input is debounced before any derived work, submission
 * goes through a client cooldown mirroring the gateway's per-IP rate limit,
 * in-flight requests lock the form, and 422/429/503 map to human-readable
 * states instead of raw errors. All requests hit our own /api/* proxies.
 */

import { useMemo, useState } from "react";
import {
  ConformalVisualizer,
  HeadlineTimeline,
  ScorelineHeatmap,
} from "@/components/features/prediction/placeholders";
import { Badge, Panel, Skeleton, Stat } from "@/components/ui/panel";
import { useCooldown, useDebounced } from "@/lib/hooks";
import type { PredictResponse, Suggestion } from "@/lib/types";

const EXAMPLES = [
  "Predict Arsenal vs Man City",
  "Arsenal vs Man City — any value bets?",
  "Liverpool vs Bayern on 2026-07-18",
];

function pct(x: number): string {
  return `${Math.round(x * 100)}%`;
}

export function PredictConsole() {
  const [text, setText] = useState("");
  const debouncedText = useDebounced(text, 450);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const cooldown = useCooldown(12_000);

  const teams = useMemo(() => {
    const pred = result?.status === "complete" ? result.prediction : null;
    const [home = "HOME", away = "AWAY"] = (pred?.match_id ?? "").split("-");
    return { home, away };
  }, [result]);

  async function submit() {
    if (!debouncedText.trim() || busy || !cooldown.ready) return;
    setBusy(true);
    setError(null);
    cooldown.fire();
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: debouncedText }),
      });
      if (res.status === 422) {
        setError(
          "Couldn't identify two teams — try “Arsenal vs Man City” or a match id like ARS-MCI-2026-07-18.",
        );
      } else if (res.status === 429) {
        setError("Rate limit reached — the agent allows a few runs per minute. Try again shortly.");
      } else if (!res.ok) {
        setError(`Gateway unavailable (${res.status}). Is the backend running?`);
      } else {
        setResult((await res.json()) as PredictResponse);
      }
    } catch {
      setError("Network error reaching the app server.");
    } finally {
      setBusy(false);
    }
  }

  async function decide(action: "approve" | "reject") {
    if (result?.status !== "pending_approval" || busy) return;
    setBusy(true);
    try {
      const res = await fetch("/api/approve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ thread_id: result.thread_id, action }),
      });
      if (res.ok) setResult((await res.json()) as PredictResponse);
      else setError(`Approval failed (${res.status}).`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex flex-col gap-4">
      {/* ------------------------------------------------ request bar */}
      <Panel title="Ask the agent">
        <div className="flex flex-col gap-3">
          <div className="flex gap-2">
            <input
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && submit()}
              placeholder="Predict Arsenal vs Man City — any value bets?"
              maxLength={200}
              disabled={busy}
              className="w-full rounded-md border border-line bg-surface-800
                px-3 py-2 text-sm text-ink-100 placeholder:text-ink-600
                focus:border-brand focus:outline-none"
            />
            <button
              onClick={submit}
              disabled={busy || !debouncedText.trim() || !cooldown.ready}
              className="shrink-0 rounded-md bg-brand px-4 py-2 text-sm
                font-semibold text-surface-950 transition-opacity
                disabled:cursor-not-allowed disabled:opacity-40"
            >
              {busy
                ? "Running…"
                : cooldown.ready
                  ? "Predict"
                  : `Wait ${Math.ceil(cooldown.remainingMs / 1000)}s`}
            </button>
          </div>
          <div className="flex flex-wrap gap-2">
            {EXAMPLES.map((ex) => (
              <button
                key={ex}
                onClick={() => setText(ex)}
                disabled={busy}
                className="rounded border border-line px-2 py-1 text-2xs
                  text-ink-400 hover:border-line-strong hover:text-ink-100"
              >
                {ex}
              </button>
            ))}
          </div>
          {error ? <p className="text-xs text-edge-neg">{error}</p> : null}
        </div>
      </Panel>

      {busy && !result ? (
        <div className="grid gap-4 md:grid-cols-3">
          <Skeleton className="h-28" />
          <Skeleton className="h-28" />
          <Skeleton className="h-28" />
        </div>
      ) : null}

      {/* --------------------------------------------- approval gate */}
      {result?.status === "pending_approval" ? (
        <Panel
          title="Staking suggestions — human approval required"
          right={<Badge tone="brand">HITL</Badge>}
        >
          <ul className="mb-4 flex flex-col gap-2">
            {result.approval_request.suggestions.map((s: Suggestion) => (
              <li
                key={`${s.market}/${s.selection}`}
                className="rounded border border-line bg-surface-800/60 p-3 text-xs"
              >
                <div className="mb-1 flex items-center gap-2">
                  <span className="font-semibold uppercase">
                    {s.market}/{s.selection}
                  </span>
                  <Badge tone={s.ev > 0 ? "pos" : "neg"}>
                    EV {(s.ev * 100).toFixed(1)}%
                  </Badge>
                  <Badge>{s.tier}</Badge>
                </div>
                <p className="text-ink-400">{s.rationale}</p>
              </li>
            ))}
          </ul>
          <div className="flex gap-2">
            <button
              onClick={() => decide("approve")}
              disabled={busy}
              className="rounded-md bg-edge-pos/90 px-4 py-2 text-sm
                font-semibold text-surface-950 disabled:opacity-40"
            >
              Approve
            </button>
            <button
              onClick={() => decide("reject")}
              disabled={busy}
              className="rounded-md border border-edge-neg/50 px-4 py-2
                text-sm font-semibold text-edge-neg disabled:opacity-40"
            >
              Reject
            </button>
          </div>
        </Panel>
      ) : null}

      {/* ------------------------------------------------- results */}
      {result?.status === "complete" && result.prediction ? (
        <>
          <div className="grid gap-4 md:grid-cols-4">
            <Stat
              label={`${teams.home} win`}
              value={pct(result.prediction.match_outcome.home)}
            />
            <Stat label="Draw" value={pct(result.prediction.match_outcome.draw)} />
            <Stat
              label={`${teams.away} win`}
              value={pct(result.prediction.match_outcome.away)}
            />
            <Stat
              label="Expected goals"
              value={`${result.prediction.expected_goals.home.toFixed(2)}–${result.prediction.expected_goals.away.toFixed(2)}`}
              hint={`model ${result.prediction.model_version}`}
            />
          </div>

          <div className="grid gap-4 lg:grid-cols-3">
            <ScorelineHeatmap
              scorelines={result.prediction.exact_score.top_scorelines}
            />
            <ConformalVisualizer
              outcome={result.prediction.match_outcome}
              home={teams.home}
              away={teams.away}
            />
            {result.prediction.headline_scenario ? (
              <HeadlineTimeline
                scenario={result.prediction.headline_scenario}
                home={teams.home}
                away={teams.away}
              />
            ) : null}
          </div>

          <Panel
            title="Agent answer"
            right={
              result.degraded.length ? (
                <Badge tone="neg">degraded ×{result.degraded.length}</Badge>
              ) : (
                <Badge tone="pos">full evidence</Badge>
              )
            }
          >
            <pre className="whitespace-pre-wrap font-sans text-sm leading-6 text-ink-100">
              {result.answer}
            </pre>
          </Panel>

          <Panel title={`Evidence trail — ${result.tool_calls.length} tool calls`}>
            <table className="tnum w-full text-left text-2xs">
              <thead className="text-ink-600">
                <tr className="border-b border-line">
                  <th className="py-1.5 pr-4 font-medium">server</th>
                  <th className="py-1.5 pr-4 font-medium">tool</th>
                  <th className="py-1.5 pr-4 font-medium">status</th>
                  <th className="py-1.5 font-medium">latency</th>
                </tr>
              </thead>
              <tbody className="text-ink-400">
                {result.tool_calls.map((c, i) => (
                  <tr key={i} className="border-b border-line/50">
                    <td className="py-1.5 pr-4">{c.server}</td>
                    <td className="py-1.5 pr-4">{c.tool}</td>
                    <td className="py-1.5 pr-4">
                      {c.ok ? (
                        <span className="text-edge-pos">ok</span>
                      ) : (
                        <span className="text-edge-neg">{c.error || "failed"}</span>
                      )}
                    </td>
                    <td className="py-1.5">{c.latency_ms.toFixed(0)} ms</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Panel>
        </>
      ) : null}

      {result?.status === "complete" && !result.prediction ? (
        <Panel title="Agent answer" right={<Badge tone="neg">no prediction</Badge>}>
          <pre className="whitespace-pre-wrap font-sans text-sm leading-6 text-ink-400">
            {result.answer}
          </pre>
        </Panel>
      ) : null}
    </div>
  );
}
