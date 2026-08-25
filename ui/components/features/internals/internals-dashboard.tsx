"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import useSWR from "swr";
import { Panel, Badge, Stat } from "@/components/ui/panel";
import { jsonFetcher } from "@/lib/hooks";
import type { AgentEvent, RunDetail, RunSummary } from "@/lib/types";

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

function fmtMs(ms: number): string {
  return ms < 1000 ? `${Math.round(ms)}ms` : `${(ms / 1000).toFixed(2)}s`;
}

function fmtTime(iso: string): string {
  try {
    return new Date(iso).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch {
    return iso;
  }
}

function pct(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

type EventFilter = "all" | "node" | "tool" | "belief" | "critic" | "other";

function eventCategory(type: string): EventFilter {
  if (type.startsWith("node_")) return "node";
  if (type.startsWith("tool_")) return "tool";
  if (type === "belief_update") return "belief";
  if (type === "critic_check") return "critic";
  return "other";
}

// ---------------------------------------------------------------------------
// sub-panels
// ---------------------------------------------------------------------------

function RunPicker({
  runs,
  selected,
  onSelect,
}: {
  runs: RunSummary[];
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="max-h-72 overflow-y-auto">
      <table className="w-full text-2xs">
        <thead className="sticky top-0 bg-surface-900">
          <tr className="text-left text-ink-600">
            <th className="px-2 py-1.5">Time</th>
            <th className="px-2 py-1.5">Match</th>
            <th className="px-2 py-1.5">Mode</th>
            <th className="px-2 py-1.5 text-right">Calls</th>
            <th className="px-2 py-1.5 text-right">Latency</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((r) => (
            <tr
              key={r.thread_id}
              onClick={() => onSelect(r.thread_id)}
              className={`cursor-pointer border-t border-line transition-colors
                ${selected === r.thread_id
                  ? "bg-surface-800 text-ink-100"
                  : "text-ink-400 hover:bg-surface-800/60"}`}
            >
              <td className="tnum px-2 py-1.5 whitespace-nowrap">
                {fmtTime(r.at_utc)}
              </td>
              <td className="px-2 py-1.5 font-medium">{r.match_id || "--"}</td>
              <td className="px-2 py-1.5">
                <Badge tone={r.mode === "swarm" ? "brand" : "neutral"}>
                  {r.mode}
                </Badge>
              </td>
              <td className="tnum px-2 py-1.5 text-right">{r.n_calls}</td>
              <td className="tnum px-2 py-1.5 text-right">
                {fmtMs(r.elapsed_ms)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {runs.length === 0 && (
        <p className="py-6 text-center text-xs text-ink-600">
          No runs recorded yet. Run a prediction to see traces here.
        </p>
      )}
    </div>
  );
}

function Waterfall({ events }: { events: AgentEvent[] }) {
  const nodeEvents = events.filter(
    (e) => e.type === "node_enter" || e.type === "node_exit",
  );
  const toolEvents = events.filter(
    (e) => e.type === "tool_call_start" || e.type === "tool_call_end",
  );

  const spans: {
    label: string;
    kind: "node" | "tool";
    ok: boolean;
    startIdx: number;
    durationMs: number;
    server?: string;
  }[] = [];

  const nodeStarts = new Map<string, number>();
  for (let i = 0; i < nodeEvents.length; i++) {
    const e = nodeEvents[i];
    const name = e.data.node as string;
    if (e.type === "node_enter") {
      nodeStarts.set(name, i);
    } else if (e.type === "node_exit") {
      const si = nodeStarts.get(name);
      spans.push({
        label: name,
        kind: "node",
        ok: true,
        startIdx: si ?? i,
        durationMs: (e.data.duration_ms as number) || 0,
      });
    }
  }

  const toolStarts = new Map<string, number>();
  for (let i = 0; i < toolEvents.length; i++) {
    const e = toolEvents[i];
    const key = `${e.data.server}.${e.data.tool}`;
    if (e.type === "tool_call_start") {
      toolStarts.set(key, i);
    } else if (e.type === "tool_call_end") {
      const si = toolStarts.get(key);
      spans.push({
        label: e.data.tool as string,
        kind: "tool",
        ok: e.data.ok as boolean,
        startIdx: si ?? i,
        durationMs: (e.data.latency_ms as number) || 0,
        server: e.data.server as string,
      });
    }
  }

  const maxDur = Math.max(1, ...spans.map((s) => s.durationMs));

  return (
    <div className="space-y-1">
      {spans.length === 0 && (
        <p className="text-xs text-ink-600">No node/tool events in this run.</p>
      )}
      {spans.map((s, i) => {
        const widthPct = Math.max(2, (s.durationMs / maxDur) * 100);
        const bg =
          s.kind === "node"
            ? "bg-brand/40"
            : s.ok
              ? "bg-edge-pos/30"
              : "bg-edge-neg/40";
        return (
          <div key={i} className="flex items-center gap-2 text-2xs">
            <span className="w-28 shrink-0 truncate text-right text-ink-400">
              {s.server ? `${s.server}/` : ""}
              {s.label}
            </span>
            <div className="relative h-5 flex-1 rounded bg-surface-800">
              <div
                className={`absolute inset-y-0 left-0 rounded ${bg}`}
                style={{ width: `${widthPct}%` }}
              />
              <span className="relative z-10 flex h-full items-center pl-1.5 text-ink-100">
                {fmtMs(s.durationMs)}
              </span>
            </div>
            {!s.ok && <Badge tone="neg">fail</Badge>}
          </div>
        );
      })}
    </div>
  );
}

function BeliefTrace({ events }: { events: AgentEvent[] }) {
  const beliefs = events.filter((e) => e.type === "belief_update");
  if (beliefs.length === 0) {
    return <p className="text-xs text-ink-600">No belief updates in this run.</p>;
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-2xs">
        <thead>
          <tr className="text-left text-ink-600">
            <th className="px-2 py-1.5">Stage</th>
            <th className="tnum px-2 py-1.5 text-right">Home</th>
            <th className="tnum px-2 py-1.5 text-right">Draw</th>
            <th className="tnum px-2 py-1.5 text-right">Away</th>
            <th className="tnum px-2 py-1.5 text-right">Delta H</th>
          </tr>
        </thead>
        <tbody>
          {beliefs.map((e, i) => {
            const after = (e.data.probs_after ?? {}) as Record<string, number>;
            const delta = (e.data.delta ?? {}) as Record<string, number>;
            const dh = delta.home ?? 0;
            return (
              <tr key={i} className="border-t border-line">
                <td className="px-2 py-1.5 font-medium text-ink-100">
                  {e.data.stage as string}
                </td>
                <td className="tnum px-2 py-1.5 text-right">
                  {pct(after.home ?? 0)}
                </td>
                <td className="tnum px-2 py-1.5 text-right">
                  {pct(after.draw ?? 0)}
                </td>
                <td className="tnum px-2 py-1.5 text-right">
                  {pct(after.away ?? 0)}
                </td>
                <td
                  className={`tnum px-2 py-1.5 text-right font-semibold ${
                    dh > 0
                      ? "text-edge-pos"
                      : dh < 0
                        ? "text-edge-neg"
                        : "text-ink-600"
                  }`}
                >
                  {dh > 0 ? "+" : ""}
                  {(dh * 100).toFixed(1)}pp
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function CriticChecks({ events }: { events: AgentEvent[] }) {
  const checks = events.filter((e) => e.type === "critic_check");
  if (checks.length === 0) {
    return <p className="text-xs text-ink-600">No critic checks in this run.</p>;
  }
  const passed = checks.filter((e) => e.data.passed).length;
  return (
    <div>
      <div className="mb-2 flex items-center gap-2 text-2xs">
        <Badge tone={passed === checks.length ? "pos" : "neg"}>
          {passed}/{checks.length} passed
        </Badge>
      </div>
      <div className="space-y-1">
        {checks.map((e, i) => (
          <div
            key={i}
            className={`flex items-center gap-2 rounded px-2 py-1 text-2xs ${
              e.data.passed
                ? "bg-edge-pos/5 text-edge-pos"
                : "bg-edge-neg/10 text-edge-neg"
            }`}
          >
            <span className="font-mono">{e.data.passed ? "ok" : "!!"}</span>
            <span className="font-medium">{e.data.name as string}</span>
            <span className="ml-auto text-ink-600">
              {JSON.stringify(e.data.lhs)} vs {JSON.stringify(e.data.rhs)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function DagView({ events }: { events: AgentEvent[] }) {
  const planEvent = events.find((e) => e.type === "plan_built");
  if (!planEvent) {
    return (
      <p className="text-xs text-ink-600">
        No DAG plan in this run (workflow mode uses a fixed graph).
      </p>
    );
  }
  const dag = planEvent.data.dag as {
    id: string;
    kind: string;
    depends_on: string[];
  }[];
  const nodeExits = new Map(
    events
      .filter((e) => e.type === "node_exit")
      .map((e) => [e.data.node as string, e]),
  );
  return (
    <div className="flex flex-wrap gap-2">
      {dag.map((n) => {
        const exitKey = Object.keys(Object.fromEntries(nodeExits)).find((k) =>
          k.includes(n.id),
        );
        const done = !!exitKey;
        return (
          <div
            key={n.id}
            className={`rounded border px-3 py-2 text-2xs ${
              done
                ? "border-edge-pos/40 bg-edge-pos/5 text-edge-pos"
                : "border-line bg-surface-800 text-ink-400"
            }`}
          >
            <div className="font-semibold">
              {n.id}: {n.kind}
            </div>
            {n.depends_on.length > 0 && (
              <div className="text-ink-600">
                depends: {n.depends_on.join(", ")}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function EventLog({
  events,
  filter,
}: {
  events: AgentEvent[];
  filter: EventFilter;
}) {
  const filtered =
    filter === "all"
      ? events
      : events.filter((e) => eventCategory(e.type) === filter);
  return (
    <div className="max-h-80 overflow-y-auto font-mono text-2xs">
      {filtered.length === 0 && (
        <p className="py-4 text-center text-ink-600">No events match filter.</p>
      )}
      {filtered.map((e, i) => (
        <div
          key={i}
          className="flex gap-2 border-b border-line/50 px-1 py-0.5 hover:bg-surface-800"
        >
          <span className="tnum w-20 shrink-0 text-ink-600">
            {fmtTime(e.at_utc)}
          </span>
          <span
            className={`w-28 shrink-0 font-semibold ${
              e.type.includes("start") || e.type.includes("enter")
                ? "text-brand"
                : e.type.includes("fail") ||
                    (e.type === "tool_call_end" && !e.data.ok)
                  ? "text-edge-neg"
                  : "text-ink-100"
            }`}
          >
            {e.type}
          </span>
          <span className="min-w-0 truncate text-ink-400">
            {JSON.stringify(e.data)}
          </span>
        </div>
      ))}
    </div>
  );
}

function InjectionPanel({ events }: { events: AgentEvent[] }) {
  const toolCalls = events.filter((e) => e.type === "tool_call_end");
  const newsTools = toolCalls.filter(
    (e) => (e.data.server as string) === "news-sentiment",
  );
  if (newsTools.length === 0) {
    return (
      <p className="text-xs text-ink-600">
        No news/sentiment tool calls in this run. The injection panel shows
        how external text is quarantined and schema-validated before crossing
        the trust boundary.
      </p>
    );
  }
  return (
    <div className="space-y-2">
      <p className="text-2xs text-ink-400">
        External text from news/sentiment servers is quarantined: raw scraped
        content never enters the agent state directly. Each tool call returns
        schema-validated values only.
      </p>
      {newsTools.map((e, i) => (
        <div
          key={i}
          className="rounded border border-line bg-surface-800 px-3 py-2 text-2xs"
        >
          <div className="flex items-center gap-2">
            <span className="font-semibold text-ink-100">
              {e.data.tool as string}
            </span>
            <Badge tone={e.data.ok ? "pos" : "neg"}>
              {e.data.ok ? "validated" : "rejected"}
            </Badge>
            <span className="tnum ml-auto text-ink-600">
              {fmtMs((e.data.latency_ms as number) || 0)}
            </span>
          </div>
          {e.data.result_digest ? (
            <div className="mt-1 font-mono text-ink-600">
              result digest: {String(e.data.result_digest)}
            </div>
          ) : null}
          {e.data.error ? (
            <div className="mt-1 text-edge-neg">{String(e.data.error)}</div>
          ) : null}
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// main dashboard
// ---------------------------------------------------------------------------

export function InternalsDashboard() {
  const { data: runsData } = useSWR<{ runs: RunSummary[] }>(
    "/api/internals",
    jsonFetcher,
    { refreshInterval: 10_000 },
  );
  const runs = runsData?.runs ?? [];

  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [eventFilter, setEventFilter] = useState<EventFilter>("all");

  useEffect(() => {
    if (!selectedRun && runs.length > 0) {
      setSelectedRun(runs[0].thread_id);
    }
  }, [runs, selectedRun]);

  const { data: runDetail } = useSWR<RunDetail>(
    selectedRun ? `/api/internals/${selectedRun}` : null,
    jsonFetcher,
  );
  const events: AgentEvent[] = runDetail?.events ?? [];

  const summary = useMemo(() => {
    const trace = runDetail?.trace ?? {};
    return {
      elapsed: (trace.elapsed_ms as number) || 0,
      nCalls: (trace.n_calls as number) || 0,
      nFailed: (trace.n_failed as number) || 0,
      mode: (trace.mode as string) || "--",
      degraded: (trace.degraded as string[]) || [],
    };
  }, [runDetail]);

  const filterButtons: { key: EventFilter; label: string }[] = [
    { key: "all", label: "All" },
    { key: "node", label: "Nodes" },
    { key: "tool", label: "Tools" },
    { key: "belief", label: "Belief" },
    { key: "critic", label: "Critic" },
    { key: "other", label: "Other" },
  ];

  return (
    <div className="grid gap-4 lg:grid-cols-3">
      {/* left column: run picker + stats */}
      <div className="space-y-4 lg:col-span-1">
        <Panel title="Recent Runs">
          <RunPicker
            runs={runs}
            selected={selectedRun}
            onSelect={setSelectedRun}
          />
        </Panel>

        {selectedRun && (
          <div className="grid grid-cols-2 gap-2">
            <Stat label="Latency" value={fmtMs(summary.elapsed)} />
            <Stat label="Mode" value={summary.mode} />
            <Stat
              label="Tool Calls"
              value={`${summary.nCalls - summary.nFailed}/${summary.nCalls}`}
              hint={summary.nFailed > 0 ? `${summary.nFailed} failed` : undefined}
            />
            <Stat label="Events" value={String(events.length)} />
          </div>
        )}

        {summary.degraded.length > 0 && (
          <Panel title="Degradation Notes">
            <ul className="space-y-1 text-2xs text-ink-400">
              {summary.degraded.map((d, i) => (
                <li key={i} className="flex gap-1.5">
                  <span className="shrink-0 text-edge-neg">--</span>
                  {d}
                </li>
              ))}
            </ul>
          </Panel>
        )}
      </div>

      {/* right column: detail panels */}
      <div className="space-y-4 lg:col-span-2">
        <Panel title="Waterfall">
          <Waterfall events={events} />
        </Panel>

        <Panel title="Belief Evolution">
          <BeliefTrace events={events} />
        </Panel>

        <div className="grid gap-4 md:grid-cols-2">
          <Panel title="Critic Checklist">
            <CriticChecks events={events} />
          </Panel>
          <Panel title="DAG View">
            <DagView events={events} />
          </Panel>
        </div>

        <Panel title="Injection Visibility">
          <InjectionPanel events={events} />
        </Panel>

        <Panel
          title="Event Log"
          right={
            <div className="flex gap-1">
              {filterButtons.map((f) => (
                <button
                  key={f.key}
                  onClick={() => setEventFilter(f.key)}
                  className={`rounded px-1.5 py-0.5 text-2xs font-semibold transition-colors ${
                    eventFilter === f.key
                      ? "bg-brand/20 text-brand"
                      : "text-ink-600 hover:text-ink-400"
                  }`}
                >
                  {f.label}
                </button>
              ))}
            </div>
          }
        >
          <EventLog events={events} filter={eventFilter} />
        </Panel>
      </div>
    </div>
  );
}
