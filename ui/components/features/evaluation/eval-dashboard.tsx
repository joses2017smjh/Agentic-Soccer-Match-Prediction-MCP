"use client";

import { useState } from "react";
import { Badge, Panel, Stat } from "@/components/ui/panel";
import type { AttributionReport, EvaluationResult, TrajectoryReport } from "@/lib/types";

function pct(x: number): string {
  return `${(x * 100).toFixed(1)}%`;
}

const SEVERITY_COLORS: Record<string, string> = {
  high: "text-edge-neg",
  medium: "text-amber-400",
  low: "text-ink-400",
};

const FAILURE_LABELS: Record<string, string> = {
  instruction_loss: "Instruction Loss",
  risk_abandonment: "Risk Abandonment",
  false_verification: "False Verification",
  silent_scope_drop: "Silent Scope Drop",
  detrimental_looping: "Detrimental Looping",
};

export function EvalDashboard() {
  const [evalResult, setEvalResult] = useState<EvaluationResult | null>(null);
  const [attrReport, setAttrReport] = useState<AttributionReport | null>(null);
  const [trajReport, setTrajReport] = useState<TrajectoryReport | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"harness" | "attribution" | "trajectory">("harness");

  // demo evaluation: run a sample through the harness
  async function runDemoEval() {
    setBusy(true);
    setError(null);
    try {
      const res = await fetch("/api/evaluation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: "evaluate",
          match: {
            match_id: "DEMO-ARS-CHE",
            prediction: {
              match_outcome: {
                home: 0.55, draw: 0.25, away: 0.20,
                conformal_set: ["home", "draw"], conformal_alpha: 0.1,
              },
              expected_goals: { home: 1.7, away: 1.1 },
              exact_score: {
                top_scorelines: [
                  { score: "2-1", prob: 0.13 }, { score: "1-0", prob: 0.12 },
                  { score: "1-1", prob: 0.11 }, { score: "2-0", prob: 0.10 },
                  { score: "0-1", prob: 0.08 },
                ],
                over_under_2_5: { over: 0.52, under: 0.48 },
                btts: { yes: 0.48, no: 0.52 },
              },
            },
            actual_outcome: "home",
            actual_home_goals: 2,
            actual_away_goals: 1,
          },
        }),
      });
      if (!res.ok) {
        setError(`Error ${res.status}`);
      } else {
        setEvalResult(await res.json());
      }
    } catch {
      setError("Network error.");
    } finally {
      setBusy(false);
    }
  }

  async function runDemoAttribution() {
    setBusy(true);
    setError(null);
    try {
      const matches = [
        { match_id: "M1", prediction: { match_outcome: { home: 0.55, draw: 0.25, away: 0.20, conformal_set: ["home"] }, expected_goals: { home: 1.7, away: 1.1 }, exact_score: { top_scorelines: [{ score: "2-1", prob: 0.13 }], over_under_2_5: { over: 0.55, under: 0.45 }, btts: { yes: 0.50, no: 0.50 } } }, actual_outcome: "home", actual_home_goals: 2, actual_away_goals: 1 },
        { match_id: "M2", prediction: { match_outcome: { home: 0.30, draw: 0.35, away: 0.35, conformal_set: ["draw", "away"] }, expected_goals: { home: 1.1, away: 1.4 }, exact_score: { top_scorelines: [{ score: "1-1", prob: 0.12 }], over_under_2_5: { over: 0.48, under: 0.52 }, btts: { yes: 0.55, no: 0.45 } } }, actual_outcome: "draw", actual_home_goals: 1, actual_away_goals: 1 },
        { match_id: "M3", prediction: { match_outcome: { home: 0.45, draw: 0.30, away: 0.25, conformal_set: ["home", "draw"] }, expected_goals: { home: 1.5, away: 1.2 }, exact_score: { top_scorelines: [{ score: "1-0", prob: 0.14 }], over_under_2_5: { over: 0.50, under: 0.50 }, btts: { yes: 0.45, no: 0.55 } } }, actual_outcome: "away", actual_home_goals: 0, actual_away_goals: 1 },
        { match_id: "M4", prediction: { match_outcome: { home: 0.60, draw: 0.22, away: 0.18, conformal_set: ["home"] }, expected_goals: { home: 2.0, away: 0.9 }, exact_score: { top_scorelines: [{ score: "2-0", prob: 0.15 }], over_under_2_5: { over: 0.58, under: 0.42 }, btts: { yes: 0.42, no: 0.58 } } }, actual_outcome: "home", actual_home_goals: 3, actual_away_goals: 0 },
        { match_id: "M5", prediction: { match_outcome: { home: 0.35, draw: 0.30, away: 0.35, conformal_set: ["home", "draw", "away"] }, expected_goals: { home: 1.3, away: 1.3 }, exact_score: { top_scorelines: [{ score: "1-1", prob: 0.11 }], over_under_2_5: { over: 0.50, under: 0.50 }, btts: { yes: 0.52, no: 0.48 } } }, actual_outcome: "draw", actual_home_goals: 2, actual_away_goals: 2 },
      ];
      const res = await fetch("/api/evaluation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "attribution", matches }),
      });
      if (!res.ok) setError(`Error ${res.status}`);
      else setAttrReport(await res.json());
    } catch {
      setError("Network error.");
    } finally {
      setBusy(false);
    }
  }

  async function runTrajectoryAnalysis() {
    setBusy(true);
    setError(null);
    try {
      const res = await fetch("/api/evaluation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "trajectory" }),
      });
      if (!res.ok) {
        const detail = await res.json().catch(() => null);
        setError(detail?.detail ?? `Error ${res.status}`);
      } else {
        setTrajReport(await res.json());
      }
    } catch {
      setError("Network error.");
    } finally {
      setBusy(false);
    }
  }

  const tabs = [
    { key: "harness" as const, label: "Mixed Verifiers" },
    { key: "attribution" as const, label: "Stage Attribution" },
    { key: "trajectory" as const, label: "Trajectory Analysis" },
  ];

  return (
    <div className="flex flex-col gap-4">
      {/* tab bar */}
      <div className="flex gap-2">
        {tabs.map((t) => (
          <button
            key={t.key}
            onClick={() => setActiveTab(t.key)}
            className={`rounded border px-3 py-1.5 text-xs font-semibold uppercase tracking-wider
              ${activeTab === t.key
                ? "border-brand bg-brand/10 text-brand"
                : "border-line text-ink-400 hover:text-ink-100"}`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {error && <p className="text-xs text-edge-neg">{error}</p>}

      {/* ---- Mixed Verifier Harness ---- */}
      {activeTab === "harness" && (
        <>
          <Panel
            title="Mixed-Verifier Evaluation (Westworld-style)"
            right={
              <button
                onClick={runDemoEval}
                disabled={busy}
                className="rounded-md bg-brand px-3 py-1.5 text-xs font-semibold
                  text-surface-950 disabled:opacity-40"
              >
                {busy ? "Running..." : "Run Demo Evaluation"}
              </button>
            }
          >
            <p className="text-xs text-ink-400">
              Three verifier types: state-based (outcome match), component-level
              (O/U, BTTS, score range), and ground-truth matching (Brier, log-loss, xG).
              Weighted composite score with reward-hacking floor test.
            </p>
          </Panel>

          {evalResult && (
            <>
              <div className="grid gap-4 md:grid-cols-4">
                <Stat label="Composite Score" value={pct(evalResult.composite_score)} hint="weighted 0-1" />
                {evalResult.verifiers.map((v) => (
                  <Stat
                    key={v.name}
                    label={v.name.charAt(0).toUpperCase() + v.name.slice(1)}
                    value={pct(v.score)}
                    hint={v.passed ? "passed" : "failed"}
                  />
                ))}
              </div>

              <Panel
                title="Verifier Details"
                right={
                  <Badge tone={evalResult.reward_hacking_safe ? "pos" : "neg"}>
                    {evalResult.reward_hacking_safe ? "reward-hack safe" : "hackable"}
                  </Badge>
                }
              >
                <div className="overflow-x-auto">
                  <table className="tnum w-full text-left text-xs">
                    <thead className="text-ink-600">
                      <tr className="border-b border-line">
                        <th className="py-1.5 pr-4 font-medium">Verifier</th>
                        <th className="py-1.5 pr-4 font-medium">Score</th>
                        <th className="py-1.5 pr-4 font-medium">Status</th>
                        <th className="py-1.5 font-medium">Weight</th>
                      </tr>
                    </thead>
                    <tbody className="text-ink-400">
                      {evalResult.verifiers.map((v) => (
                        <tr key={v.name} className="border-b border-line/50">
                          <td className="py-1.5 pr-4 text-ink-100">{v.name}</td>
                          <td className="py-1.5 pr-4">{pct(v.score)}</td>
                          <td className="py-1.5 pr-4">
                            <Badge tone={v.passed ? "pos" : "neg"}>
                              {v.passed ? "pass" : "fail"}
                            </Badge>
                          </td>
                          <td className="py-1.5">
                            {((evalResult.details as Record<string, Record<string, number>>)?.weights?.[v.name] ?? 0) * 100}%
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Panel>

              <Panel title="Reward-Hacking Floor Test">
                <div className="grid gap-4 md:grid-cols-3">
                  <Stat
                    label="Empty Submission"
                    value={evalResult.reward_hacking_test.empty_score.toFixed(3)}
                    hint={evalResult.reward_hacking_test.empty_below_floor ? "below floor" : "ABOVE floor"}
                  />
                  <Stat
                    label="Uniform Submission"
                    value={evalResult.reward_hacking_test.uniform_score.toFixed(3)}
                    hint={evalResult.reward_hacking_test.uniform_below_floor ? "below floor" : "ABOVE floor"}
                  />
                  <Stat
                    label="Floor Threshold"
                    value={evalResult.reward_hacking_test.floor.toFixed(2)}
                    hint="trivial submissions must score below"
                  />
                </div>
              </Panel>
            </>
          )}
        </>
      )}

      {/* ---- Stage Attribution ---- */}
      {activeTab === "attribution" && (
        <>
          <Panel
            title="Stage-wise Attribution (DealTrace-style)"
            right={
              <button
                onClick={runDemoAttribution}
                disabled={busy}
                className="rounded-md bg-brand px-3 py-1.5 text-xs font-semibold
                  text-surface-950 disabled:opacity-40"
              >
                {busy ? "Running..." : "Run Attribution (5 matches)"}
              </button>
            }
          >
            <p className="text-xs text-ink-400">
              Decomposes prediction quality by pipeline stage. Identifies the
              binding constraint -- the stage whose improvement would most
              move the composite score, like DealTrace's forecast beta=0.75.
            </p>
          </Panel>

          {attrReport && (
            <>
              <div className="grid gap-4 md:grid-cols-4">
                <Stat label="Mean Composite" value={pct(attrReport.mean_composite)} hint={`n=${attrReport.n_evaluated}`} />
                <Stat label="Outcome Accuracy" value={pct(attrReport.calibration_summary.outcome_accuracy)} hint="argmax correct" />
                <Stat label="Conformal Coverage" value={pct(attrReport.calibration_summary.conformal_coverage)} hint="vs 0.90 target" />
                <Stat label="Mean Brier" value={attrReport.calibration_summary.mean_brier.toFixed(3)} hint="lower is better" />
              </div>

              <Panel
                title="Stage Contributions"
                right={
                  <Badge tone="brand">
                    binding: {attrReport.binding_constraint}
                  </Badge>
                }
              >
                <div className="overflow-x-auto">
                  <table className="tnum w-full text-left text-xs">
                    <thead className="text-ink-600">
                      <tr className="border-b border-line">
                        <th className="py-1.5 pr-4 font-medium">Stage</th>
                        <th className="py-1.5 pr-4 font-medium">Mean Score</th>
                        <th className="py-1.5 pr-4 font-medium">Std Dev</th>
                        <th className="py-1.5 pr-4 font-medium">Corr w/ Composite</th>
                        <th className="py-1.5 pr-4 font-medium">Weight</th>
                        <th className="py-1.5 font-medium">Weighted Contrib</th>
                      </tr>
                    </thead>
                    <tbody className="text-ink-400">
                      {attrReport.stages.map((s) => (
                        <tr
                          key={s.stage}
                          className={`border-b border-line/50 ${
                            s.stage === attrReport.binding_constraint ? "bg-brand/5" : ""
                          }`}
                        >
                          <td className="py-1.5 pr-4 text-ink-100">
                            {s.stage}
                            {s.stage === attrReport.binding_constraint && (
                              <span className="ml-2 text-2xs text-brand">binding</span>
                            )}
                          </td>
                          <td className="py-1.5 pr-4">{pct(s.mean_score)}</td>
                          <td className="py-1.5 pr-4">{s.std_score.toFixed(3)}</td>
                          <td className="py-1.5 pr-4">
                            <span className={s.correlation_with_composite > 0.5 ? "text-edge-pos" : ""}>
                              {s.correlation_with_composite.toFixed(3)}
                            </span>
                          </td>
                          <td className="py-1.5 pr-4">{pct(s.weight)}</td>
                          <td className="py-1.5">{pct(s.weighted_contribution)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Panel>

              <Panel title="Per-Match Breakdown">
                <div className="overflow-x-auto">
                  <table className="tnum w-full text-left text-xs">
                    <thead className="text-ink-600">
                      <tr className="border-b border-line">
                        <th className="py-1.5 pr-4 font-medium">Match</th>
                        <th className="py-1.5 pr-4 font-medium">Composite</th>
                        <th className="py-1.5 pr-4 font-medium">State</th>
                        <th className="py-1.5 pr-4 font-medium">Component</th>
                        <th className="py-1.5 font-medium">Calibration</th>
                      </tr>
                    </thead>
                    <tbody className="text-ink-400">
                      {attrReport.details.per_match.map((m) => (
                        <tr key={m.match_id} className="border-b border-line/50">
                          <td className="py-1.5 pr-4 text-ink-100">{m.match_id}</td>
                          <td className="py-1.5 pr-4">{pct(m.composite)}</td>
                          <td className="py-1.5 pr-4">{pct(m.per_verifier.state ?? 0)}</td>
                          <td className="py-1.5 pr-4">{pct(m.per_verifier.component ?? 0)}</td>
                          <td className="py-1.5">{pct(m.per_verifier.calibration ?? 0)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Panel>
            </>
          )}
        </>
      )}

      {/* ---- Trajectory Analysis ---- */}
      {activeTab === "trajectory" && (
        <>
          <Panel
            title="Trajectory Failure Taxonomy (Diligence Bench-style)"
            right={
              <button
                onClick={runTrajectoryAnalysis}
                disabled={busy}
                className="rounded-md bg-brand px-3 py-1.5 text-xs font-semibold
                  text-surface-950 disabled:opacity-40"
              >
                {busy ? "Analyzing..." : "Analyze Latest Trajectory"}
              </button>
            }
          >
            <p className="text-xs text-ink-400">
              Classifies agent trajectory failures into five categories from
              the Diligence Bench: instruction loss over horizon (42.4%),
              risk abandonment, false verification (15.4%), silent scope drop,
              and detrimental looping.
            </p>
          </Panel>

          {trajReport && (
            <>
              <div className="grid gap-4 md:grid-cols-4">
                <Stat label="Quality Score" value={pct(trajReport.trajectory_quality)} hint="1.0 = no failures" />
                <Stat label="Tool Calls" value={String(trajReport.n_tool_calls)} hint={`${trajReport.n_failed_calls} failed`} />
                <Stat label="Failures Found" value={String(trajReport.failures.length)} hint="across all categories" />
                <Stat label="Elapsed" value={`${(trajReport.elapsed_ms / 1000).toFixed(1)}s`} hint="total run time" />
              </div>

              <Panel title="Failure Taxonomy">
                <div className="overflow-x-auto">
                  <table className="tnum w-full text-left text-xs">
                    <thead className="text-ink-600">
                      <tr className="border-b border-line">
                        <th className="py-1.5 pr-4 font-medium">Category</th>
                        <th className="py-1.5 pr-4 font-medium">Count</th>
                        <th className="py-1.5 font-medium">Diligence Bench Ref</th>
                      </tr>
                    </thead>
                    <tbody className="text-ink-400">
                      {Object.entries(trajReport.failure_counts).map(([cat, count]) => (
                        <tr key={cat} className="border-b border-line/50">
                          <td className="py-1.5 pr-4 text-ink-100">
                            {FAILURE_LABELS[cat] ?? cat}
                          </td>
                          <td className="py-1.5 pr-4">
                            <span className={count > 0 ? "text-edge-neg" : "text-edge-pos"}>
                              {count}
                            </span>
                          </td>
                          <td className="py-1.5 text-ink-600">
                            {cat === "instruction_loss" && "42.4% of capability failures"}
                            {cat === "false_verification" && "15.4% of capability failures"}
                            {cat === "risk_abandonment" && "7-29% carry-through rate"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Panel>

              {trajReport.failures.length > 0 && (
                <Panel title="Failure Details">
                  <div className="flex flex-col gap-2">
                    {trajReport.failures.map((f, i) => (
                      <div
                        key={i}
                        className="rounded border border-line bg-surface-800/60 px-3 py-2"
                      >
                        <div className="flex items-center gap-2">
                          <Badge tone={f.severity === "high" ? "neg" : "neutral"}>
                            {f.severity}
                          </Badge>
                          <span className="text-xs font-semibold text-ink-100">
                            {FAILURE_LABELS[f.category] ?? f.category}
                          </span>
                        </div>
                        <p className="mt-1 text-xs text-ink-400">{f.evidence}</p>
                      </div>
                    ))}
                  </div>
                </Panel>
              )}
            </>
          )}
        </>
      )}
    </div>
  );
}
