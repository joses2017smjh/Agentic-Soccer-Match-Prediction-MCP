"use client";

/**
 * Parlay Builder: add legs, see correlated pricing in real time.
 *
 * The correlation analysis is the key feature -- sportsbooks multiply
 * independent odds, but legs within a match are correlated through the
 * Dixon-Coles grid.  This builder shows the exact joint probability,
 * the correlation factor, and the edge vs sportsbook pricing.
 */

import { useMemo, useState } from "react";
import { Badge, Panel, Skeleton, Stat } from "@/components/ui/panel";
import type { ParlayLegInput, ParlayResult } from "@/lib/types";

const LEG_TYPES = [
  { value: "outcome", label: "Match Result", selections: ["home", "draw", "away"] },
  { value: "over", label: "Over Goals", selections: ["over"] },
  { value: "under", label: "Under Goals", selections: ["under"] },
  { value: "btts", label: "Both Teams Score", selections: ["yes", "no"] },
  { value: "exact_score", label: "Exact Score", selections: [] },
  { value: "anytime_scorer", label: "Anytime Scorer", selections: [] },
];

const COMMON_LINES = [0.5, 1.5, 2.5, 3.5, 4.5];

function pct(x: number): string {
  return `${(x * 100).toFixed(1)}%`;
}

interface LegFormState {
  match_id: string;
  leg_type: string;
  selection: string;
  line: number;
  decimal_odds: number;
  team_side: string;
  xg_share: number;
}

const EMPTY_LEG: LegFormState = {
  match_id: "",
  leg_type: "outcome",
  selection: "home",
  line: 2.5,
  decimal_odds: 0,
  team_side: "home",
  xg_share: 0.15,
};

export function ParlayBuilder() {
  const [legs, setLegs] = useState<ParlayLegInput[]>([]);
  const [form, setForm] = useState<LegFormState>({ ...EMPTY_LEG });
  const [result, setResult] = useState<ParlayResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const legType = useMemo(
    () => LEG_TYPES.find((t) => t.value === form.leg_type),
    [form.leg_type],
  );

  function addLeg() {
    if (!form.match_id.trim()) return;
    const leg: ParlayLegInput = {
      match_id: form.match_id.trim(),
      leg_type: form.leg_type,
      selection: form.selection,
    };
    if (form.leg_type === "over" || form.leg_type === "under") {
      leg.line = form.line;
    }
    if (form.decimal_odds > 1) {
      leg.decimal_odds = form.decimal_odds;
    }
    if (form.leg_type === "anytime_scorer") {
      leg.team_side = form.team_side;
      leg.xg_share = form.xg_share;
    }
    setLegs((prev) => [...prev, leg]);
    setForm({ ...EMPTY_LEG, match_id: form.match_id });
    setResult(null);
  }

  function removeLeg(index: number) {
    setLegs((prev) => prev.filter((_, i) => i !== index));
    setResult(null);
  }

  async function priceParlay() {
    if (legs.length === 0 || busy) return;
    setBusy(true);
    setError(null);
    try {
      const res = await fetch("/api/parlay", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ legs }),
      });
      if (!res.ok) {
        const detail = await res.json().catch(() => null);
        setError(detail?.detail ?? `Error ${res.status}`);
      } else {
        setResult((await res.json()) as ParlayResult);
      }
    } catch {
      setError("Network error reaching the server.");
    } finally {
      setBusy(false);
    }
  }

  function legLabel(leg: ParlayLegInput): string {
    const parts = [leg.match_id, leg.leg_type];
    if (leg.leg_type === "over" || leg.leg_type === "under") {
      parts.push(`${leg.line ?? 2.5}`);
    }
    parts.push(leg.selection);
    if (leg.decimal_odds && leg.decimal_odds > 1) {
      parts.push(`@ ${leg.decimal_odds.toFixed(2)}`);
    }
    return parts.join(" / ");
  }

  return (
    <div className="flex flex-col gap-4">
      {/* ---- add a leg ---- */}
      <Panel title="Add a Leg">
        <div className="flex flex-col gap-3">
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
            <div>
              <label className="mb-1 block text-2xs uppercase text-ink-600">
                Match ID (e.g. ARS-MCI)
              </label>
              <input
                value={form.match_id}
                onChange={(e) => setForm({ ...form, match_id: e.target.value })}
                placeholder="ARS-MCI"
                className="w-full rounded border border-line bg-surface-800 px-2 py-1.5
                  text-sm text-ink-100 placeholder:text-ink-600
                  focus:border-brand focus:outline-none"
              />
            </div>
            <div>
              <label className="mb-1 block text-2xs uppercase text-ink-600">
                Leg Type
              </label>
              <select
                value={form.leg_type}
                onChange={(e) => {
                  const lt = e.target.value;
                  const ltDef = LEG_TYPES.find((t) => t.value === lt);
                  setForm({
                    ...form,
                    leg_type: lt,
                    selection: ltDef?.selections[0] ?? "",
                  });
                }}
                className="w-full rounded border border-line bg-surface-800 px-2 py-1.5
                  text-sm text-ink-100 focus:border-brand focus:outline-none"
              >
                {LEG_TYPES.map((t) => (
                  <option key={t.value} value={t.value}>{t.label}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="mb-1 block text-2xs uppercase text-ink-600">
                Selection
              </label>
              {legType && legType.selections.length > 0 ? (
                <select
                  value={form.selection}
                  onChange={(e) => setForm({ ...form, selection: e.target.value })}
                  className="w-full rounded border border-line bg-surface-800 px-2 py-1.5
                    text-sm text-ink-100 focus:border-brand focus:outline-none"
                >
                  {legType.selections.map((s) => (
                    <option key={s} value={s}>
                      {s.charAt(0).toUpperCase() + s.slice(1)}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  value={form.selection}
                  onChange={(e) => setForm({ ...form, selection: e.target.value })}
                  placeholder={form.leg_type === "exact_score" ? "2-1" : "Player name"}
                  className="w-full rounded border border-line bg-surface-800 px-2 py-1.5
                    text-sm text-ink-100 placeholder:text-ink-600
                    focus:border-brand focus:outline-none"
                />
              )}
            </div>
            <div>
              <label className="mb-1 block text-2xs uppercase text-ink-600">
                Odds (optional)
              </label>
              <input
                type="number"
                step="0.01"
                min="1.01"
                value={form.decimal_odds || ""}
                onChange={(e) =>
                  setForm({ ...form, decimal_odds: parseFloat(e.target.value) || 0 })
                }
                placeholder="2.10"
                className="w-full rounded border border-line bg-surface-800 px-2 py-1.5
                  text-sm text-ink-100 placeholder:text-ink-600
                  focus:border-brand focus:outline-none"
              />
            </div>
          </div>

          {/* conditional fields */}
          {(form.leg_type === "over" || form.leg_type === "under") && (
            <div className="flex items-center gap-2">
              <span className="text-xs text-ink-400">Line:</span>
              {COMMON_LINES.map((l) => (
                <button
                  key={l}
                  onClick={() => setForm({ ...form, line: l })}
                  className={`rounded border px-2 py-0.5 text-xs
                    ${form.line === l
                      ? "border-brand bg-brand/10 text-brand"
                      : "border-line text-ink-400 hover:text-ink-100"}`}
                >
                  {l}
                </button>
              ))}
            </div>
          )}

          {form.leg_type === "anytime_scorer" && (
            <div className="flex gap-3">
              <div>
                <label className="mb-1 block text-2xs uppercase text-ink-600">Team</label>
                <select
                  value={form.team_side}
                  onChange={(e) => setForm({ ...form, team_side: e.target.value })}
                  className="rounded border border-line bg-surface-800 px-2 py-1.5
                    text-sm text-ink-100 focus:border-brand focus:outline-none"
                >
                  <option value="home">Home</option>
                  <option value="away">Away</option>
                </select>
              </div>
              <div>
                <label className="mb-1 block text-2xs uppercase text-ink-600">
                  xG Share (0-1)
                </label>
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  max="1"
                  value={form.xg_share}
                  onChange={(e) =>
                    setForm({ ...form, xg_share: parseFloat(e.target.value) || 0 })
                  }
                  className="w-24 rounded border border-line bg-surface-800 px-2 py-1.5
                    text-sm text-ink-100 focus:border-brand focus:outline-none"
                />
              </div>
            </div>
          )}

          <button
            onClick={addLeg}
            disabled={!form.match_id.trim()}
            className="self-start rounded-md bg-brand px-4 py-2 text-sm font-semibold
              text-surface-950 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Add Leg
          </button>
        </div>
      </Panel>

      {/* ---- current legs ---- */}
      {legs.length > 0 && (
        <Panel
          title={`Parlay Slip — ${legs.length} leg${legs.length > 1 ? "s" : ""}`}
          right={
            <button
              onClick={priceParlay}
              disabled={busy}
              className="rounded-md bg-brand px-3 py-1.5 text-xs font-semibold
                text-surface-950 disabled:opacity-40"
            >
              {busy ? "Pricing..." : "Price Parlay"}
            </button>
          }
        >
          <ul className="flex flex-col gap-2">
            {legs.map((leg, i) => (
              <li
                key={i}
                className="flex items-center justify-between rounded border
                  border-line bg-surface-800/60 px-3 py-2 text-xs"
              >
                <span className="text-ink-100">{legLabel(leg)}</span>
                <button
                  onClick={() => removeLeg(i)}
                  className="ml-2 text-ink-600 hover:text-edge-neg"
                >
                  remove
                </button>
              </li>
            ))}
          </ul>
          {error && <p className="mt-2 text-xs text-edge-neg">{error}</p>}
        </Panel>
      )}

      {busy && !result && (
        <div className="grid gap-4 md:grid-cols-4">
          <Skeleton className="h-20" />
          <Skeleton className="h-20" />
          <Skeleton className="h-20" />
          <Skeleton className="h-20" />
        </div>
      )}

      {/* ---- pricing result ---- */}
      {result && (
        <>
          <div className="grid gap-4 md:grid-cols-4">
            <Stat
              label="Joint Probability"
              value={pct(result.joint_probability)}
              hint="exact from grid"
            />
            <Stat
              label="Independent Prob"
              value={pct(result.independent_probability)}
              hint="product of marginals"
            />
            <Stat
              label="Correlation Factor"
              value={result.correlation_factor.toFixed(3)}
              hint={result.correlation_direction}
            />
            <Stat
              label="Model Fair Odds"
              value={result.model_fair_odds.toFixed(2)}
              hint={
                result.sportsbook_parlay_odds > 0
                  ? `vs book ${result.sportsbook_parlay_odds.toFixed(2)}`
                  : "no book odds"
              }
            />
          </div>

          {result.sportsbook_parlay_odds > 0 && (
            <div className="grid gap-4 md:grid-cols-3">
              <Stat
                label="Edge"
                value={`${(result.edge * 100).toFixed(1)}%`}
                hint={result.edge > 0 ? "model favours you" : "book has the edge"}
              />
              <Stat
                label="Expected Value"
                value={`${(result.ev * 100).toFixed(1)}%`}
                hint="per unit staked"
              />
              <Stat
                label="Kelly Fraction"
                value={result.kelly_fraction.toFixed(3)}
                hint="fractional Kelly"
              />
            </div>
          )}

          {/* per-leg breakdown */}
          <Panel title="Leg Breakdown">
            <div className="overflow-x-auto">
              <table className="tnum w-full text-left text-xs">
                <thead className="text-ink-600">
                  <tr className="border-b border-line">
                    <th className="py-1.5 pr-4 font-medium">Leg</th>
                    <th className="py-1.5 pr-4 font-medium">Model Prob</th>
                    <th className="py-1.5 pr-4 font-medium">Book Implied</th>
                    <th className="py-1.5 font-medium">Odds</th>
                  </tr>
                </thead>
                <tbody className="text-ink-400">
                  {result.legs.map((lr, i) => (
                    <tr key={i} className="border-b border-line/50">
                      <td className="py-1.5 pr-4 text-ink-100">
                        {lr.match_id} / {lr.leg_type} / {lr.selection}
                      </td>
                      <td className="py-1.5 pr-4">{pct(lr.marginal_prob)}</td>
                      <td className="py-1.5 pr-4">
                        {lr.implied_prob > 0 ? pct(lr.implied_prob) : "--"}
                      </td>
                      <td className="py-1.5">
                        {lr.decimal_odds > 0 ? lr.decimal_odds.toFixed(2) : "--"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>

          {/* correlation analysis */}
          {result.match_groups.some((mg) => mg.n_legs > 1) && (
            <Panel
              title="Correlation Analysis"
              right={
                <Badge
                  tone={
                    result.correlation_direction === "positive"
                      ? "pos"
                      : result.correlation_direction === "negative"
                        ? "neg"
                        : "neutral"
                  }
                >
                  {result.correlation_direction}
                </Badge>
              }
            >
              <div className="overflow-x-auto">
                <table className="tnum w-full text-left text-xs">
                  <thead className="text-ink-600">
                    <tr className="border-b border-line">
                      <th className="py-1.5 pr-4 font-medium">Match</th>
                      <th className="py-1.5 pr-4 font-medium">Legs</th>
                      <th className="py-1.5 pr-4 font-medium">Joint</th>
                      <th className="py-1.5 pr-4 font-medium">Independent</th>
                      <th className="py-1.5 font-medium">Corr. Factor</th>
                    </tr>
                  </thead>
                  <tbody className="text-ink-400">
                    {result.match_groups
                      .filter((mg) => mg.n_legs > 1)
                      .map((mg, i) => (
                        <tr key={i} className="border-b border-line/50">
                          <td className="py-1.5 pr-4 text-ink-100">{mg.match_id}</td>
                          <td className="py-1.5 pr-4">{mg.n_legs}</td>
                          <td className="py-1.5 pr-4">{pct(mg.joint_prob)}</td>
                          <td className="py-1.5 pr-4">{pct(mg.independent_prob)}</td>
                          <td className="py-1.5">
                            <span
                              className={
                                mg.correlation_factor > 1.02
                                  ? "text-edge-pos"
                                  : mg.correlation_factor < 0.98
                                    ? "text-edge-neg"
                                    : ""
                              }
                            >
                              {mg.correlation_factor.toFixed(3)}
                            </span>
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </Panel>
          )}

          {/* text analysis */}
          <Panel title="Analysis">
            <p className="whitespace-pre-wrap text-sm leading-6 text-ink-300">
              {result.analysis}
            </p>
          </Panel>
        </>
      )}
    </div>
  );
}
