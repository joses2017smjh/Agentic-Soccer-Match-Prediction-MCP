/**
 * Step 3 component stubs. Each declares its data contract now so the page
 * layout, grid slots, and props are stable before the real visualizations
 * land (Dixon–Coles heatmap, conformal visualizer, headline timeline).
 */
import { Panel } from "@/components/ui/panel";
import type {
  HeadlineScenario,
  MatchOutcome,
  Scoreline,
} from "@/lib/types";

function Stub({ note }: { note: string }) {
  return (
    <div className="flex h-40 items-center justify-center rounded border
      border-dashed border-line-strong text-2xs uppercase tracking-widest
      text-ink-600">
      {note}
    </div>
  );
}

export function ScorelineHeatmap({
  scorelines,
}: {
  scorelines: Scoreline[];
}) {
  return (
    <Panel title="Exact score — Dixon-Coles grid">
      <Stub note={`heatmap lands in step 3 · ${scorelines.length} scorelines ready`} />
    </Panel>
  );
}

export function ConformalVisualizer({
  outcome,
  home,
  away,
}: {
  outcome: MatchOutcome;
  home: string;
  away: string;
}) {
  return (
    <Panel title={`Uncertainty — ${(1 - outcome.conformal_alpha) * 100}% coverage`}>
      <Stub
        note={`conformal visualizer lands in step 3 · set = [${outcome.conformal_set.join(", ")}] · ${home} vs ${away}`}
      />
    </Panel>
  );
}

export function HeadlineTimeline({
  scenario,
  home,
  away,
}: {
  scenario: HeadlineScenario;
  home: string;
  away: string;
}) {
  return (
    <Panel title="Headline scenario">
      <Stub
        note={`timeline lands in step 3 · ${home} ${scenario.scoreline} ${away} (${Math.round(scenario.probability * 100)}%)`}
      />
    </Panel>
  );
}
