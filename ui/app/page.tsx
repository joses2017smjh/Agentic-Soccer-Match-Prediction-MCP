"use client";

import { PredictConsole } from "@/components/features/prediction/predict-console";
import { Badge } from "@/components/ui/panel";
import { useHealth } from "@/lib/hooks";

export default function Home() {
  const { data: health } = useHealth();

  return (
    <>
      <header className="flex items-center justify-between border-b border-line py-4">
        <div className="flex items-baseline gap-3">
          <h1 className="text-lg font-bold tracking-tight">
            Match<span className="text-brand">Intel</span>
          </h1>
          <span className="text-2xs uppercase tracking-widest text-ink-600">
            agentic soccer prediction
          </span>
        </div>
        <div className="flex items-center gap-2">
          {health?.ok ? (
            <Badge tone="pos">gateway up · {health.model_version}</Badge>
          ) : (
            <Badge tone="neg">gateway offline</Badge>
          )}
        </div>
      </header>

      <main className="flex-1 py-6">
        <PredictConsole />
      </main>

      <footer className="border-t border-line py-3 text-2xs text-ink-600">
        Model probabilities include coverage-guaranteed uncertainty sets;
        staking suggestions always require human approval. Demo model — not
        betting advice.
      </footer>
    </>
  );
}
