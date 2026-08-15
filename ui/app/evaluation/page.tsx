"use client";

import { Nav } from "@/components/ui/nav";
import { EvalDashboard } from "@/components/features/evaluation/eval-dashboard";

export default function EvaluationPage() {
  return (
    <>
      <Nav />
      <main className="min-w-0 flex-1 py-6">
        <div className="mb-6">
          <h1 className="text-xl font-bold tracking-tight">Evaluation Harness</h1>
          <p className="mt-1 text-xs text-ink-400">
            Mixed-verifier evaluation, stage-wise attribution, and trajectory
            failure taxonomy -- inspired by Halluminate's Westworld verifiers,
            DealTrace pipeline decomposition, and Diligence Bench.
          </p>
        </div>
        <EvalDashboard />
      </main>
    </>
  );
}
