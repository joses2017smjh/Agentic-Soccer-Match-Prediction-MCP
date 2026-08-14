"use client";

import { ParlayBuilder } from "@/components/features/parlay/parlay-builder";
import { Nav } from "@/components/ui/nav";

export default function ParlayPage() {
  return (
    <>
      <Nav />
      <main className="min-w-0 flex-1 py-6">
        <div className="mb-4">
          <h1 className="text-xl font-bold tracking-tight">Parlay Builder</h1>
          <p className="mt-1 text-xs text-ink-400">
            Correlated-parlay pricing from the Dixon-Coles grid. Same-match
            legs are priced jointly -- the correlation factor shows how much the
            sportsbook's independent multiplication over- or under-prices your
            parlay.
          </p>
        </div>
        <ParlayBuilder />
      </main>
      <footer className="border-t border-line py-3 text-2xs text-ink-600">
        Correlation analysis uses the bivariate Poisson grid. Not betting
        advice -- research tool for the independent-vs-joint pricing study.
      </footer>
    </>
  );
}
