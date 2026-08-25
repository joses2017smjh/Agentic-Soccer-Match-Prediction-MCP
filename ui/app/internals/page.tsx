"use client";

import { Nav } from "@/components/ui/nav";
import { InternalsDashboard } from "@/components/features/internals/internals-dashboard";

export default function InternalsPage() {
  return (
    <>
      <Nav />
      <main className="min-w-0 flex-1 py-6">
        <div className="mb-6">
          <h1 className="text-xl font-bold tracking-tight">Agent Internals</h1>
          <p className="mt-1 text-xs text-ink-400">
            Real-time event trace, belief evolution, critic checks, and tool
            call waterfall -- replay any past run or stream a live one.
          </p>
        </div>
        <InternalsDashboard />
      </main>
    </>
  );
}
