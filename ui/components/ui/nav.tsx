"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Badge } from "@/components/ui/panel";
import { useHealth } from "@/lib/hooks";

const LINKS = [
  { href: "/", label: "Leagues" },
  { href: "/bracket", label: "WWC Bracket" },
  { href: "/predict", label: "Ask the Agent" },
  { href: "/parlay", label: "Parlay Builder" },
  { href: "/evaluation", label: "Evaluation" },
];

export function Nav() {
  const pathname = usePathname();
  const { data: health, isLoading } = useHealth();
  // mobile: logo + status on row one, links wrap onto row two.
  // desktop: logo and links left, status right.
  // free-tier backends sleep when idle, so the first visit is a cold start,
  // not a failure — showing "offline" there reads as broken when it isn't
  const status = isLoading ? (
    <Badge tone="neutral">
      <span className="sm:hidden">waking…</span>
      <span className="hidden sm:inline">waking the backend…</span>
    </Badge>
  ) : health?.ok ? (
    <Badge tone="pos">
      <span className="sm:hidden">live</span>
      <span className="hidden sm:inline">gateway up · {health.model_version}</span>
    </Badge>
  ) : (
    <Badge tone="neg">
      <span className="sm:hidden">offline</span>
      <span className="hidden sm:inline">gateway offline</span>
    </Badge>
  );

  return (
    <header
      className="flex flex-col gap-2 border-b border-line py-3
        sm:flex-row sm:items-center sm:justify-between sm:py-4"
    >
      <div className="flex min-w-0 items-center justify-between gap-3
        sm:items-baseline sm:gap-6">
        <Link href="/" className="shrink-0 text-lg font-bold tracking-tight">
          Match<span className="text-brand">Intel</span>
        </Link>
        <nav className="hidden gap-4 sm:flex">
          {LINKS.map((l) => {
            const active = l.href === "/" ? pathname === "/" : pathname.startsWith(l.href);
            return (
              <Link
                key={l.href}
                href={l.href}
                className={`whitespace-nowrap text-xs font-semibold uppercase
                  tracking-widest
                  ${active ? "text-ink-100" : "text-ink-600 hover:text-ink-400"}`}
              >
                {l.label}
              </Link>
            );
          })}
        </nav>
        <span className="sm:hidden">{status}</span>
      </div>

      {/* mobile link row: full-width tap targets, no wrapping mid-label */}
      <nav className="flex gap-2 sm:hidden">
        {LINKS.map((l) => {
          const active = l.href === "/" ? pathname === "/" : pathname.startsWith(l.href);
          return (
            <Link
              key={l.href}
              href={l.href}
              className={`flex-1 whitespace-nowrap rounded border px-2 py-1.5
                text-center text-2xs font-semibold uppercase tracking-wider
                ${active
                  ? "border-line-strong bg-surface-800 text-ink-100"
                  : "border-line text-ink-400"}`}
            >
              {l.label}
            </Link>
          );
        })}
      </nav>

      <span className="hidden sm:inline">{status}</span>
    </header>
  );
}
