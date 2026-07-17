"use client";

import { useEffect, useRef, useState } from "react";
import useSWR from "swr";
import type { Health } from "./types";

export const jsonFetcher = (url: string) =>
  fetch(url).then((r) => {
    if (!r.ok) throw new Error(`${r.status}`);
    return r.json();
  });

/** Gateway liveness, SWR-cached and revalidated in the background. */
export function useHealth() {
  return useSWR<Health>("/api/health", jsonFetcher, {
    refreshInterval: 30_000,
    revalidateOnFocus: false,
  });
}

/** Debounce any changing value — inputs must never hit the agent per keystroke. */
export function useDebounced<T>(value: T, delayMs = 450): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const id = setTimeout(() => setDebounced(value), delayMs);
    return () => clearTimeout(id);
  }, [value, delayMs]);
  return debounced;
}

/**
 * Client-side submit throttle mirroring the gateway's 5/min rate limit:
 * refuses to fire more often than `minIntervalMs` and exposes the remaining
 * cooldown so the UI can show it instead of surfacing raw 429s.
 */
export function useCooldown(minIntervalMs = 12_000) {
  const lastFired = useRef(0);
  const [remainingMs, setRemainingMs] = useState(0);

  useEffect(() => {
    const id = setInterval(() => {
      const left = Math.max(0, lastFired.current + minIntervalMs - Date.now());
      setRemainingMs(left);
    }, 500);
    return () => clearInterval(id);
  }, [minIntervalMs]);

  return {
    ready: remainingMs === 0,
    remainingMs,
    fire: () => {
      lastFired.current = Date.now();
      setRemainingMs(minIntervalMs);
    },
  };
}
