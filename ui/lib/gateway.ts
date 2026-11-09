/**
 * Server-only gateway client used by Route Handlers.
 *
 * GATEWAY_URL and GATEWAY_API_KEY live in server env (.env.local / Compose)
 * and are attached here — the browser only ever talks to /api/* on this app,
 * so the FastAPI origin and credentials are never exposed client-side.
 */
import "server-only";

import { headers as incomingHeaders } from "next/headers";

const GATEWAY_URL = process.env.GATEWAY_URL ?? "http://localhost:8000";

/**
 * The address of the browser behind the current request, passed to the
 * gateway so it can rate-limit per client: without it every user shares one
 * bucket, because the gateway only ever sees this server's address. The
 * gateway honours the header only on requests that carry the API key.
 *
 * Trust: this is only as reliable as whatever sits in front of this server.
 * x-real-ip is set by the host (Vercel, or an nginx proxy_set_header), so it
 * is preferred. Failing that, the LAST hop of x-forwarded-for is the address
 * the nearest proxy saw; the first hop is whatever the browser claimed. If
 * this server is reachable with no proxy in front, both headers are
 * client-supplied and a caller can evade per-client limiting.
 */
async function clientAddress(): Promise<string | null> {
  try {
    const incoming = await incomingHeaders();
    const realIp = incoming.get("x-real-ip")?.trim();
    if (realIp) return realIp;
    const hops = (incoming.get("x-forwarded-for") ?? "")
      .split(",")
      .map((hop) => hop.trim())
      .filter(Boolean);
    return hops.length ? hops[hops.length - 1] : null;
  } catch {
    return null; // not inside a request (build step, scripts)
  }
}

export async function gatewayFetch(
  path: string,
  init: RequestInit = {},
): Promise<Response> {
  const headers = new Headers(init.headers);
  headers.set("Content-Type", "application/json");
  // ngrok free tier returns an interstitial HTML page unless this header is set
  headers.set("ngrok-skip-browser-warning", "true");
  const key = process.env.GATEWAY_API_KEY;
  if (key) headers.set("X-API-Key", key);
  const client = await clientAddress();
  if (client) headers.set("X-Client-Address", client);

  return fetch(`${GATEWAY_URL}${path}`, {
    ...init,
    headers,
    cache: "no-store", // predictions are never statically cached
    signal: init.signal ?? AbortSignal.timeout(60_000),
  });
}

/** Forward status + JSON body verbatim (incl. 422 parse errors and 429s). */
export async function proxyJson(upstream: Response): Promise<Response> {
  const body = await upstream.text();
  return new Response(body, {
    status: upstream.status,
    headers: { "Content-Type": "application/json" },
  });
}
