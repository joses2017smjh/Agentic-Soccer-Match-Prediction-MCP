/**
 * Server-only gateway client used by Route Handlers.
 *
 * GATEWAY_URL and GATEWAY_API_KEY live in server env (.env.local / Compose)
 * and are attached here — the browser only ever talks to /api/* on this app,
 * so the FastAPI origin and credentials are never exposed client-side.
 */
import "server-only";

const GATEWAY_URL = process.env.GATEWAY_URL ?? "http://localhost:8000";

export async function gatewayFetch(
  path: string,
  init: RequestInit = {},
): Promise<Response> {
  const headers = new Headers(init.headers);
  headers.set("Content-Type", "application/json");
  const key = process.env.GATEWAY_API_KEY;
  if (key) headers.set("X-API-Key", key);

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
