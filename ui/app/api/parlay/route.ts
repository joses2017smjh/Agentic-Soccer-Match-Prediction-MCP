import { gatewayFetch, proxyJson } from "@/lib/gateway";

export const runtime = "nodejs";

export async function POST(request: Request): Promise<Response> {
  const body = await request.json().catch(() => null);
  if (!body || !Array.isArray(body.legs) || body.legs.length === 0) {
    return Response.json({ detail: "legs array is required" }, { status: 400 });
  }
  const upstream = await gatewayFetch("/parlay/price", {
    method: "POST",
    body: JSON.stringify(body),
  });
  return proxyJson(upstream);
}
