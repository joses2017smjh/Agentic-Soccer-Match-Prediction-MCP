import { gatewayFetch, proxyJson } from "@/lib/gateway";

export const runtime = "nodejs";

export async function POST(request: Request): Promise<Response> {
  const body = await request.json().catch(() => null);
  if (!body || typeof body.message !== "string" || !body.message.trim()) {
    return Response.json({ detail: "message is required" }, { status: 400 });
  }
  const upstream = await gatewayFetch("/chat", {
    method: "POST",
    body: JSON.stringify({
      message: body.message.slice(0, 500),
      history: Array.isArray(body.history) ? body.history.slice(-20) : [],
    }),
  });
  return proxyJson(upstream);
}
