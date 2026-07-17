import { gatewayFetch, proxyJson } from "@/lib/gateway";

export const runtime = "nodejs";

export async function POST(request: Request): Promise<Response> {
  const body = await request.json().catch(() => null);
  if (!body || typeof body.text !== "string" || !body.text.trim()) {
    return Response.json({ detail: "text is required" }, { status: 400 });
  }
  const upstream = await gatewayFetch("/predict", {
    method: "POST",
    body: JSON.stringify({ text: body.text.slice(0, 500) }),
  });
  return proxyJson(upstream);
}
