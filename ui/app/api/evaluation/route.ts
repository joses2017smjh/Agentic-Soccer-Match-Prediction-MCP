import { gatewayFetch, proxyJson } from "@/lib/gateway";

export const runtime = "nodejs";

export async function POST(request: Request): Promise<Response> {
  const body = await request.json().catch(() => null);
  if (!body) {
    return Response.json({ detail: "request body required" }, { status: 400 });
  }

  const action = body.action as string | undefined;

  if (action === "evaluate" && body.match) {
    const upstream = await gatewayFetch("/evaluation/evaluate", {
      method: "POST",
      body: JSON.stringify(body.match),
    });
    return proxyJson(upstream);
  }

  if (action === "attribution" && Array.isArray(body.matches)) {
    const upstream = await gatewayFetch("/evaluation/attribution", {
      method: "POST",
      body: JSON.stringify({ matches: body.matches }),
    });
    return proxyJson(upstream);
  }

  if (action === "trajectory") {
    const upstream = await gatewayFetch("/evaluation/trajectory", {
      method: "POST",
    });
    return proxyJson(upstream);
  }

  return Response.json(
    { detail: "action must be 'evaluate', 'attribution', or 'trajectory'" },
    { status: 400 },
  );
}
