import { gatewayFetch, proxyJson } from "@/lib/gateway";

export const runtime = "nodejs";

const ACTIONS = new Set(["approve", "reject", "edit"]);

export async function POST(request: Request): Promise<Response> {
  const body = await request.json().catch(() => null);
  if (!body || typeof body.thread_id !== "string" || !ACTIONS.has(body.action)) {
    return Response.json(
      { detail: "thread_id and action (approve|reject|edit) required" },
      { status: 400 },
    );
  }
  const upstream = await gatewayFetch("/approve", {
    method: "POST",
    body: JSON.stringify(body),
  });
  return proxyJson(upstream);
}
