import { gatewayFetch, proxyJson } from "@/lib/gateway";

export const runtime = "nodejs";

export async function GET(): Promise<Response> {
  try {
    return await proxyJson(await gatewayFetch("/runs?limit=50"));
  } catch {
    return Response.json({ runs: [] }, { status: 503 });
  }
}
