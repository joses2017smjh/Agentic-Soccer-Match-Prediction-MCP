import { gatewayFetch, proxyJson } from "@/lib/gateway";

export const runtime = "nodejs";

export async function GET(): Promise<Response> {
  try {
    return await proxyJson(await gatewayFetch("/health"));
  } catch {
    return Response.json({ ok: false, model_version: null }, { status: 503 });
  }
}
