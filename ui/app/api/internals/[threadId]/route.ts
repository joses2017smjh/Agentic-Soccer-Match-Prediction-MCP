import { gatewayFetch, proxyJson } from "@/lib/gateway";

export const runtime = "nodejs";

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ threadId: string }> },
): Promise<Response> {
  const { threadId } = await params;
  try {
    return await proxyJson(
      await gatewayFetch(`/runs/${encodeURIComponent(threadId)}/events`),
    );
  } catch {
    return Response.json(
      { error: "run not found" },
      { status: 404 },
    );
  }
}
