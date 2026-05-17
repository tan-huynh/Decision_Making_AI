import { NextResponse } from "next/server";

const BACKEND_URL = process.env.DECISION_BACKEND_URL || "http://127.0.0.1:8008";

export async function GET() {
  try {
    const response = await fetch(`${BACKEND_URL}/ollama-status`, { cache: "no-store" });
    return NextResponse.json(await response.json(), { status: response.status });
  } catch {
    return NextResponse.json({ ok: false, models: [], error: "Backend unavailable" }, { status: 503 });
  }
}
