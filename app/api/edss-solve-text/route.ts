import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.DECISION_BACKEND_URL || "http://127.0.0.1:8008";

export async function POST(request: NextRequest) {
  const payload = await request.json();
  try {
    const response = await fetch(`${BACKEND_URL}/edss/solve-text`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      cache: "no-store"
    });
    return NextResponse.json(await response.json(), { status: response.status });
  } catch (error) {
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : "Backend unavailable" },
      { status: 503 }
    );
  }
}
