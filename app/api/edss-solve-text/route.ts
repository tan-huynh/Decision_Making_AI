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

    const raw = await response.text();
    try {
      const data = raw ? JSON.parse(raw) : {};
      return NextResponse.json(data, { status: response.status });
    } catch {
      return NextResponse.json(
        {
          detail: response.ok
            ? "Backend returned a non-JSON response."
            : `Backend returned ${response.status}: ${raw.slice(0, 300) || response.statusText}`
        },
        { status: response.ok ? 502 : response.status }
      );
    }
  } catch (error) {
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : "Backend unavailable" },
      { status: 503 }
    );
  }
}
