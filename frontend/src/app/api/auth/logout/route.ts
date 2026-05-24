import { NextRequest, NextResponse } from "next/server";

import {
  SESSION_COOKIE_NAME,
  buildClearedSessionCookieOptions,
  isTrustedSameOriginRequest,
} from "@/lib/shared-session";

export async function POST(request: NextRequest): Promise<NextResponse> {
  if (!isTrustedSameOriginRequest(request.headers, request.nextUrl)) {
    return NextResponse.json({ detail: "Request origin is not allowed." }, { status: 403 });
  }

  const response = NextResponse.json({ ok: true });
  response.cookies.set(
    SESSION_COOKIE_NAME,
    "",
    buildClearedSessionCookieOptions(request.nextUrl.hostname),
  );
  return response;
}
