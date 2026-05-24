import { compare } from "bcryptjs";
import { NextRequest, NextResponse } from "next/server";

import {
  SESSION_COOKIE_NAME,
  buildSessionCookieOptions,
  clearFailedLoginAttempts,
  createSessionToken,
  getClientAddress,
  getSessionSecret,
  getSharedPasswordHash,
  getSessionTtlSeconds,
  isTrustedSameOriginRequest,
  isLoginRateLimited,
  normalizeNextPath,
  registerFailedLoginAttempt,
} from "@/lib/shared-session";

interface LoginRequestBody {
  next?: string;
  password?: string;
}

function parseLoginRequestBody(value: unknown): LoginRequestBody {
  if (!value || typeof value !== "object") {
    return {};
  }
  const body = value as Record<string, unknown>;
  return {
    next: typeof body.next === "string" ? body.next : undefined,
    password: typeof body.password === "string" ? body.password : undefined,
  };
}

export async function POST(request: NextRequest): Promise<NextResponse> {
  if (!isTrustedSameOriginRequest(request.headers, request.nextUrl)) {
    return NextResponse.json({ detail: "Request origin is not allowed." }, { status: 403 });
  }

  const sessionSecret = getSessionSecret();
  const sharedPasswordHash = getSharedPasswordHash();
  if (!sessionSecret || !sharedPasswordHash) {
    return NextResponse.json(
      { detail: "Application auth is not configured." },
      { status: 500 },
    );
  }

  const clientAddress = getClientAddress(request.headers);
  if (isLoginRateLimited(clientAddress)) {
    return NextResponse.json(
      { detail: "Too many login attempts. Try again later." },
      { status: 429 },
    );
  }

  let body: LoginRequestBody;
  try {
    body = parseLoginRequestBody(await request.json());
  } catch {
    return NextResponse.json({ detail: "Invalid login payload." }, { status: 400 });
  }

  const password = body.password?.trim() ?? "";
  if (!password) {
    return NextResponse.json({ detail: "Password is required." }, { status: 400 });
  }

  const isValidPassword = await compare(password, sharedPasswordHash);
  if (!isValidPassword) {
    registerFailedLoginAttempt(clientAddress);
    return NextResponse.json({ detail: "Incorrect password." }, { status: 401 });
  }

  clearFailedLoginAttempts(clientAddress);
  const token = await createSessionToken(sessionSecret, getSessionTtlSeconds());
  const nextPath = normalizeNextPath(body.next);
  const redirectTo = nextPath === "/login" ? "/" : nextPath;
  const response = NextResponse.json({
    ok: true,
    redirectTo,
  });
  response.cookies.set(
    SESSION_COOKIE_NAME,
    token,
    buildSessionCookieOptions(request.nextUrl.hostname),
  );
  return response;
}
