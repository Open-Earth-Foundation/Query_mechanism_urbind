import { NextRequest, NextResponse } from "next/server";

import {
  SESSION_COOKIE_NAME,
  getSessionSecret,
  normalizeNextPath,
  verifySessionToken,
} from "@/lib/shared-session";

function isPublicRoute(pathname: string): boolean {
  return (
    pathname === "/healthz" ||
    pathname === "/login" ||
    pathname.startsWith("/api/auth/")
  );
}

function buildLoginRedirect(request: NextRequest): NextResponse {
  const loginUrl = request.nextUrl.clone();
  loginUrl.pathname = "/login";
  loginUrl.search = "";
  const nextPath = normalizeNextPath(
    `${request.nextUrl.pathname}${request.nextUrl.search}`,
  );
  if (nextPath !== "/") {
    loginUrl.searchParams.set("next", nextPath);
  }
  return NextResponse.redirect(loginUrl);
}

export default async function proxy(request: NextRequest): Promise<NextResponse> {
  const pathname = request.nextUrl.pathname;
  const sessionSecret = getSessionSecret();
  const sessionToken = request.cookies.get(SESSION_COOKIE_NAME)?.value;
  const sessionPayload =
    sessionSecret && sessionToken
      ? await verifySessionToken(sessionToken, sessionSecret)
      : null;

  if (pathname === "/login" && sessionPayload) {
    const nextPath = normalizeNextPath(request.nextUrl.searchParams.get("next"));
    const redirectPath = nextPath === "/login" ? "/" : nextPath;
    return NextResponse.redirect(new URL(redirectPath, request.url));
  }

  if (isPublicRoute(pathname)) {
    return NextResponse.next();
  }

  if (!sessionPayload) {
    return buildLoginRedirect(request);
  }

  return NextResponse.next();
}

export const config = {
  matcher: [
    "/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|jpeg|png|gif|svg|webp|ttf|woff2?|ico|csv|docx|xlsx|zip|json)).*)",
  ],
};
