export const SESSION_COOKIE_NAME = "urbind_session";
export const DEFAULT_SESSION_TTL_SECONDS = 604_800;
export const DEFAULT_LOGIN_RATE_LIMIT_MAX_ATTEMPTS = 5;
export const DEFAULT_LOGIN_RATE_LIMIT_WINDOW_SECONDS = 900;
export const MIN_SESSION_SECRET_LENGTH = 32;

const SESSION_SUBJECT = "shared-gate";
const SESSION_VERSION = 1;
const LOGIN_RATE_LIMIT_GLOBAL_MAX_ATTEMPTS = 50;
const GLOBAL_RATE_LIMIT_KEY = "__global__";
const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder();

let cachedHmacSecret = "";
let cachedHmacKeyPromise: Promise<CryptoKey> | null = null;

interface SessionPayload {
  sub: string;
  iat: number;
  exp: number;
  v: number;
}

interface RateLimitEntry {
  attempts: number;
  windowEndsAt: number;
}

const failedLoginAttempts = new Map<string, RateLimitEntry>();

function readNonEmptyEnv(name: string): string | null {
  const value = process.env[name]?.trim();
  return value && value.length > 0 ? value : null;
}

function parsePositiveInteger(value: string | undefined, fallback: number): number {
  if (!value) {
    return fallback;
  }
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function encodeBase64(binary: string): string {
  if (typeof Buffer !== "undefined") {
    return Buffer.from(binary, "binary").toString("base64");
  }
  return btoa(binary);
}

function decodeBase64(value: string): string {
  if (typeof Buffer !== "undefined") {
    return Buffer.from(value, "base64").toString("binary");
  }
  return atob(value);
}

function base64UrlEncodeBytes(bytes: Uint8Array): string {
  let binary = "";
  bytes.forEach((value) => {
    binary += String.fromCharCode(value);
  });
  return encodeBase64(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
}

function base64UrlDecodeBytes(value: string): Uint8Array | null {
  const normalized = value.replace(/-/g, "+").replace(/_/g, "/");
  const padded = normalized.padEnd(Math.ceil(normalized.length / 4) * 4, "=");
  try {
    const binary = decodeBase64(padded);
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) {
      bytes[index] = binary.charCodeAt(index);
    }
    return bytes;
  } catch {
    return null;
  }
}

function constantTimeEqual(left: Uint8Array, right: Uint8Array): boolean {
  const maxLength = Math.max(left.length, right.length);
  let mismatch = left.length === right.length ? 0 : 1;
  for (let index = 0; index < maxLength; index += 1) {
    mismatch |= (left[index] ?? 0) ^ (right[index] ?? 0);
  }
  return mismatch === 0;
}

async function getHmacKey(secret: string): Promise<CryptoKey> {
  if (cachedHmacKeyPromise && cachedHmacSecret === secret) {
    return await cachedHmacKeyPromise;
  }
  cachedHmacSecret = secret;
  cachedHmacKeyPromise = crypto.subtle.importKey(
    "raw",
    textEncoder.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  return await cachedHmacKeyPromise;
}

async function signPayloadSegment(payloadSegment: string, secret: string): Promise<string> {
  const key = await getHmacKey(secret);
  const signature = await crypto.subtle.sign("HMAC", key, textEncoder.encode(payloadSegment));
  return base64UrlEncodeBytes(new Uint8Array(signature));
}

function isValidSessionPayload(value: unknown, nowSeconds: number): value is SessionPayload {
  if (!value || typeof value !== "object") {
    return false;
  }
  const payload = value as Partial<SessionPayload>;
  return (
    payload.sub === SESSION_SUBJECT &&
    payload.v === SESSION_VERSION &&
    typeof payload.iat === "number" &&
    Number.isFinite(payload.iat) &&
    typeof payload.exp === "number" &&
    Number.isFinite(payload.exp) &&
    payload.iat <= payload.exp &&
    payload.exp >= nowSeconds
  );
}

function isLocalHostname(hostname: string): boolean {
  return hostname === "localhost" || hostname === "127.0.0.1";
}

function cleanupExpiredRateLimitEntries(nowMs: number): void {
  failedLoginAttempts.forEach((entry, key) => {
    if (entry.windowEndsAt <= nowMs) {
      failedLoginAttempts.delete(key);
    }
  });
}

export function getSessionSecret(): string | null {
  const secret = readNonEmptyEnv("APP_SESSION_SECRET");
  if (!secret || secret.length < MIN_SESSION_SECRET_LENGTH) {
    return null;
  }
  return secret;
}

export function getSharedPassword(): string | null {
  return readNonEmptyEnv("APP_SHARED_PASSWORD");
}

export function getSessionCookieDomain(): string | undefined {
  return readNonEmptyEnv("APP_SESSION_COOKIE_DOMAIN") ?? undefined;
}

export function getSessionTtlSeconds(): number {
  return parsePositiveInteger(
    process.env.APP_SESSION_TTL_SECONDS,
    DEFAULT_SESSION_TTL_SECONDS,
  );
}

export function getLoginRateLimitMaxAttempts(): number {
  return parsePositiveInteger(
    process.env.APP_LOGIN_RATE_LIMIT_MAX_ATTEMPTS,
    DEFAULT_LOGIN_RATE_LIMIT_MAX_ATTEMPTS,
  );
}

export function getLoginRateLimitWindowSeconds(): number {
  return parsePositiveInteger(
    process.env.APP_LOGIN_RATE_LIMIT_WINDOW_SECONDS,
    DEFAULT_LOGIN_RATE_LIMIT_WINDOW_SECONDS,
  );
}

export function shouldUseSecureCookies(hostname: string): boolean {
  return !isLocalHostname(hostname);
}

export function normalizeNextPath(value: string | null | undefined): string {
  const cleaned = value?.trim();
  if (!cleaned || !cleaned.startsWith("/") || cleaned.startsWith("//")) {
    return "/";
  }
  return cleaned;
}

function normalizeOrigin(value: string | null): string | null {
  if (!value) {
    return null;
  }
  try {
    return new URL(value).origin;
  } catch {
    return null;
  }
}

function getFirstHeaderValue(value: string | null): string | null {
  if (!value) {
    return null;
  }
  for (const item of value.split(",")) {
    const trimmed = item.trim();
    if (trimmed) {
      return trimmed;
    }
  }
  return null;
}

function getForwardedRequestOrigin(headers: Headers): string | null {
  const proto = getFirstHeaderValue(headers.get("x-forwarded-proto"));
  const host =
    getFirstHeaderValue(headers.get("x-forwarded-host")) ??
    getFirstHeaderValue(headers.get("host"));
  return proto && host ? normalizeOrigin(`${proto}://${host}`) : null;
}

export function isTrustedSameOriginRequest(headers: Headers, requestUrl: URL): boolean {
  const expectedOrigins = new Set(
    [requestUrl.origin, getForwardedRequestOrigin(headers)].filter(
      (origin): origin is string => origin !== null,
    ),
  );
  const origin = normalizeOrigin(headers.get("origin"));
  if (origin) {
    return expectedOrigins.has(origin);
  }
  const refererOrigin = normalizeOrigin(headers.get("referer"));
  return refererOrigin !== null && expectedOrigins.has(refererOrigin);
}

export async function createSessionToken(
  secret: string,
  ttlSeconds: number,
  nowSeconds = Math.floor(Date.now() / 1000),
): Promise<string> {
  const payloadSegment = base64UrlEncodeBytes(
    textEncoder.encode(
      JSON.stringify({
        sub: SESSION_SUBJECT,
        iat: nowSeconds,
        exp: nowSeconds + ttlSeconds,
        v: SESSION_VERSION,
      }),
    ),
  );
  const signatureSegment = await signPayloadSegment(payloadSegment, secret);
  return `${payloadSegment}.${signatureSegment}`;
}

export async function verifySessionToken(
  token: string | undefined,
  secret: string,
  nowSeconds = Math.floor(Date.now() / 1000),
): Promise<SessionPayload | null> {
  if (!token) {
    return null;
  }
  const parts = token.split(".");
  if (parts.length !== 2 || !parts[0] || !parts[1]) {
    return null;
  }

  const [payloadSegment, signatureSegment] = parts;
  const expectedSignature = await signPayloadSegment(payloadSegment, secret);
  if (
    !constantTimeEqual(
      textEncoder.encode(signatureSegment),
      textEncoder.encode(expectedSignature),
    )
  ) {
    return null;
  }

  const payloadBytes = base64UrlDecodeBytes(payloadSegment);
  if (!payloadBytes) {
    return null;
  }

  try {
    const payload = JSON.parse(textDecoder.decode(payloadBytes));
    return isValidSessionPayload(payload, nowSeconds) ? payload : null;
  } catch {
    return null;
  }
}

export function buildSessionCookieOptions(hostname: string): {
  domain?: string;
  httpOnly: true;
  maxAge: number;
  path: "/";
  sameSite: "lax";
  secure: boolean;
} {
  return {
    domain: getSessionCookieDomain(),
    httpOnly: true,
    maxAge: getSessionTtlSeconds(),
    path: "/",
    sameSite: "lax",
    secure: shouldUseSecureCookies(hostname),
  };
}

export function buildClearedSessionCookieOptions(hostname: string): {
  domain?: string;
  httpOnly: true;
  maxAge: 0;
  path: "/";
  sameSite: "lax";
  secure: boolean;
} {
  return {
    ...buildSessionCookieOptions(hostname),
    maxAge: 0,
  };
}

export function getClientAddress(headers: Headers): string {
  const forwardedFor = headers.get("x-forwarded-for");
  if (forwardedFor) {
    const firstAddress = forwardedFor
      .split(",")
      .map((value) => value.trim())
      .find((value) => value.length > 0);
    if (firstAddress) {
      return firstAddress;
    }
  }
  return headers.get("x-real-ip")?.trim() || "unknown";
}

function isRateLimitKeyBlocked(key: string, maxAttempts: number, nowMs: number): boolean {
  const entry = failedLoginAttempts.get(key);
  if (!entry) {
    return false;
  }
  return entry.windowEndsAt > nowMs && entry.attempts >= maxAttempts;
}

function registerRateLimitKey(key: string, nowMs: number): void {
  const windowMs = getLoginRateLimitWindowSeconds() * 1000;
  const existing = failedLoginAttempts.get(key);
  if (!existing || existing.windowEndsAt <= nowMs) {
    failedLoginAttempts.set(key, {
      attempts: 1,
      windowEndsAt: nowMs + windowMs,
    });
    return;
  }
  failedLoginAttempts.set(key, {
    attempts: existing.attempts + 1,
    windowEndsAt: existing.windowEndsAt,
  });
}

export function isLoginRateLimited(clientAddress: string, nowMs = Date.now()): boolean {
  cleanupExpiredRateLimitEntries(nowMs);
  return (
    isRateLimitKeyBlocked(clientAddress, getLoginRateLimitMaxAttempts(), nowMs) ||
    isRateLimitKeyBlocked(
      GLOBAL_RATE_LIMIT_KEY,
      LOGIN_RATE_LIMIT_GLOBAL_MAX_ATTEMPTS,
      nowMs,
    )
  );
}

export function registerFailedLoginAttempt(clientAddress: string, nowMs = Date.now()): void {
  cleanupExpiredRateLimitEntries(nowMs);
  registerRateLimitKey(clientAddress, nowMs);
  registerRateLimitKey(GLOBAL_RATE_LIMIT_KEY, nowMs);
}

export function clearFailedLoginAttempts(clientAddress: string): void {
  failedLoginAttempts.delete(clientAddress);
}

export function constantTimeEqualStrings(left: string, right: string): boolean {
  return constantTimeEqual(textEncoder.encode(left), textEncoder.encode(right));
}
