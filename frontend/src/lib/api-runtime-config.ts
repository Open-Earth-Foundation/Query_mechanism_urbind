const configuredBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL?.trim() ?? "";
const localApiPort = process.env.NEXT_PUBLIC_LOCAL_API_PORT?.trim() || "8000";

function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl.replace(/\/+$/, "");
}

function resolveLocalFallbackApiBaseUrl(): string {
  if (typeof window === "undefined") {
    return `http://127.0.0.1:${localApiPort}`;
  }
  const protocol = window.location.protocol === "https:" ? "https:" : "http:";
  const hostname = window.location.hostname || "127.0.0.1";
  if (hostname === "localhost" || hostname === "127.0.0.1") {
    return `http://127.0.0.1:${localApiPort}`;
  }
  return `${protocol}//${hostname}:${localApiPort}`;
}

export function getApiBaseUrl(): string {
  const baseUrl =
    configuredBaseUrl.length > 0 ? configuredBaseUrl : resolveLocalFallbackApiBaseUrl();
  return normalizeBaseUrl(baseUrl);
}
