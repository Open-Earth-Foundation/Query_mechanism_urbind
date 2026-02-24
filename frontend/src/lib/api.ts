export type RunStatus =
  | "queued"
  | "running"
  | "completed"
  | "completed_with_gaps"
  | "failed"
  | "stopped";

export interface RunError {
  code: string;
  message: string;
}

export interface CreateRunRequest {
  question: string;
  run_id?: string;
  cities?: string[];
  config_path?: string;
  markdown_path?: string;
  log_llm_payload?: boolean;
}

export interface CreateRunResponse {
  run_id: string;
  status: RunStatus;
  status_url: string;
  output_url: string;
  context_url: string;
}

export interface RunStatusResponse {
  run_id: string;
  status: RunStatus;
  started_at: string;
  completed_at?: string | null;
  finish_reason?: string | null;
  error?: RunError | null;
}

export interface RunOutputResponse {
  run_id: string;
  status: RunStatus;
  content: string;
  final_output_path: string;
}

export interface RunContextResponse {
  run_id: string;
  status: RunStatus;
  context_bundle: Record<string, unknown>;
  context_bundle_path: string;
}

export interface RunReferenceResponse {
  run_id: string;
  ref_id: string;
  excerpt_index: number;
  city_name: string;
  quote: string;
  partial_answer: string;
  source_chunk_ids: string[];
}

export interface RunSummary {
  run_id: string;
  question: string;
}

export interface RunListResponse {
  runs: RunSummary[];
  total: number;
}

export interface CityListResponse {
  cities: string[];
  total: number;
  markdown_dir: string;
}

export interface CityGroup {
  id: string;
  name: string;
  description?: string | null;
  cities: string[];
}

export interface CityGroupListResponse {
  groups: CityGroup[];
  total: number;
  groups_path: string;
}

export interface MissingDataItem {
  city: string;
  missing_description: string;
  proposed_number: number | string | null;
}

export interface AssumptionsDiscoverResponse {
  run_id: string;
  items: MissingDataItem[];
  grouped_by_city: Record<string, MissingDataItem[]>;
  verification_summary: {
    first_pass_count: number;
    second_pass_count: number;
    merged_count: number;
    added_in_verification: number;
  };
}

export interface ApplyAssumptionsRequest {
  items: MissingDataItem[];
  rewrite_instructions?: string;
  persist_artifacts?: boolean;
}

export interface ApplyAssumptionsResponse {
  run_id: string;
  revised_output_path?: string | null;
  revised_content: string;
  assumptions_path?: string | null;
}

export type ChatRole = "user" | "assistant";

export interface ChatMessage {
  role: ChatRole;
  content: string;
  created_at: string;
}

export interface ChatSessionResponse {
  run_id: string;
  conversation_id: string;
  created_at: string;
  updated_at: string;
  messages: ChatMessage[];
}

export interface ChatSessionListResponse {
  run_id: string;
  conversations: string[];
  total: number;
}

export interface ChatContextSummary {
  run_id: string;
  question: string;
  status: RunStatus;
  started_at: string;
  final_output_path: string;
  context_bundle_path: string;
  document_tokens: number;
  bundle_tokens: number;
  total_tokens: number;
}

export interface ChatContextCatalogResponse {
  contexts: ChatContextSummary[];
  total: number;
  token_cap: number;
}

export interface ChatSessionContextsResponse {
  run_id: string;
  conversation_id: string;
  context_run_ids: string[];
  contexts: ChatContextSummary[];
  total_tokens: number;
  token_cap: number;
  excluded_context_run_ids: string[];
  is_capped: boolean;
}

export interface SendChatMessageResponse {
  run_id: string;
  conversation_id: string;
  user_message: ChatMessage;
  assistant_message: ChatMessage;
}

function normalizeCityKey(value: string): string {
  return value.trim().toLocaleLowerCase();
}

function normalizeCityKeys(values?: string[]): string[] | undefined {
  if (!values || values.length === 0) {
    return undefined;
  }
  const normalized: string[] = [];
  const seen = new Set<string>();
  values.forEach((value) => {
    const key = normalizeCityKey(value);
    if (!key || seen.has(key)) {
      return;
    }
    seen.add(key);
    normalized.push(key);
  });
  return normalized.length > 0 ? normalized : undefined;
}

const configuredBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL?.trim() ?? "";
const LOCAL_API_BASE_URL = "http://127.0.0.1:8000";
const DEFAULT_API_BASE_URL = "https://urbind-query-mechanism-api.openearth.dev";

function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl.replace(/\/+$/, "");
}

function resolveClientFallbackApiBaseUrl(): string {
  const locationHost = globalThis.location?.hostname?.toLowerCase() ?? "";
  if (locationHost === "localhost" || locationHost === "127.0.0.1") {
    return LOCAL_API_BASE_URL;
  }
  if (locationHost.endsWith(".openearth.dev")) {
    return DEFAULT_API_BASE_URL;
  }
  if (locationHost.length > 0) {
    return `http://${locationHost}:8000`;
  }
  return DEFAULT_API_BASE_URL;
}

export const apiBaseUrl = normalizeBaseUrl(
  configuredBaseUrl.length > 0 ? configuredBaseUrl : DEFAULT_API_BASE_URL,
);

export function getApiBaseUrl(): string {
  if (configuredBaseUrl.length > 0) {
    return normalizeBaseUrl(configuredBaseUrl);
  }
  if (typeof window !== "undefined") {
    return normalizeBaseUrl(resolveClientFallbackApiBaseUrl());
  }
  return apiBaseUrl;
}

let userApiKey: string | null = null;

export function setUserApiKey(key: string | null): void {
  const cleaned = key?.trim() ?? "";
  userApiKey = cleaned.length > 0 ? cleaned : null;
}

function buildHeaders(includeJsonContentType: boolean): HeadersInit {
  const headers: Record<string, string> = {};
  if (includeJsonContentType) {
    headers["Content-Type"] = "application/json";
  }
  if (userApiKey) {
    headers["X-OpenRouter-Api-Key"] = userApiKey;
  }
  return headers;
}

async function requestJson<T>(
  path: string,
  init?: RequestInit,
  includeJsonContentType = false,
): Promise<T> {
  const response = await fetch(`${getApiBaseUrl()}${path}`, {
    ...init,
    headers: {
      ...buildHeaders(includeJsonContentType),
      ...(init?.headers ?? {}),
    },
  });

  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const payload = (await response.json()) as { detail?: unknown };
      if (typeof payload.detail === "string" && payload.detail.trim().length > 0) {
        message = payload.detail;
      }
    } catch {
      // ignore JSON parse errors for non-JSON responses
    }
    throw new Error(message);
  }

  return (await response.json()) as T;
}

export async function fetchRuns(): Promise<RunListResponse> {
  return requestJson<RunListResponse>("/api/v1/runs");
}

export async function startRun(payload: CreateRunRequest): Promise<CreateRunResponse> {
  const normalizedPayload: CreateRunRequest = {
    ...payload,
    cities: normalizeCityKeys(payload.cities),
  };
  return requestJson<CreateRunResponse>(
    "/api/v1/runs",
    {
      method: "POST",
      body: JSON.stringify(normalizedPayload),
    },
    true,
  );
}

export async function fetchRunStatus(runId: string): Promise<RunStatusResponse> {
  return requestJson<RunStatusResponse>(`/api/v1/runs/${encodeURIComponent(runId)}/status`);
}

export async function fetchRunOutput(runId: string): Promise<RunOutputResponse> {
  return requestJson<RunOutputResponse>(`/api/v1/runs/${encodeURIComponent(runId)}/output`);
}

export async function fetchRunContext(runId: string): Promise<RunContextResponse> {
  return requestJson<RunContextResponse>(`/api/v1/runs/${encodeURIComponent(runId)}/context`);
}

export async function fetchRunReference(
  runId: string,
  refId: string,
): Promise<RunReferenceResponse> {
  return requestJson<RunReferenceResponse>(
    `/api/v1/runs/${encodeURIComponent(runId)}/references/${encodeURIComponent(refId)}`,
  );
}

export async function fetchCities(): Promise<CityListResponse> {
  return requestJson<CityListResponse>("/api/v1/cities");
}

export async function fetchCityGroups(): Promise<CityGroupListResponse> {
  return requestJson<CityGroupListResponse>("/api/v1/city-groups");
}

export async function discoverRunAssumptions(
  runId: string,
): Promise<AssumptionsDiscoverResponse> {
  return requestJson<AssumptionsDiscoverResponse>(
    `/api/v1/runs/${encodeURIComponent(runId)}/assumptions/discover`,
    { method: "POST" },
  );
}

export async function applyRunAssumptions(
  runId: string,
  payload: ApplyAssumptionsRequest,
): Promise<ApplyAssumptionsResponse> {
  return requestJson<ApplyAssumptionsResponse>(
    `/api/v1/runs/${encodeURIComponent(runId)}/assumptions/apply`,
    {
      method: "POST",
      body: JSON.stringify(payload),
    },
    true,
  );
}

export async function fetchChatContextCatalog(): Promise<ChatContextCatalogResponse> {
  return requestJson<ChatContextCatalogResponse>("/api/v1/chat/contexts");
}

export async function listChatSessions(runId: string): Promise<ChatSessionListResponse> {
  return requestJson<ChatSessionListResponse>(
    `/api/v1/runs/${encodeURIComponent(runId)}/chat/sessions`,
  );
}

export async function createChatSession(runId: string): Promise<ChatSessionResponse> {
  return requestJson<ChatSessionResponse>(
    `/api/v1/runs/${encodeURIComponent(runId)}/chat/sessions`,
    {
      method: "POST",
      body: JSON.stringify({}),
    },
    true,
  );
}

export async function fetchChatSession(
  runId: string,
  conversationId: string,
): Promise<ChatSessionResponse> {
  return requestJson<ChatSessionResponse>(
    `/api/v1/runs/${encodeURIComponent(runId)}/chat/sessions/${encodeURIComponent(conversationId)}`,
  );
}

export async function fetchChatSessionContexts(
  runId: string,
  conversationId: string,
): Promise<ChatSessionContextsResponse> {
  return requestJson<ChatSessionContextsResponse>(
    `/api/v1/runs/${encodeURIComponent(runId)}/chat/sessions/${encodeURIComponent(conversationId)}/contexts`,
  );
}

export async function updateChatSessionContexts(
  runId: string,
  conversationId: string,
  contextRunIds: string[],
): Promise<ChatSessionContextsResponse> {
  return requestJson<ChatSessionContextsResponse>(
    `/api/v1/runs/${encodeURIComponent(runId)}/chat/sessions/${encodeURIComponent(conversationId)}/contexts`,
    {
      method: "PUT",
      body: JSON.stringify({ context_run_ids: contextRunIds }),
    },
    true,
  );
}

export async function sendChatMessage(
  runId: string,
  conversationId: string,
  content: string,
): Promise<SendChatMessageResponse> {
  return requestJson<SendChatMessageResponse>(
    `/api/v1/runs/${encodeURIComponent(runId)}/chat/sessions/${encodeURIComponent(conversationId)}/messages`,
    {
      method: "POST",
      body: JSON.stringify({ content }),
    },
    true,
  );
}

