"use client";

export type FrontendMode = "standard" | "dev";

export interface FrontendDevFeatureFlags {
  showApiKeyControls: boolean;
  showAssumptionsEntry: boolean;
  showChatTokenMetrics: boolean;
  showContextManager: boolean;
  showRunId: boolean;
}

export const FRONTEND_MODE_STORAGE_KEY = "frontend_mode";

const DEFAULT_FRONTEND_MODE: FrontendMode = "standard";
const DEV_FRONTEND_MODE: FrontendMode = "dev";

function parseFrontendMode(value: string | undefined | null): FrontendMode | null {
  if (value === DEFAULT_FRONTEND_MODE || value === DEV_FRONTEND_MODE) {
    return value;
  }
  return null;
}

function parseBooleanFlag(value: string | undefined): boolean {
  if (!value) {
    return false;
  }
  return value.trim().toLowerCase() === "true";
}

export function getDefaultFrontendMode(): FrontendMode {
  return parseFrontendMode(process.env.NEXT_PUBLIC_FRONTEND_MODE) ?? DEFAULT_FRONTEND_MODE;
}

export function isDevModeToggleEnabled(): boolean {
  return parseBooleanFlag(process.env.NEXT_PUBLIC_ENABLE_DEV_MODE_TOGGLE);
}

export function readStoredFrontendMode(): FrontendMode | null {
  if (typeof window === "undefined") {
    return null;
  }
  return parseFrontendMode(window.localStorage.getItem(FRONTEND_MODE_STORAGE_KEY));
}

export function persistFrontendMode(mode: FrontendMode): void {
  if (typeof window === "undefined") {
    return;
  }
  window.localStorage.setItem(FRONTEND_MODE_STORAGE_KEY, mode);
}

export function getDevFeatureFlags(mode: FrontendMode): FrontendDevFeatureFlags {
  const enabled = mode === DEV_FRONTEND_MODE;
  return {
    showApiKeyControls: enabled,
    showAssumptionsEntry: enabled,
    showChatTokenMetrics: enabled,
    showContextManager: enabled,
    showRunId: enabled,
  };
}
