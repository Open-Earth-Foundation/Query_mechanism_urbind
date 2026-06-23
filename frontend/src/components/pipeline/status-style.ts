import type { ArtifactFieldStatus } from "@/lib/api";

/**
 * Shared visual tokens for field status, used by both the live build view and
 * the post-completion audit view so the two surfaces speak one language.
 *
 * Team-agreed colors: blue = estimated, red = non-estimable. Unresolved gaps
 * get amber so they never masquerade as either.
 */
export interface StatusStyle {
  label: string;
  dot: string;
  border: string;
  tint: string;
  text: string;
  pill: string;
}

export const FIELD_STATUS_STYLES: Record<ArtifactFieldStatus, StatusStyle> = {
  estimated: {
    label: "Estimated",
    dot: "bg-sky-500",
    border: "border-l-sky-500",
    tint: "bg-sky-50/70",
    text: "text-sky-700",
    pill: "border-sky-200 bg-sky-50 text-sky-700",
  },
  non_estimable: {
    label: "Non-estimable",
    dot: "bg-rose-500",
    border: "border-l-rose-500",
    tint: "bg-rose-50/60",
    text: "text-rose-700",
    pill: "border-rose-200 bg-rose-50 text-rose-700",
  },
  unresolved: {
    label: "Unresolved",
    dot: "bg-amber-500",
    border: "border-l-amber-500",
    tint: "bg-amber-50/60",
    text: "text-amber-700",
    pill: "border-amber-200 bg-amber-50 text-amber-700",
  },
};

export function fieldStatusStyle(status: ArtifactFieldStatus): StatusStyle {
  return FIELD_STATUS_STYLES[status] ?? FIELD_STATUS_STYLES.unresolved;
}

export function confidencePillClass(confidence: string | null | undefined): string {
  switch (confidence?.toUpperCase()) {
    case "HIGH":
      return "border-teal-200 bg-teal-50 text-teal-700";
    case "MEDIUM":
      return "border-amber-200 bg-amber-50 text-amber-700";
    case "LOW":
    case "VERY_LOW":
      return "border-rose-200 bg-rose-50 text-rose-700";
    default:
      return "border-slate-200 bg-slate-50 text-slate-600";
  }
}

/** Turn snake_case field/stage ids into a readable title. */
export function humanizeField(value: string): string {
  return value
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .replace(/\bPv\b/g, "PV")
    .replace(/\bMw\b/g, "MW")
    .replace(/\bPct\b/g, "%");
}
