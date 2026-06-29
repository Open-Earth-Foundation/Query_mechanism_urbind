"use client";

import { useMemo } from "react";
import {
  CheckCircle2,
  CircleDashed,
  Loader2,
  MinusCircle,
  Sparkles,
  XCircle,
} from "lucide-react";

import type {
  PipelineStep,
  PipelineStepItem,
  RunArtifactsResponse,
} from "@/lib/api";
import { humanizeField } from "@/components/pipeline/status-style";
import { formatCityLabel } from "@/lib/utils";

const TERMINAL_STEP_STATUSES = new Set(["completed", "skipped", "error"]);

// Visual styling only — the human label is owned by the backend (reason_label)
// and read off the field records so this chip never drifts from the cards.
const REASON_DOT: Record<string, string> = {
  found_not_validated: "bg-rose-400",
  no_source_data: "bg-slate-400",
  too_few_peers: "bg-sky-400",
};

function StepIcon({ status }: { status: string }) {
  switch (status) {
    case "completed":
      return <CheckCircle2 className="h-4 w-4 shrink-0 text-teal-600" />;
    case "running":
      return <Loader2 className="h-4 w-4 shrink-0 animate-spin text-amber-600" />;
    case "skipped":
      return <MinusCircle className="h-4 w-4 shrink-0 text-slate-400" />;
    case "error":
      return <XCircle className="h-4 w-4 shrink-0 text-rose-500" />;
    default:
      return <CircleDashed className="h-4 w-4 shrink-0 text-slate-300" />;
  }
}

function itemBadge(item: PipelineStepItem): { text: string; cls: string } | null {
  const meta = item.metadata ?? {};
  const classification = meta.classification as string | undefined;
  const status = meta.status as string | undefined;
  const priority = meta.priority as string | undefined;
  const confidence = meta.confidence as string | undefined;
  const red = "border-rose-200 bg-rose-50 text-rose-700";
  const amber = "border-amber-200 bg-amber-50 text-amber-700";
  const blue = "border-sky-200 bg-sky-50 text-sky-700";
  const neutral = "border-slate-200 bg-slate-50 text-slate-600";
  if (item.item_type === "estimate") return { text: confidence ?? "estimated", cls: blue };
  if (classification === "non_estimable" || status === "NON_ESTIMABLE")
    return { text: "non-estimable", cls: red };
  if (status === "insufficient_anchors" || status === "still_missing")
    return { text: status.replace(/_/g, " "), cls: amber };
  if (classification) return { text: classification.replace(/_/g, " "), cls: blue };
  if (priority === "high") return { text: "high", cls: red };
  if (priority === "medium") return { text: "medium", cls: amber };
  if (priority) return { text: priority, cls: neutral };
  return null;
}

function isBlockItem(item: PipelineStepItem): boolean {
  return (
    item.item_type === "field" ||
    item.item_type === "gap" ||
    item.item_type === "estimate" ||
    item.item_type === "search_result"
  );
}

/** Streaming items (web findings, classifications) as chips. */
function BlockCards({ items }: { items: PipelineStepItem[] }) {
  const blocks = items.filter(isBlockItem);
  if (blocks.length === 0) return null;
  return (
    <div className="flex flex-wrap gap-1.5">
      {blocks.map((item, index) => {
        const badge = itemBadge(item);
        const domain =
          item.item_type === "search_result"
            ? (item.metadata?.source_name as string | undefined) ?? item.domain
            : undefined;
        return (
          <span
            key={`${item.title ?? item.text}-${index}`}
            className="field-card-enter inline-flex max-w-full items-center gap-1.5 rounded-md border border-slate-200 bg-white px-2 py-1 text-xs shadow-sm"
            style={{ animationDelay: `${Math.min(index, 10) * 35}ms` }}
          >
            <span className="truncate font-medium text-slate-700">
              {item.title ?? item.text}
            </span>
            {domain ? <span className="shrink-0 text-[10px] text-slate-400">{domain}</span> : null}
            {badge ? (
              <span
                className={`shrink-0 rounded-full border px-1.5 py-0.5 text-[10px] font-medium ${badge.cls}`}
              >
                {badge.text}
              </span>
            ) : null}
          </span>
        );
      })}
    </div>
  );
}

function stepSummaryLine(step: PipelineStep): string | null {
  const plain = step.items.filter((i) => !i.item_type && i.text);
  const concise = plain.find((i) => /\d/.test(i.text) && i.text.length < 48);
  return (concise ?? plain[plain.length - 1])?.text ?? null;
}

export interface LiveBuildTimelineProps {
  steps: PipelineStep[] | null | undefined;
  runStatus?: string | null;
  /** Partial artifacts polled during the build (per-stage detail as it lands). */
  artifacts?: RunArtifactsResponse | null;
}

/**
 * Build-time view: a progress header + timeline. The current (or just-finished)
 * stage auto-reveals its detail — gap rationale, validated/unused evidence,
 * assumption outcomes — streamed from the partial artifacts; earlier stages
 * stay compact.
 */
export function LiveBuildTimeline({
  steps,
  runStatus,
  artifacts,
}: LiveBuildTimelineProps) {
  const { done, total, activeLabel } = useMemo(() => {
    const list = steps ?? [];
    return {
      done: list.filter((s) => TERMINAL_STEP_STATUSES.has(s.status)).length,
      total: list.length,
      activeLabel: list.find((s) => s.status === "running")?.label ?? null,
    };
  }, [steps]);

  // Focus the latest stage that has detail to show. A running stage hasn't
  // written its artifacts yet, so this surfaces the freshest landed finding
  // (e.g. gap rationale while external search is still running).
  const focusStepId = useMemo(() => {
    const list = steps ?? [];
    const hasDetail = (step: PipelineStep): boolean => {
      if (step.id === "gap_analysis") {
        return (artifacts?.gap_analysis?.fields.length ?? 0) > 0;
      }
      if (step.id === "external_sources") {
        const e = artifacts?.external_search;
        return (
          !!e && (e.validated.length > 0 || e.unused_total > 0 || e.no_evidence.length > 0)
        );
      }
      if (step.id === "assumptions") {
        const asm = artifacts?.enrichment_steps?.find((s) => s.key === "assumptions");
        const b = (asm?.metrics?.reason_breakdown ?? {}) as Record<string, number>;
        return Object.values(b).some((n) => n > 0);
      }
      return step.items.some(isBlockItem);
    };
    for (let i = list.length - 1; i >= 0; i -= 1) {
      if (hasDetail(list[i])) return list[i].id;
    }
    const running = list.find((s) => s.status === "running");
    if (running) return running.id;
    for (let i = list.length - 1; i >= 0; i -= 1) {
      if (TERMINAL_STEP_STATUSES.has(list[i].status)) return list[i].id;
    }
    return null;
  }, [steps, artifacts]);

  const isComplete = runStatus === "completed" || runStatus === "completed_with_gaps";
  const pct = total > 0 ? Math.round((done / total) * 100) : 0;

  function renderFocusDetail(step: PipelineStep) {
    // Gap analysis: classified fields + rationale + per-city priority.
    if (step.id === "gap_analysis" && (artifacts?.gap_analysis?.fields.length ?? 0) > 0) {
      const g = artifacts!.gap_analysis!;
      return (
        <div className="space-y-2">
          <div className="grid gap-1.5 sm:grid-cols-2">
            {g.fields.slice(0, 6).map((f) => (
              <div key={f.field} className="field-card-enter rounded-md border border-slate-200 bg-white p-2">
                <div className="flex items-center justify-between gap-2">
                  <span className="truncate text-xs font-semibold text-slate-700">
                    {humanizeField(f.field)}
                  </span>
                  {f.classification ? (
                    <span className="shrink-0 rounded-full border border-sky-200 bg-sky-50 px-1.5 py-0.5 text-[10px] font-medium text-sky-700">
                      {f.classification.replace(/_/g, " ")}
                    </span>
                  ) : null}
                </div>
                {f.rationale ? (
                  <p className="mt-1 line-clamp-2 text-[11px] leading-snug text-slate-500">
                    {f.rationale}
                  </p>
                ) : null}
              </div>
            ))}
          </div>
          {g.city_gaps.length > 0 ? (
            <div className="flex flex-wrap gap-1">
              {g.city_gaps.map((c) => {
                const dot =
                  c.priority === "high"
                    ? "bg-rose-500"
                    : c.priority === "medium"
                      ? "bg-amber-500"
                      : "bg-slate-400";
                return (
                  <span
                    key={c.city}
                    className="inline-flex items-center gap-1 rounded-full border border-slate-200 bg-white px-2 py-0.5 text-[11px] text-slate-600"
                  >
                    <span className={`h-1.5 w-1.5 rounded-full ${dot}`} />
                    {formatCityLabel(c.city)}
                  </span>
                );
              })}
            </div>
          ) : null}
        </div>
      );
    }

    // External + governed search: validated anchors + found/no-evidence counts.
    if (step.id === "external_sources" && artifacts?.external_search) {
      const e = artifacts.external_search;
      if (e.validated.length > 0 || e.unused_total > 0 || e.no_evidence.length > 0) {
        return (
          <div className="space-y-1.5">
            {e.validated.slice(0, 3).map((v, i) => (
              <div
                key={`${v.city}-${i}`}
                className="field-card-enter flex flex-wrap items-center gap-1.5 rounded-md border border-teal-200 border-l-[3px] border-l-teal-500 bg-teal-50/40 p-2 text-xs"
              >
                <span className="rounded bg-white/80 px-1.5 py-0.5 font-semibold text-slate-800 ring-1 ring-inset ring-teal-200">
                  {v.value}
                  {v.unit ? ` ${v.unit}` : ""}
                </span>
                <span className="font-medium text-slate-700">
                  {formatCityLabel(v.city)} · {humanizeField(v.field)}
                </span>
                {v.source_id ? (
                  <span className="text-[10px] text-slate-400">{v.source_id}</span>
                ) : null}
              </div>
            ))}
            <p className="text-[11px] text-slate-500">
              <span className="font-medium text-slate-600">{e.validated.length}</span>{" "}
              validated ·{" "}
              <span className="font-medium text-slate-600">{e.unused_total}</span> found,
              not validated ·{" "}
              <span className="font-medium text-slate-600">{e.no_evidence.length}</span> no
              evidence
            </p>
          </div>
        );
      }
    }

    // Assumptions: live reason rollup.
    if (step.id === "assumptions") {
      const asm = artifacts?.enrichment_steps?.find((s) => s.key === "assumptions");
      const breakdown = (asm?.metrics?.reason_breakdown ?? {}) as Record<string, number>;
      const entries = Object.entries(breakdown).filter(([, n]) => n > 0);
      if (entries.length > 0) {
        // Labels come from the backend reason_label on the field records, so
        // this rollup always matches the per-field cards and the audit view.
        const labelByCode = new Map<string, string>();
        for (const f of artifacts?.fields ?? []) {
          if (f.reason && f.reason_label && !labelByCode.has(f.reason)) {
            labelByCode.set(f.reason, f.reason_label);
          }
        }
        return (
          <div className="flex flex-wrap gap-1">
            {entries.map(([code, n]) => {
              const label = labelByCode.get(code) ?? code.replace(/_/g, " ");
              const dot = REASON_DOT[code] ?? "bg-slate-400";
              return (
                <span
                  key={code}
                  className="inline-flex items-center gap-1 rounded-full border border-slate-200 bg-white px-2 py-0.5 text-[11px] text-slate-600"
                >
                  <span className={`h-1.5 w-1.5 rounded-full ${dot}`} />
                  <span className="font-semibold text-slate-700">{n}</span>{" "}
                  {label.toLowerCase()}
                </span>
              );
            })}
          </div>
        );
      }
    }

    // Default: whatever blocks the step streamed (e.g. web-research findings).
    return <BlockCards items={step.items} />;
  }

  return (
    <div className="space-y-4 rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="space-y-2.5">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2 text-sm font-semibold text-slate-800">
            {isComplete ? (
              <Sparkles className="h-4 w-4 text-teal-600" />
            ) : (
              <Loader2 className="h-4 w-4 animate-spin text-amber-600" />
            )}
            {isComplete ? "Report built" : "Building your report…"}
          </div>
          {total > 0 ? (
            <span className="text-xs font-medium tabular-nums text-slate-500">
              {done}/{total} stages
            </span>
          ) : null}
        </div>

        <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-100">
          <div
            className="h-full rounded-full bg-gradient-to-r from-amber-400 to-teal-500 transition-[width] duration-500 ease-out"
            style={{ width: `${pct}%` }}
          />
        </div>

        {activeLabel && !isComplete ? (
          <p className="text-xs text-slate-500">
            <span className="font-medium text-amber-700">{activeLabel}</span>
            {" — leave this page open; broad questions can take a few minutes."}
          </p>
        ) : null}
      </div>

      {steps && steps.length > 0 ? (
        <div className="space-y-1">
          {steps.map((step) => {
            const summary = stepSummaryLine(step);
            const isActive = step.status === "running";
            const isFocus = step.id === focusStepId;
            return (
              <div key={step.id} className="relative pl-6">
                <div className="absolute left-[7px] top-0 h-full w-px bg-slate-200" />
                <div className="flex items-center gap-2.5 py-1.5">
                  <span className="-ml-[1.4rem] bg-white">
                    <StepIcon status={step.status} />
                  </span>
                  <span
                    className={`shrink-0 whitespace-nowrap text-sm font-medium ${
                      isActive
                        ? "text-amber-800"
                        : step.status === "skipped"
                          ? "text-slate-400"
                          : "text-slate-800"
                    }`}
                  >
                    {step.label}
                  </span>
                  {summary ? (
                    <span
                      className="hidden min-w-0 flex-1 truncate text-right text-xs text-slate-400 sm:block"
                      title={summary}
                    >
                      {summary}
                    </span>
                  ) : null}
                </div>
                {isFocus ? (
                  <div className="pb-2.5 pl-2">{renderFocusDetail(step)}</div>
                ) : null}
              </div>
            );
          })}
        </div>
      ) : (
        <div className="space-y-2">
          <div className="h-2 animate-pulse rounded bg-slate-200" />
          <div className="h-2 w-11/12 animate-pulse rounded bg-slate-200" />
          <div className="h-2 w-9/12 animate-pulse rounded bg-slate-200" />
        </div>
      )}
    </div>
  );
}
