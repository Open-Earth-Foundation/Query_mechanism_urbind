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

import type { PipelineStep, PipelineStepItem } from "@/lib/api";

const TERMINAL_STEP_STATUSES = new Set(["completed", "skipped", "error"]);

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

/** Map a streamed item to a status-colored badge, matching the audit colors. */
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

  if (item.item_type === "estimate") {
    return { text: confidence ?? "estimated", cls: blue };
  }
  if (classification === "non_estimable" || status === "NON_ESTIMABLE") {
    return { text: "non-estimable", cls: red };
  }
  if (status === "insufficient_anchors" || status === "still_missing") {
    return { text: status.replace(/_/g, " "), cls: amber };
  }
  if (classification) {
    return { text: classification.replace(/_/g, " "), cls: blue };
  }
  if (priority === "high") return { text: "high", cls: red };
  if (priority === "medium") return { text: "medium", cls: amber };
  if (priority) return { text: priority, cls: neutral };
  return null;
}

/** Items worth surfacing as cards (vs. plain status lines). */
function isBlockItem(item: PipelineStepItem): boolean {
  return (
    item.item_type === "field" ||
    item.item_type === "gap" ||
    item.item_type === "estimate" ||
    item.item_type === "search_result"
  );
}

function BlockCards({ items }: { items: PipelineStepItem[] }) {
  const blocks = items.filter(isBlockItem);
  if (blocks.length === 0) return null;
  return (
    <div className="flex flex-wrap gap-1.5">
      {blocks.map((item, index) => {
        const badge = itemBadge(item);
        return (
          <span
            key={`${item.title ?? item.text}-${index}`}
            className="field-card-enter inline-flex max-w-full items-center gap-1.5 rounded-md border border-slate-200 bg-white px-2 py-1 text-xs shadow-sm"
            style={{ animationDelay: `${Math.min(index, 10) * 35}ms` }}
          >
            <span className="truncate font-medium text-slate-700">
              {item.title ?? item.text}
            </span>
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

/** A short status line from the step's plain-text items (e.g. "14 excerpts"). */
function stepSummaryLine(step: PipelineStep): string | null {
  const plain = step.items.filter((i) => !i.item_type && i.text);
  // Prefer concise count-like lines.
  const concise = plain.find((i) => /\d/.test(i.text) && i.text.length < 48);
  return (concise ?? plain[plain.length - 1])?.text ?? null;
}

export interface LiveBuildTimelineProps {
  steps: PipelineStep[] | null | undefined;
  runStatus?: string | null;
}

/**
 * The build-time view: a progress header plus a timeline where each stage's
 * building blocks (classified fields, gaps, estimates) appear as cards and
 * accumulate as the run progresses — a live preview of the Enrichment Process.
 */
export function LiveBuildTimeline({ steps, runStatus }: LiveBuildTimelineProps) {
  const { done, total, activeLabel } = useMemo(() => {
    const list = steps ?? [];
    return {
      done: list.filter((s) => TERMINAL_STEP_STATUSES.has(s.status)).length,
      total: list.length,
      activeLabel: list.find((s) => s.status === "running")?.label ?? null,
    };
  }, [steps]);

  const isComplete = runStatus === "completed" || runStatus === "completed_with_gaps";
  const pct = total > 0 ? Math.round((done / total) * 100) : 0;

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
            const blocks = step.items.filter(isBlockItem);
            const summary = stepSummaryLine(step);
            const isActive = step.status === "running";
            return (
              <div key={step.id} className="relative pl-6">
                <div className="absolute left-[7px] top-0 h-full w-px bg-slate-200" />
                <div className="flex items-center gap-2.5 py-1.5">
                  <span className="-ml-[1.4rem] bg-white">
                    <StepIcon status={step.status} />
                  </span>
                  <span
                    className={`flex-1 text-sm font-medium ${
                      isActive
                        ? "text-amber-800"
                        : step.status === "skipped"
                          ? "text-slate-400"
                          : "text-slate-800"
                    }`}
                  >
                    {step.label}
                  </span>
                  {summary && !blocks.length ? (
                    <span className="text-xs text-slate-400">{summary}</span>
                  ) : null}
                </div>
                {blocks.length > 0 ? (
                  <div className="pb-2.5 pl-2">
                    <BlockCards items={step.items} />
                  </div>
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
