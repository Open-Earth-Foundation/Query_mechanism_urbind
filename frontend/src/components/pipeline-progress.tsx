"use client";

import { useState, useRef, useEffect, useMemo } from "react";
import {
  CheckCircle2,
  ChevronDown,
  CircleDashed,
  ExternalLink,
  Loader2,
  MinusCircle,
  Search,
  XCircle,
} from "lucide-react";

import type { PipelineStep, PipelineStepItem } from "@/lib/api";
import { Badge } from "@/components/ui/badge";

/* ------------------------------------------------------------------ */
/*  Step-level icon                                                    */
/* ------------------------------------------------------------------ */

function StepIcon({ status }: { status: string }) {
  switch (status) {
    case "completed":
      return <CheckCircle2 className="h-5 w-5 shrink-0 text-teal-600" />;
    case "running":
      return <Loader2 className="h-5 w-5 shrink-0 animate-spin text-amber-600" />;
    case "skipped":
      return <MinusCircle className="h-5 w-5 shrink-0 text-slate-400" />;
    case "error":
      return <XCircle className="h-5 w-5 shrink-0 text-red-500" />;
    default:
      return <CircleDashed className="h-5 w-5 shrink-0 text-slate-300" />;
  }
}

/* ------------------------------------------------------------------ */
/*  Badge color helpers                                                */
/* ------------------------------------------------------------------ */

function priorityColor(priority: string | undefined): string {
  switch (priority) {
    case "high":
      return "border-red-300 bg-red-50 text-red-700";
    case "medium":
      return "border-amber-300 bg-amber-50 text-amber-700";
    default:
      return "border-slate-200 bg-slate-50 text-slate-600";
  }
}

function confidenceColor(confidence: string | undefined): string {
  switch (confidence?.toUpperCase()) {
    case "HIGH":
      return "border-teal-300 bg-teal-50 text-teal-700";
    case "MEDIUM":
      return "border-amber-300 bg-amber-50 text-amber-700";
    case "LOW":
    case "VERY_LOW":
      return "border-red-300 bg-red-50 text-red-700";
    default:
      return "border-slate-200 bg-slate-50 text-slate-600";
  }
}

/* ------------------------------------------------------------------ */
/*  Row renderers (used inside grouped boxes)                          */
/* ------------------------------------------------------------------ */

function FieldRow({ item }: { item: PipelineStepItem }) {
  const classification =
    (item.metadata?.classification as string) ??
    (item.metadata?.status as string);
  return (
    <div className="flex items-center justify-between gap-3 px-4 py-2.5">
      <span className="min-w-0 truncate text-sm text-slate-800">
        {item.title ?? item.text}
      </span>
      {classification ? (
        <Badge
          variant="outline"
          className={`shrink-0 rounded-full px-2.5 py-0.5 text-xs font-normal ${
            classification === "non_estimable"
              ? "border-red-300 bg-red-50 text-red-700"
              : "border-slate-200 bg-slate-50 text-slate-600"
          }`}
        >
          {classification}
        </Badge>
      ) : null}
    </div>
  );
}

function GapRow({ item }: { item: PipelineStepItem }) {
  const priority = item.metadata?.priority as string | undefined;
  return (
    <div className="flex items-center justify-between gap-3 px-4 py-2.5">
      <span className="min-w-0 truncate text-sm text-slate-800">
        {item.title ?? item.text}
      </span>
      <div className="flex shrink-0 items-center gap-2">
        {item.count != null ? (
          <Badge
            variant="outline"
            className="rounded-full border-slate-200 bg-slate-50 px-2.5 py-0.5 text-xs font-normal text-slate-600"
          >
            {item.count} gaps
          </Badge>
        ) : null}
        {priority ? (
          <Badge
            variant="outline"
            className={`rounded-full px-2.5 py-0.5 text-xs font-normal ${priorityColor(priority)}`}
          >
            {priority}
          </Badge>
        ) : null}
      </div>
    </div>
  );
}

function SearchResultRow({ item }: { item: PipelineStepItem }) {
  return (
    <div className="flex items-center gap-3 px-4 py-2.5">
      <Search className="h-4 w-4 shrink-0 text-slate-400" />
      <span className="min-w-0 flex-1 truncate text-sm text-slate-800">
        {item.title ?? item.text}
      </span>
      {item.domain ? (
        <span className="shrink-0 text-xs text-slate-400">{item.domain}</span>
      ) : null}
      {item.url ? (
        <a
          href={item.url}
          target="_blank"
          rel="noopener noreferrer"
          className="shrink-0 text-slate-400 transition-colors hover:text-blue-500"
          onClick={(e) => e.stopPropagation()}
        >
          <ExternalLink className="h-3.5 w-3.5" />
        </a>
      ) : null}
    </div>
  );
}

function EstimateRow({ item }: { item: PipelineStepItem }) {
  const method = item.metadata?.method as string | undefined;
  const confidence = item.metadata?.confidence as string | undefined;
  const mid = item.metadata?.mid as string | undefined;
  return (
    <div className="flex items-center justify-between gap-3 px-4 py-2.5">
      <span className="min-w-0 truncate text-sm text-slate-800">
        {item.title ?? item.text}
        {mid ? (
          <span className="ml-1.5 font-medium text-slate-900">{mid}</span>
        ) : null}
      </span>
      <div className="flex shrink-0 items-center gap-2">
        {method ? (
          <Badge
            variant="outline"
            className="rounded-full border-slate-200 bg-slate-50 px-2.5 py-0.5 text-xs font-normal text-slate-600"
          >
            {method.replace(/_/g, " ")}
          </Badge>
        ) : null}
        {confidence ? (
          <Badge
            variant="outline"
            className={`rounded-full px-2.5 py-0.5 text-xs font-normal ${confidenceColor(confidence)}`}
          >
            {confidence}
          </Badge>
        ) : null}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Grouped box: renders a set of typed items inside a bordered card   */
/* ------------------------------------------------------------------ */

function GroupedBox({
  items,
  renderer,
}: {
  items: PipelineStepItem[];
  renderer: (item: PipelineStepItem) => React.ReactNode;
}) {
  if (items.length === 0) return null;
  return (
    <div className="max-h-72 overflow-y-auto rounded-lg border border-slate-200 bg-white divide-y divide-slate-100">
      {items.map((item, idx) => (
        <div key={idx}>{renderer(item)}</div>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Batch summary header (rendered above search result boxes)          */
/* ------------------------------------------------------------------ */

function BatchHeader({ item }: { item: PipelineStepItem }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <span className="min-w-0 truncate text-sm font-semibold text-slate-700">
        {item.title ?? item.text}
      </span>
      {item.count != null ? (
        <Badge className="shrink-0 rounded-full bg-amber-100 px-2.5 py-0.5 text-xs font-medium text-amber-800 border-0">
          {item.count} findings
        </Badge>
      ) : null}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Group items by type into sections                                  */
/* ------------------------------------------------------------------ */

interface ItemGroup {
  kind: "plain" | "field_box" | "gap_box" | "batch_section" | "estimate_box";
  items: PipelineStepItem[];
  header?: PipelineStepItem;
}

function groupItems(items: PipelineStepItem[]): ItemGroup[] {
  const groups: ItemGroup[] = [];
  let currentFields: PipelineStepItem[] = [];
  let currentGaps: PipelineStepItem[] = [];
  let currentEstimates: PipelineStepItem[] = [];
  let currentResults: PipelineStepItem[] = [];
  let currentBatchHeader: PipelineStepItem | undefined;

  const flushFields = () => {
    if (currentFields.length > 0) {
      groups.push({ kind: "field_box", items: [...currentFields] });
      currentFields = [];
    }
  };
  const flushGaps = () => {
    if (currentGaps.length > 0) {
      groups.push({ kind: "gap_box", items: [...currentGaps] });
      currentGaps = [];
    }
  };
  const flushEstimates = () => {
    if (currentEstimates.length > 0) {
      groups.push({ kind: "estimate_box", items: [...currentEstimates] });
      currentEstimates = [];
    }
  };
  const flushBatch = () => {
    if (currentResults.length > 0 || currentBatchHeader) {
      groups.push({
        kind: "batch_section",
        items: [...currentResults],
        header: currentBatchHeader,
      });
      currentResults = [];
      currentBatchHeader = undefined;
    }
  };

  for (const item of items) {
    switch (item.item_type) {
      case "field": {
        flushGaps();
        flushBatch();
        flushEstimates();
        currentFields.push(item);
        break;
      }
      case "gap": {
        flushFields();
        flushBatch();
        flushEstimates();
        currentGaps.push(item);
        break;
      }
      case "estimate": {
        flushFields();
        flushGaps();
        flushBatch();
        currentEstimates.push(item);
        break;
      }
      case "batch_summary": {
        // Flush any previous batch section
        flushBatch();
        flushFields();
        flushGaps();
        flushEstimates();
        currentBatchHeader = item;
        break;
      }
      case "search_result": {
        flushFields();
        flushGaps();
        flushEstimates();
        currentResults.push(item);
        break;
      }
      default: {
        // Plain text — flush all typed groups first
        flushFields();
        flushGaps();
        flushBatch();
        flushEstimates();
        groups.push({ kind: "plain", items: [item] });
        break;
      }
    }
  }

  // Final flush
  flushFields();
  flushGaps();
  flushBatch();
  flushEstimates();

  return groups;
}

/* ------------------------------------------------------------------ */
/*  Step panel                                                         */
/* ------------------------------------------------------------------ */

function StepPanel({ step }: { step: PipelineStep }) {
  const [manualToggle, setManualToggle] = useState<boolean | null>(null);
  const prevStatusRef = useRef(step.status);

  useEffect(() => {
    if (prevStatusRef.current !== "running" && step.status === "running") {
      setManualToggle(null);
    }
    prevStatusRef.current = step.status;
  }, [step.status]);

  const autoExpanded = step.status === "running";
  const isOpen = manualToggle ?? autoExpanded;

  const itemGroups = useMemo(() => groupItems(step.items), [step.items]);

  return (
    <div className="relative pl-6">
      {/* Timeline connector line */}
      <div className="absolute left-[9px] top-0 h-full w-px bg-slate-200" />
      <button
        type="button"
        onClick={() => setManualToggle((prev) => !(prev ?? autoExpanded))}
        className="group flex w-full items-center gap-2.5 rounded-md px-2 py-2 text-left transition hover:bg-slate-50"
      >
        <StepIcon status={step.status} />
        <span
          className={`flex-1 text-base font-semibold ${
            step.status === "running"
              ? "text-amber-800"
              : step.status === "skipped"
                ? "text-slate-400"
                : "text-slate-800"
          }`}
        >
          {step.label}
        </span>
        <ChevronDown
          className={`h-4 w-4 text-slate-400 transition-transform ${
            isOpen ? "rotate-180" : ""
          }`}
        />
      </button>

      {isOpen && step.items.length > 0 ? (
        <div className="ml-9 space-y-3 pb-3 pt-1">
          {itemGroups.map((group, gIdx) => {
            switch (group.kind) {
              case "field_box":
                return (
                  <GroupedBox
                    key={gIdx}
                    items={group.items}
                    renderer={(item) => <FieldRow item={item} />}
                  />
                );
              case "gap_box":
                return (
                  <GroupedBox
                    key={gIdx}
                    items={group.items}
                    renderer={(item) => <GapRow item={item} />}
                  />
                );
              case "estimate_box":
                return (
                  <GroupedBox
                    key={gIdx}
                    items={group.items}
                    renderer={(item) => <EstimateRow item={item} />}
                  />
                );
              case "batch_section":
                return (
                  <div key={gIdx} className="space-y-2">
                    {group.header ? <BatchHeader item={group.header} /> : null}
                    {group.items.length > 0 ? (
                      <GroupedBox
                        items={group.items}
                        renderer={(item) => <SearchResultRow item={item} />}
                      />
                    ) : null}
                  </div>
                );
              case "plain":
              default:
                return group.items.map((item, iIdx) => (
                  <p
                    key={`${gIdx}-${iIdx}`}
                    className="text-sm text-slate-500"
                  >
                    {item.text}
                  </p>
                ));
            }
          })}
        </div>
      ) : null}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Main export                                                        */
/* ------------------------------------------------------------------ */

export interface PipelineProgressProps {
  steps: PipelineStep[] | null | undefined;
  compact?: boolean;
}

export function PipelineProgress({ steps, compact }: PipelineProgressProps) {
  if (!steps || steps.length === 0) {
    return null;
  }

  return (
    <div
      className={`space-y-1 ${compact ? "" : "rounded-lg border border-slate-200 bg-white p-4"}`}
    >
      {!compact ? (
        <p className="mb-3 text-xs font-semibold uppercase tracking-wide text-slate-500">
          Pipeline Progress
        </p>
      ) : null}
      {steps.map((step) => (
        <StepPanel key={step.id} step={step} />
      ))}
    </div>
  );
}
