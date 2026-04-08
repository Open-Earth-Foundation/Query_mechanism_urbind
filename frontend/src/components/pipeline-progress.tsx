"use client";

import { useState, useRef, useEffect } from "react";
import {
  CheckCircle2,
  ChevronDown,
  CircleDashed,
  Loader2,
  MinusCircle,
  XCircle,
} from "lucide-react";

import type { PipelineStep } from "@/lib/api";

function StepIcon({ status }: { status: string }) {
  switch (status) {
    case "completed":
      return <CheckCircle2 className="h-4 w-4 shrink-0 text-teal-600" />;
    case "running":
      return <Loader2 className="h-4 w-4 shrink-0 animate-spin text-amber-600" />;
    case "skipped":
      return <MinusCircle className="h-4 w-4 shrink-0 text-slate-400" />;
    case "error":
      return <XCircle className="h-4 w-4 shrink-0 text-red-500" />;
    default:
      return <CircleDashed className="h-4 w-4 shrink-0 text-slate-300" />;
  }
}

function StepPanel({ step }: { step: PipelineStep }) {
  const [manualToggle, setManualToggle] = useState<boolean | null>(null);
  const prevStatusRef = useRef(step.status);

  // Auto-expand when a step transitions to "running"
  useEffect(() => {
    if (prevStatusRef.current !== "running" && step.status === "running") {
      setManualToggle(null);
    }
    prevStatusRef.current = step.status;
  }, [step.status]);

  const autoExpanded = step.status === "running";
  const isOpen = manualToggle ?? autoExpanded;

  return (
    <div className="relative pl-5">
      {/* Timeline connector line */}
      <div className="absolute left-[7px] top-0 h-full w-px bg-slate-200" />
      <button
        type="button"
        onClick={() => setManualToggle((prev) => !(prev ?? autoExpanded))}
        className="group flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm transition hover:bg-slate-50"
      >
        <StepIcon status={step.status} />
        <span
          className={`flex-1 font-medium ${
            step.status === "running"
              ? "text-amber-800"
              : step.status === "skipped"
                ? "text-slate-400"
                : "text-slate-700"
          }`}
        >
          {step.label}
        </span>
        <ChevronDown
          className={`h-3.5 w-3.5 text-slate-400 transition-transform ${
            isOpen ? "rotate-180" : ""
          }`}
        />
      </button>

      {isOpen && step.items.length > 0 ? (
        <div className="ml-8 space-y-0.5 pb-1 pt-0.5">
          {step.items.map((item, idx) => (
            <p key={idx} className="text-xs text-slate-500">
              {item.text}
            </p>
          ))}
        </div>
      ) : null}
    </div>
  );
}

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
      className={`space-y-0.5 ${compact ? "" : "rounded-md border border-slate-200 bg-white p-3"}`}
    >
      {!compact ? (
        <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
          Pipeline Progress
        </p>
      ) : null}
      {steps.map((step) => (
        <StepPanel key={step.id} step={step} />
      ))}
    </div>
  );
}
