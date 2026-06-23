"use client";

import { useState } from "react";
import { ChevronDown, FileText, Quote } from "lucide-react";

import type { ArtifactField } from "@/lib/api";
import { formatCityLabel } from "@/lib/utils";
import {
  confidencePillClass,
  fieldStatusStyle,
  humanizeField,
} from "@/components/pipeline/status-style";

function formatEstimate(estimate: NonNullable<ArtifactField["estimate"]>): string {
  const parts = [estimate.low, estimate.mid, estimate.high].filter(
    (value) => value !== null && value !== undefined && value !== "",
  );
  if (parts.length === 3) {
    return `${parts[0]} – ${parts[1]} – ${parts[2]}`;
  }
  return parts.map(String).join(" · ") || "—";
}

export interface FieldCardProps {
  field: ArtifactField;
  /** Stagger index for the entrance animation. */
  index?: number;
}

export function FieldCard({ field, index = 0 }: FieldCardProps) {
  const [open, setOpen] = useState(false);
  const style = fieldStatusStyle(field.status);
  const estimate = field.estimate ?? null;
  const sources = field.sources ?? [];
  const hasDetail =
    !!field.explanation ||
    !!field.recommendation ||
    !!estimate ||
    sources.length > 0;

  return (
    <div
      className={`field-card-enter group rounded-xl border border-slate-200 border-l-[3px] ${style.border} ${style.tint} p-3.5 shadow-sm transition-shadow hover:shadow-md`}
      style={{ animationDelay: `${Math.min(index, 12) * 45}ms` }}
    >
      <button
        type="button"
        onClick={() => hasDetail && setOpen((v) => !v)}
        className="flex w-full items-start justify-between gap-3 text-left"
        aria-expanded={hasDetail ? open : undefined}
      >
        <div className="min-w-0 space-y-1">
          <p className="truncate text-sm font-semibold text-slate-800">
            {humanizeField(field.field)}
          </p>
          <p className="flex flex-wrap items-center gap-x-1.5 gap-y-0.5 text-xs text-slate-500">
            <span className="font-medium text-slate-600">
              {formatCityLabel(field.city)}
            </span>
            {field.type ? (
              <>
                <span aria-hidden>·</span>
                <span>{field.type.replace(/_/g, " ")}</span>
              </>
            ) : null}
            {field.scope ? (
              <>
                <span aria-hidden>·</span>
                <span>{field.scope}</span>
              </>
            ) : null}
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <span
            className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium ${style.pill}`}
          >
            <span className={`h-1.5 w-1.5 rounded-full ${style.dot}`} />
            {style.label}
          </span>
          {hasDetail ? (
            <ChevronDown
              className={`h-4 w-4 text-slate-400 transition-transform ${open ? "rotate-180" : ""}`}
            />
          ) : null}
        </div>
      </button>

      {field.status === "non_estimable" && field.reason_label ? (
        <div className="mt-2 inline-flex items-center gap-1.5 rounded-md bg-white/70 px-2 py-1 text-[11px] font-medium text-slate-600 ring-1 ring-inset ring-slate-200">
          <span
            className={`h-1.5 w-1.5 rounded-full ${
              field.reason === "shape_mismatch" ? "bg-amber-500" : "bg-slate-400"
            }`}
          />
          {field.reason_label}
        </div>
      ) : null}

      {estimate ? (
        <div className="mt-2.5 flex flex-wrap items-center gap-2 text-xs">
          <span className="rounded-md bg-white/80 px-2 py-1 font-medium text-slate-700 ring-1 ring-inset ring-slate-200">
            {formatEstimate(estimate)}
          </span>
          {estimate.method ? (
            <span className="text-slate-500">{estimate.method.replace(/_/g, " ")}</span>
          ) : null}
          {estimate.confidence ? (
            <span
              className={`rounded-full border px-2 py-0.5 font-medium ${confidencePillClass(estimate.confidence)}`}
            >
              {estimate.confidence}
            </span>
          ) : null}
        </div>
      ) : null}

      {open && hasDetail ? (
        <div className="mt-3 space-y-3 border-t border-slate-200/70 pt-3 text-sm">
          {field.explanation ? (
            <p className="leading-relaxed text-slate-600">{field.explanation}</p>
          ) : null}
          {field.recommendation ? (
            <p className="flex gap-2 rounded-md bg-white/70 p-2.5 text-xs leading-relaxed text-slate-600 ring-1 ring-inset ring-slate-200">
              <FileText className="mt-0.5 h-3.5 w-3.5 shrink-0 text-slate-400" />
              <span>{field.recommendation}</span>
            </p>
          ) : null}
          {sources.length > 0 ? (
            <div className="space-y-1.5">
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                {sources.some((s) => s.has_evidence)
                  ? `${sources.length} source${sources.length > 1 ? "s" : ""}`
                  : `${sources.length} searched · no confirmed evidence`}
              </p>
              {sources.map((source, sourceIndex) => (
                <div
                  key={`${source.source_id ?? "src"}-${sourceIndex}`}
                  className="rounded-md bg-white/70 p-2.5 ring-1 ring-inset ring-slate-200"
                >
                  {source.title ? (
                    <p className="mb-1 text-xs font-medium text-slate-700">
                      {source.title}
                    </p>
                  ) : null}
                  {source.quote ? (
                    <p className="flex gap-1.5 text-xs leading-relaxed text-slate-500">
                      <Quote className="mt-0.5 h-3 w-3 shrink-0 text-slate-300" />
                      <span className="italic">{source.quote}</span>
                    </p>
                  ) : null}
                </div>
              ))}
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
