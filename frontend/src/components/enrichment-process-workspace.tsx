"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  ArrowLeft,
  ChevronDown,
  Loader2,
  RefreshCw,
  ScanSearch,
} from "lucide-react";

import {
  ArtifactField,
  ArtifactFieldStatus,
  EnrichmentStep,
  RunArtifactsResponse,
  fetchRunArtifacts,
} from "@/lib/api";
import { FieldCard } from "@/components/pipeline/field-card";
import { fieldStatusStyle, humanizeField } from "@/components/pipeline/status-style";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { formatCityLabel } from "@/lib/utils";

const UNUSED_PREVIEW_COUNT = 6;

interface EnrichmentProcessWorkspaceProps {
  runId: string;
  onClose?: () => void;
}

const STATUS_ORDER: ArtifactFieldStatus[] = ["estimated", "unresolved", "non_estimable"];
const STEP_NUMBER: Record<string, number> = {
  gap_analysis: 1,
  external_web: 2,
  assumptions: 3,
};

// Visual styling only — the human label is owned by the backend (reason_label)
// and read off the field records so the rollup never diverges from the cards.
const REASON_DOT: Record<string, string> = {
  found_not_validated: "bg-rose-400",
  no_source_data: "bg-slate-400",
  too_few_peers: "bg-sky-400",
};

function humanizeReasonCode(code: string): string {
  return code.replace(/_/g, " ");
}

function WarnBadge({ text }: { text: string }) {
  return (
    <span
      className="inline-flex items-center gap-1 rounded-full border border-amber-200 bg-amber-50 px-2 py-0.5 text-[11px] font-medium text-amber-700"
      title={text}
    >
      <AlertTriangle className="h-3 w-3" />
      Break
    </span>
  );
}

export function EnrichmentProcessWorkspace({
  runId,
  onClose,
}: EnrichmentProcessWorkspaceProps) {
  const [data, setData] = useState<RunArtifactsResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [openSteps, setOpenSteps] = useState<Record<string, boolean>>({});
  const [showAllUnused, setShowAllUnused] = useState(false);

  const load = useCallback(
    async (signal?: AbortSignal) => {
      setIsLoading(true);
      setError(null);
      try {
        const payload = await fetchRunArtifacts(runId, { signal });
        if (!signal?.aborted) {
          setData(payload);
        }
      } catch (err) {
        if (signal?.aborted) return;
        setError(err instanceof Error ? err.message : "Failed to load enrichment process.");
      } finally {
        if (!signal?.aborted) setIsLoading(false);
      }
    },
    [runId],
  );

  useEffect(() => {
    const controller = new AbortController();
    void load(controller.signal);
    return () => controller.abort();
  }, [load]);

  // Default: open the Assumptions detail and any step that broke.
  useEffect(() => {
    if (!data) return;
    const next: Record<string, boolean> = {};
    for (const step of data.enrichment_steps) {
      next[step.key] = step.key === "assumptions" || !!step.warn;
    }
    setOpenSteps(next);
  }, [data]);

  const grouped = useMemo(() => {
    const groups = new Map<ArtifactFieldStatus, ArtifactField[]>();
    for (const field of data?.fields ?? []) {
      const list = groups.get(field.status) ?? [];
      list.push(field);
      groups.set(field.status, list);
    }
    return groups;
  }, [data]);

  // Map reason code -> the backend's label, read off the field records so the
  // rollup chips always match the individual cards (single source of truth).
  const reasonLabelByCode = useMemo(() => {
    const map = new Map<string, string>();
    for (const field of data?.fields ?? []) {
      if (field.reason && field.reason_label && !map.has(field.reason)) {
        map.set(field.reason, field.reason_label);
      }
    }
    return map;
  }, [data]);

  const statusCounts = useMemo(
    () =>
      STATUS_ORDER.map((status) => ({
        status,
        count: grouped.get(status)?.length ?? 0,
      })).filter((entry) => entry.count > 0),
    [grouped],
  );

  function toggleStep(key: string): void {
    setOpenSteps((current) => ({ ...current, [key]: !current[key] }));
  }

  function renderStepBody(step: EnrichmentStep) {
    if (step.key === "gap_analysis") {
      // Sourced from the enrichment field_manifest (the full classified set),
      // not reconstructed from the assumptions-derived `fields` — so a field
      // dropped before the assumptions stage still appears here.
      const gapFields = data?.gap_analysis?.fields ?? [];
      const cityGaps = data?.gap_analysis?.city_gaps ?? [];
      if (gapFields.length === 0) {
        return (
          <p className="text-xs text-slate-500">
            Field-classification detail isn’t available for this run.
          </p>
        );
      }
      return (
        <div className="space-y-3">
          <div className="grid gap-2 md:grid-cols-2">
            {gapFields.map((f) => (
              <div
                key={f.field}
                className="rounded-lg border border-slate-200 bg-white p-3"
              >
                <div className="flex items-start justify-between gap-2">
                  <p className="text-sm font-semibold text-slate-800">
                    {humanizeField(f.field)}
                  </p>
                  {f.classification ? (
                    <span className="shrink-0 rounded-full border border-slate-200 bg-slate-50 px-2 py-0.5 text-[10px] font-medium text-slate-600">
                      {f.classification.replace(/_/g, " ")}
                    </span>
                  ) : null}
                </div>
                {f.scope ? (
                  <p className="mt-0.5 text-xs text-slate-400">scope: {f.scope}</p>
                ) : null}
                {f.rationale ? (
                  <p className="mt-1.5 text-xs leading-relaxed text-slate-600">
                    {f.rationale}
                  </p>
                ) : null}
              </div>
            ))}
          </div>
          {cityGaps.length > 0 ? (
            <div className="space-y-1.5">
              <p className="text-xs font-medium text-slate-500">
                Per-city gap priority
              </p>
              <div className="flex flex-wrap gap-1.5">
                {cityGaps.map((g) => {
                  const dot =
                    g.priority === "high"
                      ? "bg-rose-500"
                      : g.priority === "medium"
                        ? "bg-amber-500"
                        : "bg-slate-400";
                  return (
                    <span
                      key={g.city}
                      className="inline-flex items-center gap-1.5 rounded-full border border-slate-200 bg-white px-2.5 py-0.5 text-xs text-slate-600"
                    >
                      <span className={`h-1.5 w-1.5 rounded-full ${dot}`} />
                      {formatCityLabel(g.city)}
                      {g.priority ? (
                        <span className="text-slate-400">· {g.priority}</span>
                      ) : null}
                    </span>
                  );
                })}
              </div>
            </div>
          ) : null}
        </div>
      );
    }

    if (step.key === "external_web") {
      const detail = data?.external_search;
      const validated = detail?.validated ?? [];
      const unused = detail?.unused ?? [];
      const noEvidence = detail?.no_evidence ?? [];
      const unusedTotal = detail?.unused_total ?? unused.length;
      const shownUnused = showAllUnused ? unused : unused.slice(0, UNUSED_PREVIEW_COUNT);
      return (
        <div className="space-y-4 text-sm text-slate-600">
          {step.warn ? (
            <div className="flex items-start gap-2 rounded-md border border-amber-200 bg-amber-50 p-3 text-amber-900">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
              {/* Render the backend's scoped warning verbatim. It describes the
                  external/web step only — the downstream assumptions step can
                  still estimate from peer/national data, so we don't claim the
                  whole chain failed here. */}
              <p>{step.warn}</p>
            </div>
          ) : null}

          {validated.length > 0 ? (
            <div className="space-y-2">
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                Validated into anchors ({validated.length})
              </p>
              {validated.map((v, i) => (
                <div
                  key={`${v.city}-${v.field}-${i}`}
                  className="rounded-lg border border-teal-200 border-l-[3px] border-l-teal-500 bg-teal-50/40 p-3"
                >
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="rounded-md bg-white/80 px-2 py-1 text-xs font-semibold text-slate-800 ring-1 ring-inset ring-teal-200">
                      {v.value}
                      {v.unit ? ` ${v.unit}` : ""}
                    </span>
                    <span className="text-xs font-medium text-slate-700">
                      {formatCityLabel(v.city)} · {humanizeField(v.field)}
                    </span>
                    {v.source_id ? (
                      <span className="text-xs text-slate-400">
                        {v.source_id}
                        {v.publication_year ? ` (${v.publication_year})` : ""}
                      </span>
                    ) : null}
                  </div>
                  {v.quote ? (
                    <p className="mt-1.5 text-xs italic leading-relaxed text-slate-500">
                      “{v.quote}”
                    </p>
                  ) : null}
                </div>
              ))}
            </div>
          ) : null}

          {unused.length > 0 ? (
            <div className="space-y-2">
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                Found, not validated ({unusedTotal})
              </p>
              {shownUnused.map((u, i) => (
                <div
                  key={`${u.city}-${u.source_id ?? "src"}-${i}`}
                  className="rounded-lg border border-slate-200 bg-white p-3"
                >
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-xs font-medium text-slate-700">
                      {formatCityLabel(u.city)} · {humanizeField(u.field)}
                    </span>
                    {u.title ? (
                      <span className="text-xs text-slate-400">{u.title}</span>
                    ) : null}
                  </div>
                  {u.quote ? (
                    <p className="mt-1.5 line-clamp-2 text-xs italic leading-relaxed text-slate-500">
                      “{u.quote}”
                    </p>
                  ) : null}
                </div>
              ))}
              {unused.length > UNUSED_PREVIEW_COUNT ? (
                <button
                  type="button"
                  onClick={() => setShowAllUnused((v) => !v)}
                  className="text-xs font-medium text-slate-500 underline-offset-2 hover:text-slate-700 hover:underline"
                >
                  {showAllUnused
                    ? "Show fewer"
                    : `Show ${unused.length - UNUSED_PREVIEW_COUNT} more`}
                </button>
              ) : null}
            </div>
          ) : null}

          {noEvidence.length > 0 ? (
            <p className="text-xs text-slate-500">
              <span className="font-medium text-slate-600">{noEvidence.length}</span>{" "}
              field{noEvidence.length === 1 ? "" : "s"} searched with no evidence found.
            </p>
          ) : null}

          {validated.length === 0 && unused.length === 0 && noEvidence.length === 0 ? (
            <p className="text-xs text-slate-500">
              No external or web search ran for this run.
            </p>
          ) : null}
        </div>
      );
    }

    // assumptions
    const breakdown = (step.metrics.reason_breakdown ?? {}) as Record<string, number>;
    const breakdownEntries = Object.entries(breakdown)
      .filter(([, count]) => count > 0)
      .sort((a, b) => b[1] - a[1]);
    return (
      <div className="space-y-4">
        {breakdownEntries.length > 0 ? (
          <div className="space-y-1.5">
            <p className="text-xs font-medium text-slate-500">Why fields broke</p>
            <div className="flex flex-wrap gap-1.5">
              {breakdownEntries.map(([code, count]) => {
                const label =
                  reasonLabelByCode.get(code) ?? humanizeReasonCode(code);
                const dot = REASON_DOT[code] ?? "bg-slate-400";
                return (
                  <span
                    key={code}
                    className="inline-flex items-center gap-1.5 rounded-full border border-slate-200 bg-white px-2.5 py-0.5 text-xs text-slate-600"
                  >
                    <span className={`h-1.5 w-1.5 rounded-full ${dot}`} />
                    <span className="font-semibold text-slate-700">{count}</span>{" "}
                    {label.toLowerCase()}
                  </span>
                );
              })}
            </div>
          </div>
        ) : null}
        {STATUS_ORDER.map((status) => {
          const fields = grouped.get(status);
          if (!fields || fields.length === 0) return null;
          const style = fieldStatusStyle(status);
          return (
            <div key={status} className="space-y-2.5">
              <div className="flex items-center gap-2">
                <span className={`h-2 w-2 rounded-full ${style.dot}`} />
                <h4 className="text-sm font-semibold text-slate-700">{style.label}</h4>
                <span className="text-xs text-slate-400">{fields.length}</span>
              </div>
              <div className="grid gap-2.5 md:grid-cols-2">
                {fields.map((field, index) => (
                  <FieldCard
                    key={`${field.city}-${field.field}`}
                    field={field}
                    index={index}
                  />
                ))}
              </div>
            </div>
          );
        })}
      </div>
    );
  }

  const totalFields = data?.fields.length ?? 0;

  return (
    <Card className="border-slate-300">
      <CardHeader className="pb-4">
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <ScanSearch className="h-4 w-4 text-teal-600" />
              Enrichment Process
            </CardTitle>
            <CardDescription>
              Audit how each data gap was classified, searched, and estimated — and where
              the chain broke down.
            </CardDescription>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {onClose ? (
              <Button type="button" size="sm" variant="outline" onClick={onClose}>
                <ArrowLeft className="h-4 w-4" />
                Open Full Document
              </Button>
            ) : null}
            <Button
              type="button"
              size="sm"
              variant="ghost"
              onClick={() => void load()}
              disabled={isLoading}
            >
              {isLoading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <RefreshCw className="h-4 w-4" />
              )}
              Refresh
            </Button>
          </div>
        </div>

        {data ? (
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <span className="text-xs font-medium text-slate-500">
              {totalFields} field{totalFields === 1 ? "" : "s"}
            </span>
            {statusCounts.map(({ status, count }) => {
              const style = fieldStatusStyle(status);
              return (
                <span
                  key={status}
                  className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium ${style.pill}`}
                >
                  <span className={`h-1.5 w-1.5 rounded-full ${style.dot}`} />
                  {count} {style.label.toLowerCase()}
                </span>
              );
            })}
          </div>
        ) : null}
      </CardHeader>

      <CardContent className="space-y-3">
        {error ? <p className="text-sm text-rose-600">{error}</p> : null}

        {!data && isLoading ? (
          <div className="flex items-center gap-2 rounded-md border border-slate-200 bg-white p-6 text-sm text-slate-600">
            <Loader2 className="h-4 w-4 animate-spin" />
            Loading enrichment process…
          </div>
        ) : null}

        {data?.degraded ? (
          <div className="flex items-start gap-2 rounded-md border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
            <p>
              Some run artifacts exist but couldn’t be read, so this view may be
              incomplete. This is an artifact-read problem, not necessarily a
              pipeline failure.
            </p>
          </div>
        ) : null}

        {data && data.enrichment_steps.length === 0 ? (
          <div className="rounded-md border border-dashed border-slate-300 bg-white p-8 text-center text-sm text-slate-600">
            {data.degraded
              ? "Enrichment artifacts couldn’t be read for this run."
              : "No enrichment ran for this run (it may have been disabled)."}
          </div>
        ) : null}

        {data?.enrichment_steps.map((step) => {
          const open = openSteps[step.key] ?? false;
          return (
            <section
              key={step.key}
              // No overflow-hidden: a field card's hover popover needs to escape
              // this box. Corners are rounded on the header/body instead.
              className="rounded-xl border border-slate-200"
            >
              <button
                type="button"
                onClick={() => toggleStep(step.key)}
                aria-expanded={open}
                className={`flex w-full items-center gap-3 px-4 py-3 text-left transition hover:bg-slate-50 rounded-t-xl ${open ? "" : "rounded-b-xl"}`}
              >
                <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-slate-100 text-xs font-semibold text-slate-600">
                  {STEP_NUMBER[step.key] ?? "•"}
                </span>
                <span className="flex-1">
                  <span className="text-sm font-semibold text-slate-800">{step.label}</span>
                  <span className="ml-2 text-xs text-slate-500">{step.summary}</span>
                </span>
                {step.warn ? <WarnBadge text={step.warn} /> : null}
                <ChevronDown
                  className={`h-4 w-4 shrink-0 text-slate-400 transition-transform ${open ? "rotate-180" : ""}`}
                />
              </button>
              {open ? (
                <div className="border-t border-slate-100 bg-slate-50/40 px-4 py-3.5 rounded-b-xl">
                  {renderStepBody(step)}
                </div>
              ) : null}
            </section>
          );
        })}
      </CardContent>
    </Card>
  );
}
