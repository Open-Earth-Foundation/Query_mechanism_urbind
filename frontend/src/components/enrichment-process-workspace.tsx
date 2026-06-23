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

const REASON_META: Record<string, { label: string; dot: string }> = {
  shape_mismatch: {
    label: "source has related data, in a different shape",
    dot: "bg-amber-500",
  },
  found_not_validated: { label: "found, but not validated", dot: "bg-rose-400" },
  no_source_data: { label: "no source data found", dot: "bg-slate-400" },
  too_few_peers: { label: "too few comparable cities", dot: "bg-sky-400" },
};

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

  // One representative classification per distinct field name, for gap analysis.
  const distinctClassifications = useMemo(() => {
    const seen = new Map<string, { field: string; type: string | null }>();
    for (const field of data?.fields ?? []) {
      if (!seen.has(field.field)) {
        seen.set(field.field, { field: field.field, type: field.type ?? null });
      }
    }
    return [...seen.values()];
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
      return (
        <div className="flex flex-wrap gap-1.5">
          {distinctClassifications.map((entry) => (
            <span
              key={entry.field}
              className="inline-flex items-center gap-1.5 rounded-md border border-slate-200 bg-white px-2 py-1 text-xs text-slate-600"
            >
              <span className="font-medium text-slate-700">
                {humanizeField(entry.field)}
              </span>
              {entry.type ? (
                <span className="text-slate-400">{entry.type.replace(/_/g, " ")}</span>
              ) : null}
            </span>
          ))}
        </div>
      );
    }

    if (step.key === "external_web") {
      const found = Number(step.metrics.web_findings ?? 0);
      const validated = Number(step.metrics.validated_evidence ?? 0);
      return (
        <div className="space-y-2 text-sm text-slate-600">
          {step.warn ? (
            <div className="flex items-start gap-2 rounded-md border border-amber-200 bg-amber-50 p-3 text-amber-900">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
              <p>
                <span className="font-medium">{found}</span> web finding
                {found === 1 ? "" : "s"} returned, but{" "}
                <span className="font-medium">{validated}</span> validated into a usable
                anchor — so the assumptions step had nothing to estimate from. This is the
                point in the chain to investigate.
              </p>
            </div>
          ) : (
            <p>
              {found} finding{found === 1 ? "" : "s"} returned, {validated} validated into
              anchors.
            </p>
          )}
          <p className="text-xs text-slate-500">
            Per-field evidence quotes are shown inside each field card below.
          </p>
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
                const meta = REASON_META[code] ?? { label: code, dot: "bg-slate-400" };
                return (
                  <span
                    key={code}
                    className="inline-flex items-center gap-1.5 rounded-full border border-slate-200 bg-white px-2.5 py-0.5 text-xs text-slate-600"
                  >
                    <span className={`h-1.5 w-1.5 rounded-full ${meta.dot}`} />
                    <span className="font-semibold text-slate-700">{count}</span>{" "}
                    {meta.label}
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

        {data && data.enrichment_steps.length === 0 ? (
          <div className="rounded-md border border-dashed border-slate-300 bg-white p-8 text-center text-sm text-slate-600">
            No enrichment ran for this run (it may have been disabled).
          </div>
        ) : null}

        {data?.enrichment_steps.map((step) => {
          const open = openSteps[step.key] ?? false;
          return (
            <section
              key={step.key}
              className="overflow-hidden rounded-xl border border-slate-200"
            >
              <button
                type="button"
                onClick={() => toggleStep(step.key)}
                aria-expanded={open}
                className="flex w-full items-center gap-3 px-4 py-3 text-left transition hover:bg-slate-50"
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
                <div className="border-t border-slate-100 bg-slate-50/40 px-4 py-3.5">
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
