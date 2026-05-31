"use client";

import { useEffect, useState } from "react";
import { AlertTriangle, Loader2, RefreshCw } from "lucide-react";

import { Button } from "@/components/ui/button";
import { fetchRunDiagnostics, RunDiagnosticsResponse, RunStatusResponse } from "@/lib/api";

interface RunDiagnosticsPanelProps {
  runId: string;
  runStatus: RunStatusResponse | null;
}

function formatTimestamp(value: string | null | undefined): string {
  if (!value) {
    return "n/a";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toLocaleString();
}

function formatJsonBlock(value: Record<string, unknown> | null | undefined): string {
  if (!value) {
    return "{}";
  }
  return JSON.stringify(value, null, 2);
}

export function RunDiagnosticsPanel({
  runId,
  runStatus,
}: RunDiagnosticsPanelProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [diagnostics, setDiagnostics] = useState<RunDiagnosticsResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    setIsExpanded(false);
    setIsLoading(false);
    setDiagnostics(null);
    setErrorMessage(null);
  }, [runId]);

  async function loadDiagnostics(): Promise<void> {
    setIsLoading(true);
    setErrorMessage(null);
    try {
      setDiagnostics(await fetchRunDiagnostics(runId));
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : "Failed to load run diagnostics.",
      );
    } finally {
      setIsLoading(false);
    }
  }

  async function handleToggleExpanded(): Promise<void> {
    const nextExpanded = !isExpanded;
    setIsExpanded(nextExpanded);
    if (nextExpanded && !diagnostics && !isLoading) {
      await loadDiagnostics();
    }
  }

  const statusText = diagnostics?.status ?? runStatus?.status ?? "unknown";
  const finishReason = diagnostics?.finish_reason ?? runStatus?.finish_reason ?? "n/a";
  const warningEntries = diagnostics?.warning_entries ?? [];
  const logTail = diagnostics?.log_tail ?? [];
  const errorLogText = diagnostics?.error_log_text ?? null;
  const writerCoverage = diagnostics?.writer_citation_coverage ?? null;
  const writerMultiPass = diagnostics?.writer_multi_pass ?? null;
  const writerSavedEvidence = diagnostics?.writer_saved_evidence ?? null;
  const missingCities = writerCoverage?.missing_cities ?? [];

  return (
    <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-sm font-medium text-slate-900">Developer diagnostics</p>
          <p className="text-xs text-slate-600">
            Dev-only warnings, retries, and failure excerpts from persisted run artifacts.
          </p>
        </div>
        <div className="flex gap-2">
          {isExpanded ? (
            <Button
              type="button"
              size="sm"
              variant="outline"
              onClick={() => void loadDiagnostics()}
              disabled={isLoading}
            >
              {isLoading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />}
              Refresh
            </Button>
          ) : null}
          <Button
            type="button"
            size="sm"
            variant="outline"
            onClick={() => void handleToggleExpanded()}
          >
            {isExpanded ? "Hide" : "Inspect"}
          </Button>
        </div>
      </div>

      {isExpanded ? (
        <div className="mt-3 space-y-3">
          <div className="grid gap-2 text-xs text-slate-700 md:grid-cols-2">
            <div className="rounded-md border border-slate-200 bg-white px-3 py-2">
              <span className="font-medium text-slate-900">Status:</span> {statusText}
            </div>
            <div className="rounded-md border border-slate-200 bg-white px-3 py-2">
              <span className="font-medium text-slate-900">Finish reason:</span> {finishReason}
            </div>
            <div className="rounded-md border border-slate-200 bg-white px-3 py-2">
              <span className="font-medium text-slate-900">Started:</span>{" "}
              {formatTimestamp(diagnostics?.started_at ?? runStatus?.started_at)}
            </div>
            <div className="rounded-md border border-slate-200 bg-white px-3 py-2">
              <span className="font-medium text-slate-900">Completed:</span>{" "}
              {formatTimestamp(diagnostics?.completed_at ?? runStatus?.completed_at)}
            </div>
          </div>

          {errorMessage ? (
            <div className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
              {errorMessage}
            </div>
          ) : null}

          {diagnostics?.error ? (
            <div className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900">
              <div className="mb-1 flex items-center gap-2 font-medium">
                <AlertTriangle className="h-3.5 w-3.5" />
                Failure payload
              </div>
              <p>
                {diagnostics.error.code}: {diagnostics.error.message}
              </p>
            </div>
          ) : null}

          {writerCoverage ? (
            <details
              className="rounded-md border border-slate-200 bg-white"
              open={writerCoverage.status !== "confirmed"}
            >
              <summary className="cursor-pointer px-3 py-2 text-sm font-medium text-slate-900">
                Writer citation coverage ({writerCoverage.coverage_ratio})
              </summary>
              <div className="space-y-2 border-t border-slate-200 px-3 py-2 text-xs text-slate-700">
                <p>
                  <span className="font-medium text-slate-900">Status:</span> {writerCoverage.status}
                </p>
                <p>
                  <span className="font-medium text-slate-900">Coverage:</span>{" "}
                  {writerCoverage.coverage_confirmed} of {writerCoverage.coverage_required} cities
                </p>
                {writerCoverage.attempt != null && writerCoverage.max_attempts != null ? (
                  <p>
                    <span className="font-medium text-slate-900">Attempts:</span>{" "}
                    {writerCoverage.attempt}/{writerCoverage.max_attempts}
                  </p>
                ) : null}
                {writerCoverage.analysis_mode ? (
                  <p>
                    <span className="font-medium text-slate-900">Analysis mode:</span>{" "}
                    {writerCoverage.analysis_mode}
                  </p>
                ) : null}
                {missingCities.length > 0 ? (
                  <div>
                    <p className="font-medium text-slate-900">
                      Missing cities ({missingCities.length})
                    </p>
                    <pre className="max-h-48 overflow-auto whitespace-pre-wrap text-xs text-slate-700">
                      {missingCities.join(", ")}
                    </pre>
                  </div>
                ) : (
                  <p className="text-slate-500">No missing cities were recorded for this draft.</p>
                )}
              </div>
            </details>
          ) : null}

          {writerMultiPass ? (
            <details className="rounded-md border border-slate-200 bg-white" open>
              <summary className="cursor-pointer px-3 py-2 text-sm font-medium text-slate-900">
                Writer multi-pass fallback ({writerMultiPass.batch_count} batches)
              </summary>
              <div className="space-y-2 border-t border-slate-200 px-3 py-2 text-xs text-slate-700">
                <p>
                  <span className="font-medium text-slate-900">Strategy:</span>{" "}
                  {writerMultiPass.strategy} + {writerMultiPass.combine_strategy}
                </p>
                <p>
                  <span className="font-medium text-slate-900">Analysis mode:</span>{" "}
                  {writerMultiPass.analysis_mode}
                </p>
                <p>
                  <span className="font-medium text-slate-900">Prompt size:</span>{" "}
                  {writerMultiPass.payload_tokens.toLocaleString()} tokens
                </p>
                <p>
                  <span className="font-medium text-slate-900">Split threshold:</span>{" "}
                  {writerMultiPass.threshold_tokens.toLocaleString()} tokens
                </p>
                <div>
                  <p className="font-medium text-slate-900">Batches</p>
                  <pre className="max-h-56 overflow-auto whitespace-pre-wrap text-xs text-slate-700">
                    {writerMultiPass.batches
                      .map(
                        (batch) =>
                          `#${batch.batch_index}: ${batch.payload_tokens.toLocaleString()} tokens, ${batch.excerpt_count} excerpts, ${batch.city_names.length} cities\n${batch.city_names.join(", ")}`,
                      )
                      .join("\n\n")}
                  </pre>
                </div>
              </div>
            </details>
          ) : null}

          {writerSavedEvidence ? (
            <details className="rounded-md border border-slate-200 bg-white" open>
              <summary className="cursor-pointer px-3 py-2 text-sm font-medium text-slate-900">
                Writer saved evidence ({writerSavedEvidence.saved_count} saved)
              </summary>
              <div className="space-y-2 border-t border-slate-200 px-3 py-2 text-xs text-slate-700">
                <p>
                  <span className="font-medium text-slate-900">Curator status:</span>{" "}
                  {writerSavedEvidence.curator_status ?? "n/a"}
                </p>
                <p>
                  <span className="font-medium text-slate-900">Covered cities:</span>{" "}
                  {writerSavedEvidence.covered_cities.length > 0
                    ? writerSavedEvidence.covered_cities.join(", ")
                    : "none"}
                </p>
                <div>
                  <p className="font-medium text-slate-900">Source kinds</p>
                  <pre className="max-h-32 overflow-auto whitespace-pre-wrap text-xs text-slate-700">
                    {formatJsonBlock(writerSavedEvidence.source_kind_counts)}
                  </pre>
                </div>
                {writerSavedEvidence.saved_evidence.length > 0 ? (
                  <div>
                    <p className="font-medium text-slate-900">Saved evidence</p>
                    <pre className="max-h-56 overflow-auto whitespace-pre-wrap text-xs text-slate-700">
                      {writerSavedEvidence.saved_evidence
                        .map(
                          (item) =>
                            `${item.saved_id} ${item.ref_id} ${item.source_kind} ${item.city_name || "n/a"} ${item.field || ""}\n${item.reason}`,
                        )
                        .join("\n\n")}
                    </pre>
                  </div>
                ) : null}
                {writerSavedEvidence.missing_records.length > 0 ? (
                  <div>
                    <p className="font-medium text-slate-900">
                      Missing records ({writerSavedEvidence.missing_records.length})
                    </p>
                    <pre className="max-h-48 overflow-auto whitespace-pre-wrap text-xs text-slate-700">
                      {writerSavedEvidence.missing_records
                        .map(
                          (item) =>
                            `${item.missing_id} ${item.city_name || "n/a"} ${item.field || ""}\n${item.reason}`,
                        )
                        .join("\n\n")}
                    </pre>
                  </div>
                ) : null}
              </div>
            </details>
          ) : null}

          <details className="rounded-md border border-slate-200 bg-white">
            <summary className="cursor-pointer px-3 py-2 text-sm font-medium text-slate-900">
              Warning entries ({warningEntries.length})
            </summary>
            <div className="border-t border-slate-200 px-3 py-2">
              {warningEntries.length > 0 ? (
                <pre className="max-h-64 overflow-auto whitespace-pre-wrap text-xs text-slate-700">
                  {warningEntries.join("\n")}
                </pre>
              ) : (
                <p className="text-xs text-slate-500">No warning or retry entries were captured.</p>
              )}
            </div>
          </details>

          <details className="rounded-md border border-slate-200 bg-white" open={!!errorLogText}>
            <summary className="cursor-pointer px-3 py-2 text-sm font-medium text-slate-900">
              Error log excerpt
            </summary>
            <div className="border-t border-slate-200 px-3 py-2">
              {errorLogText ? (
                <pre className="max-h-72 overflow-auto whitespace-pre-wrap text-xs text-slate-700">
                  {errorLogText}
                </pre>
              ) : (
                <p className="text-xs text-slate-500">No error excerpt was captured for this run.</p>
              )}
            </div>
          </details>

          <details className="rounded-md border border-slate-200 bg-white">
            <summary className="cursor-pointer px-3 py-2 text-sm font-medium text-slate-900">
              Recent log tail ({logTail.length} lines)
            </summary>
            <div className="border-t border-slate-200 px-3 py-2">
              {logTail.length > 0 ? (
                <pre className="max-h-72 overflow-auto whitespace-pre-wrap text-xs text-slate-700">
                  {logTail.join("\n")}
                </pre>
              ) : (
                <p className="text-xs text-slate-500">No run log is available for this run yet.</p>
              )}
            </div>
          </details>

          <details className="rounded-md border border-slate-200 bg-white">
            <summary className="cursor-pointer px-3 py-2 text-sm font-medium text-slate-900">
              Retry summary
            </summary>
            <div className="border-t border-slate-200 px-3 py-2">
              <pre className="max-h-48 overflow-auto whitespace-pre-wrap text-xs text-slate-700">
                {formatJsonBlock(diagnostics?.retry_summary)}
              </pre>
            </div>
          </details>

          <details className="rounded-md border border-slate-200 bg-white">
            <summary className="cursor-pointer px-3 py-2 text-sm font-medium text-slate-900">
              LLM usage
            </summary>
            <div className="border-t border-slate-200 px-3 py-2">
              <pre className="max-h-48 overflow-auto whitespace-pre-wrap text-xs text-slate-700">
                {formatJsonBlock(diagnostics?.llm_usage)}
              </pre>
            </div>
          </details>

          <details className="rounded-md border border-slate-200 bg-white">
            <summary className="cursor-pointer px-3 py-2 text-sm font-medium text-slate-900">
              Artifact paths
            </summary>
            <div className="border-t border-slate-200 px-3 py-2 text-xs text-slate-700">
              <p>
                <span className="font-medium text-slate-900">run.log:</span>{" "}
                {diagnostics?.artifacts.run_log ?? "n/a"}
              </p>
              <p>
                <span className="font-medium text-slate-900">run_summary.txt:</span>{" "}
                {diagnostics?.artifacts.run_summary ?? "n/a"}
              </p>
              <p>
                <span className="font-medium text-slate-900">error_log.txt:</span>{" "}
                {diagnostics?.artifacts.error_log ?? "n/a"}
              </p>
            </div>
          </details>
        </div>
      ) : null}
    </div>
  );
}
