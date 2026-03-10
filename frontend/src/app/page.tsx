"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  CircleDashed,
  Loader2,
  MessageSquareText,
  PanelLeftClose,
  PanelLeftOpen,
  RefreshCw,
  Sparkles,
} from "lucide-react";

import { AssumptionsWorkspace } from "@/components/assumptions-workspace";
import { ContextChatWorkspace } from "@/components/context-chat-workspace";
import { DevModeToggle } from "@/components/dev-mode-toggle";
import { DevToolsPanel } from "@/components/dev-tools-panel";
import { MarkdownWithReferences } from "@/components/markdown-with-references";
import { SearchableCityPicker } from "@/components/searchable-city-picker";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Textarea } from "@/components/ui/textarea";
import {
  FrontendMode,
  getDefaultFrontendMode,
  getDevFeatureFlags,
  persistFrontendMode,
  readStoredFrontendMode,
} from "@/lib/frontend-mode";
import { formatCityLabel } from "@/lib/utils";
import {
  CityGroup,
  CreateRunResponse,
  RunContextResponse,
  RunOutputResponse,
  RunSummary,
  RunStatus,
  RunStatusResponse,
  fetchCities,
  fetchCityGroups,
  fetchRuns,
  fetchRunContext,
  fetchRunOutput,
  fetchRunStatus,
  getApiBaseUrl,
  startRun,
} from "@/lib/api";

const TERMINAL_STATUSES: RunStatus[] = [
  "completed",
  "completed_with_gaps",
  "failed",
  "stopped",
];

type CityScopeMode = "all" | "group" | "manual";
type AnalysisMode = "aggregate" | "city_by_city";
const LAST_RUN_ID_STORAGE_KEY = "last_run_id";
const CONTROLS_COLLAPSED_STORAGE_KEY = "build_controls_collapsed";

export default function Home() {
  const [question, setQuestion] = useState("");
  const [scopeMode, setScopeMode] = useState<CityScopeMode>("all");
  const [analysisMode, setAnalysisMode] = useState<AnalysisMode>("aggregate");
  const [cities, setCities] = useState<string[]>([]);
  const [selectedCities, setSelectedCities] = useState<string[]>([]);
  const [cityGroups, setCityGroups] = useState<CityGroup[]>([]);
  const [selectedGroupId, setSelectedGroupId] = useState<string | null>(null);
  const [isLoadingCities, setIsLoadingCities] = useState(false);
  const [citiesError, setCitiesError] = useState<string | null>(null);
  const [isLoadingGroups, setIsLoadingGroups] = useState(false);
  const [groupsError, setGroupsError] = useState<string | null>(null);

  const [runResponse, setRunResponse] = useState<CreateRunResponse | null>(null);
  const [runStatus, setRunStatus] = useState<RunStatusResponse | null>(null);
  const [runOutput, setRunOutput] = useState<RunOutputResponse | null>(null);
  const [runContext, setRunContext] = useState<RunContextResponse | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isPolling, setIsPolling] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [availableRuns, setAvailableRuns] = useState<RunSummary[]>([]);
  const [selectedExistingRunId, setSelectedExistingRunId] = useState("");
  const [isLoadingRuns, setIsLoadingRuns] = useState(false);
  const [runsError, setRunsError] = useState<string | null>(null);
  const [isLoadingSelectedRun, setIsLoadingSelectedRun] = useState(false);

  const [chatOpen, setChatOpen] = useState(false);
  const [assumptionsOpen, setAssumptionsOpen] = useState(false);
  const [isControlsCollapsed, setIsControlsCollapsed] = useState(false);
  const [frontendMode, setFrontendMode] = useState<FrontendMode>(getDefaultFrontendMode());
  const [hasHydratedFrontendMode, setHasHydratedFrontendMode] = useState(false);

  const runId = runResponse?.run_id ?? null;
  const statusValue = runStatus?.status ?? runResponse?.status ?? null;
  const canFetchArtifacts = statusValue === "completed" || statusValue === "completed_with_gaps";
  const documentReady = !!runOutput?.content && canFetchArtifacts;
  const devFeatures = useMemo(() => getDevFeatureFlags(frontendMode), [frontendMode]);

  useEffect(() => {
    const storedMode = readStoredFrontendMode();
    if (storedMode) {
      setFrontendMode(storedMode);
    }
    setHasHydratedFrontendMode(true);
  }, []);

  useEffect(() => {
    if (!hasHydratedFrontendMode) {
      return;
    }
    persistFrontendMode(frontendMode);
  }, [frontendMode, hasHydratedFrontendMode]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const stored = window.localStorage.getItem(CONTROLS_COLLAPSED_STORAGE_KEY);
    setIsControlsCollapsed(stored === "1");
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    window.localStorage.setItem(
      CONTROLS_COLLAPSED_STORAGE_KEY,
      isControlsCollapsed ? "1" : "0",
    );
  }, [isControlsCollapsed]);

  const hydrateRunById = useCallback(async (targetRunId: string): Promise<void> => {
    const trimmedRunId = targetRunId.trim();
    if (!trimmedRunId) {
      return;
    }
    const statusPayload = await fetchRunStatus(trimmedRunId);
    setRunResponse({
      run_id: trimmedRunId,
      status: statusPayload.status,
      status_url: `${getApiBaseUrl()}/api/v1/runs/${trimmedRunId}/status`,
      output_url: `${getApiBaseUrl()}/api/v1/runs/${trimmedRunId}/output`,
      context_url: `${getApiBaseUrl()}/api/v1/runs/${trimmedRunId}/context`,
    });
    setRunStatus(statusPayload);
    setRunOutput(null);
    setRunContext(null);

    if (
      statusPayload.status === "completed" ||
      statusPayload.status === "completed_with_gaps"
    ) {
      const [outputPayload, contextPayload] = await Promise.all([
        fetchRunOutput(trimmedRunId),
        fetchRunContext(trimmedRunId),
      ]);
      setRunOutput(outputPayload);
      setRunContext(contextPayload);
    }
    if (typeof window !== "undefined") {
      window.localStorage.setItem(LAST_RUN_ID_STORAGE_KEY, trimmedRunId);
    }
    setSelectedExistingRunId(trimmedRunId);
  }, []);

  const refreshRunList = useCallback(
    async (preferredRunId?: string): Promise<void> => {
      setIsLoadingRuns(true);
      setRunsError(null);
      try {
        const payload = await fetchRuns();
        setAvailableRuns(payload.runs);
        setSelectedExistingRunId((current) => {
          const preferred = (preferredRunId ?? current).trim();
          if (preferred && payload.runs.some((run) => run.run_id === preferred)) {
            return preferred;
          }
          if (payload.runs.length > 0) {
            return payload.runs[0].run_id;
          }
          return "";
        });
      } catch (error) {
        setRunsError(error instanceof Error ? error.message : "Failed to load runs.");
      } finally {
        setIsLoadingRuns(false);
      }
    },
    [],
  );

  async function handleLoadExistingRun(): Promise<void> {
    const trimmed = selectedExistingRunId.trim();
    if (!trimmed || isLoadingSelectedRun) {
      return;
    }
    setIsLoadingSelectedRun(true);
    setRunError(null);
    setChatOpen(false);
    setAssumptionsOpen(false);
    try {
      await hydrateRunById(trimmed);
    } catch (error) {
      setRunError(
        error instanceof Error ? error.message : "Failed to load selected run.",
      );
    } finally {
      setIsLoadingSelectedRun(false);
    }
  }

  useEffect(() => {
    void refreshRunList();
  }, [refreshRunList]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const storedRunId = (window.localStorage.getItem(LAST_RUN_ID_STORAGE_KEY) ?? "").trim();
    if (!storedRunId) {
      return;
    }
    setSelectedExistingRunId(storedRunId);
    let cancelled = false;
    setIsLoadingSelectedRun(true);
    void refreshRunList(storedRunId);
    hydrateRunById(storedRunId)
      .catch(() => {
        // Ignore stale run ids on startup; user can load another run manually.
      })
      .finally(() => {
        if (!cancelled) {
          setIsLoadingSelectedRun(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [refreshRunList, hydrateRunById]);

  useEffect(() => {
    let cancelled = false;
    setIsLoadingCities(true);
    setCitiesError(null);
    fetchCities()
      .then((payload) => {
        if (cancelled) {
          return;
        }
        setCities(payload.cities);
      })
      .catch((error) => {
        if (cancelled) {
          return;
        }
        setCitiesError(error instanceof Error ? error.message : "Failed to load cities.");
      })
      .finally(() => {
        if (!cancelled) {
          setIsLoadingCities(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    setIsLoadingGroups(true);
    setGroupsError(null);
    fetchCityGroups()
      .then((payload) => {
        if (cancelled) {
          return;
        }
        setCityGroups(payload.groups);
      })
      .catch((error) => {
        if (cancelled) {
          return;
        }
        setGroupsError(
          error instanceof Error ? error.message : "Failed to load city groups.",
        );
        setCityGroups([]);
      })
      .finally(() => {
        if (!cancelled) {
          setIsLoadingGroups(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (selectedGroupId && cityGroups.some((group) => group.id === selectedGroupId)) {
      return;
    }
    setSelectedGroupId(cityGroups.length > 0 ? cityGroups[0].id : null);
  }, [cityGroups, selectedGroupId]);

  useEffect(() => {
    if (!runId || !statusValue || !["queued", "running"].includes(statusValue)) {
      setIsPolling(false);
      return;
    }
    let cancelled = false;
    let nextPollHandle: ReturnType<typeof setTimeout> | null = null;
    let activeController: AbortController | null = null;
    setIsPolling(true);

    const pollOnce = (): void => {
      if (cancelled) {
        return;
      }
      activeController = new AbortController();
      fetchRunStatus(runId, { signal: activeController.signal })
        .then((payload) => {
          if (cancelled) {
            return;
          }
          setRunStatus(payload);
        })
        .catch((error) => {
          if (cancelled) {
            return;
          }
          if (error instanceof DOMException && error.name === "AbortError") {
            return;
          }
          setRunError(error instanceof Error ? error.message : "Status polling failed.");
        })
        .finally(() => {
          activeController = null;
          if (cancelled) {
            return;
          }
          nextPollHandle = setTimeout(() => {
            pollOnce();
          }, 2500);
        });
    };

    pollOnce();

    return () => {
      cancelled = true;
      if (nextPollHandle) {
        clearTimeout(nextPollHandle);
      }
      activeController?.abort();
      setIsPolling(false);
    };
  }, [runId, statusValue]);

  useEffect(() => {
    if (!runId || !runStatus || !TERMINAL_STATUSES.includes(runStatus.status)) {
      return;
    }
    if (!canFetchArtifacts) {
      return;
    }
    let cancelled = false;
    Promise.all([fetchRunOutput(runId), fetchRunContext(runId)])
      .then(([outputPayload, contextPayload]) => {
        if (cancelled) {
          return;
        }
        setRunOutput(outputPayload);
        setRunContext(contextPayload);
      })
      .catch((error) => {
        if (cancelled) {
          return;
        }
        setRunError(error instanceof Error ? error.message : "Failed to load run artifacts.");
      });
    return () => {
      cancelled = true;
    };
  }, [runId, runStatus, canFetchArtifacts]);

  const selectedGroup = useMemo(() => {
    if (!selectedGroupId) {
      return null;
    }
    return cityGroups.find((group) => group.id === selectedGroupId) ?? null;
  }, [cityGroups, selectedGroupId]);

  const effectiveScopeCities = useMemo(() => {
    if (scopeMode === "all") {
      return cities;
    }
    if (scopeMode === "group") {
      return selectedGroup?.cities ?? [];
    }
    return selectedCities;
  }, [cities, scopeMode, selectedCities, selectedGroup]);

  const scopeCountLabel = useMemo(() => {
    if (scopeMode === "all") {
      return `${cities.length} cities`;
    }
    return `${effectiveScopeCities.length} selected`;
  }, [cities.length, effectiveScopeCities.length, scopeMode]);

  function toggleCity(city: string): void {
    setSelectedCities((current) =>
      current.includes(city)
        ? current.filter((value) => value !== city)
        : [...current, city],
    );
  }

  async function handleBuildDocument(): Promise<void> {
    const trimmed = question.trim();
    if (!trimmed || isSubmitting) {
      return;
    }
    const scopedCities =
      scopeMode === "group" ? (selectedGroup?.cities ?? []) : selectedCities;
    if (scopeMode === "group" && scopedCities.length === 0) {
      setRunError("Select a predefined city group before starting the build.");
      return;
    }
    if (scopeMode === "manual" && scopedCities.length === 0) {
      setRunError("Select at least one city for manual scope.");
      return;
    }

    setIsSubmitting(true);
    setRunError(null);
    setRunOutput(null);
    setRunContext(null);
    setRunResponse(null);
    setRunStatus(null);
    setChatOpen(false);
    setAssumptionsOpen(false);

    try {
      const payload = await startRun({
        question: trimmed,
        cities: scopeMode === "all" ? undefined : scopedCities,
        analysis_mode: analysisMode,
      });
      setRunResponse(payload);
      setSelectedExistingRunId(payload.run_id);
      if (typeof window !== "undefined") {
        window.localStorage.setItem(LAST_RUN_ID_STORAGE_KEY, payload.run_id);
      }
      void refreshRunList(payload.run_id);
      const initialStatus = await fetchRunStatus(payload.run_id);
      setRunStatus(initialStatus);
    } catch (error) {
      setRunError(
        error instanceof Error ? error.message : "Document build trigger failed.",
      );
    } finally {
      setIsSubmitting(false);
    }
  }

  function formatRunOptionLabel(run: RunSummary): string {
    const compactQuestion = run.question.replace(/\s+/g, " ").trim();
    const preview =
      compactQuestion.length > 56
        ? `${compactQuestion.slice(0, 53)}...`
        : compactQuestion;
    return `${run.run_id} | ${preview || "No question"}`;
  }

  const isTerminal = !!statusValue && TERMINAL_STATUSES.includes(statusValue);
  const isLongWait = !!statusValue && ["queued", "running"].includes(statusValue);
  const hasValidScope =
    scopeMode === "all" ||
    (scopeMode === "group"
      ? (selectedGroup?.cities.length ?? 0) > 0
      : selectedCities.length > 0);
  const hasApiKeyIssue =
    /api key|authentication|unauthorized|401|403/i.test(runError ?? "") ||
    /api key|authentication|unauthorized|401|403/i.test(
      runStatus?.error?.message ?? "",
    );

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_20%_20%,#f8edd6_0%,#f2f6f6_45%,#eef2ff_100%)] px-4 py-8 md:px-8">
      <div className="mx-auto max-w-7xl space-y-6">
        <header className="rounded-xl border border-slate-200 bg-white/80 p-6 shadow-sm backdrop-blur-sm">
          <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
            <div>
              <p className="mb-1 text-xs font-semibold uppercase tracking-[0.18em] text-amber-700">
                Document Builder
              </p>
              <h1 className="text-3xl font-semibold text-slate-900 md:text-4xl">
                Build the answer as a report, then explore it.
              </h1>
              <p className="mt-2 max-w-3xl text-sm text-slate-600 md:text-base">
                This flow is document-first. You submit a build run, wait for completion, review the generated document, then switch into context chat workspace.
              </p>
            </div>
            <DevModeToggle mode={frontendMode} onModeChange={setFrontendMode} />
          </div>
        </header>

        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={() => setIsControlsCollapsed((current) => !current)}
          aria-label={isControlsCollapsed ? "Show controls panel" : "Hide controls panel"}
          className="group fixed left-0 top-1/2 z-40 h-10 w-10 -translate-y-1/2 justify-start gap-2 overflow-hidden rounded-l-none rounded-r-full border border-slate-300 bg-white/90 px-3 text-slate-700 shadow-sm backdrop-blur-sm transition-all duration-300 ease-out hover:w-40 focus-visible:w-40"
        >
          <span className="shrink-0">
            {isControlsCollapsed ? (
              <PanelLeftOpen className="h-4 w-4" />
            ) : (
              <PanelLeftClose className="h-4 w-4" />
            )}
          </span>
          <span className="max-w-0 overflow-hidden whitespace-nowrap text-xs font-medium opacity-0 transition-all duration-300 ease-out group-hover:max-w-24 group-hover:opacity-100 group-focus-visible:max-w-24 group-focus-visible:opacity-100">
            {isControlsCollapsed ? "Show Controls" : "Hide Controls"}
          </span>
        </Button>

        <main className="flex flex-col gap-6 lg:flex-row">
          <div
            className={`overflow-hidden transition-[width,opacity,transform] duration-300 ease-in-out lg:shrink-0 ${
              isControlsCollapsed
                ? "lg:w-0 lg:-translate-x-4 lg:opacity-0 lg:pointer-events-none"
                : "lg:w-[26rem] lg:translate-x-0 lg:opacity-100"
            }`}
          >
            <Card className="h-fit border-slate-300">
              <CardHeader>
                <CardTitle>Build Controls</CardTitle>
                <CardDescription>Select scope and trigger a long-running build.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-5">
              <div className="space-y-2">
                <Label htmlFor="question">Question</Label>
                <Textarea
                  id="question"
                  placeholder="Example: Build a concise report for selected cities on main climate initiatives and progress."
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  className="min-h-32"
                />
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label htmlFor="existing-run">Load Previous Answer</Label>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={() => void refreshRunList(selectedExistingRunId)}
                    disabled={isLoadingRuns}
                    className="h-7 px-2 text-xs"
                  >
                    {isLoadingRuns ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : null}
                    Refresh
                  </Button>
                </div>
                <div className="flex gap-2">
                  <select
                    id="existing-run"
                    value={selectedExistingRunId}
                    onChange={(event) => setSelectedExistingRunId(event.target.value)}
                    disabled={isLoadingRuns || availableRuns.length === 0}
                    className="h-11 w-full rounded-md border border-slate-300 bg-white px-3 text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-slate-300 disabled:cursor-not-allowed disabled:bg-slate-100"
                  >
                    {availableRuns.length === 0 ? (
                      <option value="">
                        {isLoadingRuns ? "Loading runs..." : "No runs found"}
                      </option>
                    ) : null}
                    {availableRuns.map((run) => (
                      <option key={run.run_id} value={run.run_id}>
                        {formatRunOptionLabel(run)}
                      </option>
                    ))}
                  </select>
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => void handleLoadExistingRun()}
                    disabled={isLoadingSelectedRun || !selectedExistingRunId.trim()}
                  >
                    {isLoadingSelectedRun ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
                    Load
                  </Button>
                </div>
                <p className="text-xs text-slate-500">
                  {availableRuns.length} runs discovered in backend storage.
                </p>
                {runsError ? <p className="text-xs text-red-600">{runsError}</p> : null}
                <p className="text-xs text-slate-500">
                  Load a previous answer without re-running the full pipeline.
                </p>
              </div>

              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label>City scope</Label>
                  <Badge variant="secondary">{scopeCountLabel}</Badge>
                </div>
                <div className="grid grid-cols-3 gap-2">
                  <Button
                    type="button"
                    variant={scopeMode === "all" ? "default" : "outline"}
                    onClick={() => setScopeMode("all")}
                    className="w-full"
                  >
                    All
                  </Button>
                  <Button
                    type="button"
                    variant={scopeMode === "group" ? "default" : "outline"}
                    onClick={() => setScopeMode("group")}
                    className="w-full"
                  >
                    Group
                  </Button>
                  <Button
                    type="button"
                    variant={scopeMode === "manual" ? "default" : "outline"}
                    onClick={() => setScopeMode("manual")}
                    className="w-full"
                  >
                    Manual
                  </Button>
                </div>

                {scopeMode === "all" ? (
                  <div className="rounded-md border border-slate-200 bg-slate-50 p-3 text-sm text-slate-700">
                    This will process all cities and take a lot of time.
                  </div>
                ) : null}

                {scopeMode === "group" ? (
                  <div className="space-y-2 rounded-md border border-slate-200 p-3">
                    {isLoadingGroups ? (
                      <p className="text-sm text-slate-500">Loading predefined groups...</p>
                    ) : groupsError ? (
                      <p className="text-sm text-red-600">{groupsError}</p>
                    ) : cityGroups.length === 0 ? (
                      <p className="text-sm text-slate-500">
                        No predefined groups are available.
                      </p>
                    ) : (
                      <>
                        <div className="flex flex-wrap gap-2">
                          {cityGroups.map((group) => {
                            const selected = selectedGroupId === group.id;
                            return (
                              <Button
                                key={group.id}
                                type="button"
                                variant={selected ? "default" : "outline"}
                                size="sm"
                                onClick={() => setSelectedGroupId(group.id)}
                                className="h-8 rounded-full px-3"
                              >
                                {group.name}
                              </Button>
                            );
                          })}
                        </div>
                        {selectedGroup ? (
                          <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
                            <p className="text-sm font-medium text-slate-800">
                              {selectedGroup.name}
                            </p>
                            {selectedGroup.description ? (
                              <p className="text-xs text-slate-600">
                                {selectedGroup.description}
                              </p>
                            ) : null}
                            <p className="mt-1 text-xs text-slate-600">
                              Cities: {selectedGroup.cities.map(formatCityLabel).join(", ")}
                            </p>
                          </div>
                        ) : null}
                      </>
                    )}
                  </div>
                ) : null}

                {scopeMode === "manual" ? (
                  <SearchableCityPicker
                    cities={cities}
                    selectedCities={selectedCities}
                    onSelectCity={toggleCity}
                    errorMessage={citiesError}
                    isLoading={isLoadingCities}
                    loadingMessage="Loading cities..."
                  />
                ) : null}
              </div>

              <div className="space-y-3 rounded-md border border-slate-200 p-3">
                <div className="flex items-center justify-between">
                  <Label>Answer mode</Label>
                  <Badge variant="secondary">
                    {analysisMode === "aggregate" ? "Aggregate Mode" : "City-by-City Mode"}
                  </Badge>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <Button
                    type="button"
                    variant={analysisMode === "aggregate" ? "default" : "outline"}
                    onClick={() => setAnalysisMode("aggregate")}
                    className="w-full"
                  >
                    Aggregate Mode
                  </Button>
                  <Button
                    type="button"
                    variant={analysisMode === "city_by_city" ? "default" : "outline"}
                    onClick={() => setAnalysisMode("city_by_city")}
                    className="w-full"
                  >
                    City-by-City Mode
                  </Button>
                </div>
                <p className="text-xs text-slate-600">
                  {analysisMode === "aggregate"
                    ? "One integrated answer across selected cities."
                    : "Aswering one city section at a time; similarities at the end."}
                </p>
              </div>

              <Button
                onClick={handleBuildDocument}
                disabled={isSubmitting || !question.trim() || !hasValidScope}
                className="w-full"
              >
                {isSubmitting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
                Generate Report
              </Button>

              <Separator />

              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <p className="text-sm font-medium text-slate-800">Run Status</p>
                  {statusValue ? <Badge variant="outline">{statusValue}</Badge> : null}
                </div>
                {!runId ? (
                  <p className="text-sm text-slate-500">No run submitted yet.</p>
                ) : isLongWait ? (
                  <div className="rounded-md border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
                    <div className="mb-2 flex items-center gap-2 font-medium">
                      <CircleDashed className="h-4 w-4 animate-spin" />
                      Build in progress
                    </div>
                    <p>Leave this page open. Document generation may take several minutes for broad questions.</p>
                  </div>
                ) : isTerminal ? (
                  <div className="rounded-md border border-slate-200 bg-slate-50 p-3 text-sm text-slate-700">
                    <div className="mb-1 flex items-center gap-2 font-medium">
                      {statusValue === "completed" || statusValue === "completed_with_gaps" ? (
                        <CheckCircle2 className="h-4 w-4 text-teal-700" />
                      ) : (
                        <AlertTriangle className="h-4 w-4 text-red-600" />
                      )}
                      Terminal status: {statusValue}
                    </div>
                    {runStatus?.error ? (
                      <p className="text-xs text-red-700">
                        {runStatus.error.code}: {runStatus.error.message}
                      </p>
                    ) : null}
                    {hasApiKeyIssue ? (
                      <p className="mt-1 text-xs text-amber-700">
                        API key issue detected. Verify backend OpenRouter credentials and retry the run.
                      </p>
                    ) : null}
                  </div>
                ) : null}
                {runError ? <p className="text-sm text-red-600">{runError}</p> : null}
              </div>

              {devFeatures.showRunId || devFeatures.showApiKeyControls ? (
                <>
                  <Separator />
                  <DevToolsPanel apiKeyIssue={hasApiKeyIssue} runId={runId} />
                </>
              ) : null}
              </CardContent>
            </Card>
          </div>

          <div className="min-w-0 flex-1">
            {devFeatures.showAssumptionsEntry && assumptionsOpen && documentReady && runId ? (
              <AssumptionsWorkspace
                runId={runId}
                enabled={documentReady}
                onClose={() => setAssumptionsOpen(false)}
              />
            ) : chatOpen && documentReady && runId ? (
              <ContextChatWorkspace
                runId={runId}
                enabled={documentReady}
                onClose={() => setChatOpen(false)}
                showContextManager={devFeatures.showContextManager}
                showTokenMetrics={devFeatures.showChatTokenMetrics}
              />
            ) : (
              <Card className="border-slate-300">
                <CardHeader className="pb-4">
                  <div>
                    <div>
                      <CardTitle>Generated Document</CardTitle>
                      <CardDescription>
                        The main answer is rendered as a report. Context chat opens as a dedicated workspace.
                      </CardDescription>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  {documentReady ? (
                    <>
                      <div className="mb-3 flex justify-end">
                        <div className="flex gap-2">
                          {devFeatures.showAssumptionsEntry ? (
                            <Button
                              type="button"
                              variant="outline"
                              onClick={() => {
                                setChatOpen(false);
                                setAssumptionsOpen(true);
                              }}
                              disabled={!runId}
                            >
                              <Sparkles className="h-4 w-4" />
                              Assumptions Review
                            </Button>
                          ) : null}
                          <Button
                            type="button"
                            onClick={() => {
                              setAssumptionsOpen(false);
                              setChatOpen(true);
                            }}
                            disabled={!runId}
                          >
                            <MessageSquareText className="h-4 w-4" />
                            Chat About the Answer
                          </Button>
                        </div>
                      </div>
                      <article className="document-markdown rounded-md border border-slate-200 bg-white p-5 shadow-inner">
                        <MarkdownWithReferences
                          content={runOutput.content}
                          runId={runId}
                        />
                      </article>
                    </>
                  ) : isLongWait ? (
                    <div className="space-y-3 rounded-md border border-slate-200 bg-white p-6">
                      <div className="flex items-center gap-2 text-slate-700">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        Generating document...
                      </div>
                      <div className="space-y-2">
                        <div className="h-2 animate-pulse rounded bg-slate-200" />
                        <div className="h-2 w-11/12 animate-pulse rounded bg-slate-200" />
                        <div className="h-2 w-10/12 animate-pulse rounded bg-slate-200" />
                        <div className="h-2 w-8/12 animate-pulse rounded bg-slate-200" />
                      </div>
                    </div>
                  ) : (
                    <div className="rounded-md border border-dashed border-slate-300 bg-white p-8 text-center text-slate-600">
                      <p className="text-base font-medium">Document output will appear here.</p>
                      <p className="mt-1 text-sm">
                        {isControlsCollapsed
                          ? "Open Build Controls to start a run."
                          : "Submit a run from the left panel to start building."}
                      </p>
                    </div>
                  )}

                  {runContext ? (
                    <p className="mt-4 text-xs text-slate-500">
                      Context bundle loaded from: {runContext.context_bundle_path}
                    </p>
                  ) : null}
                </CardContent>
              </Card>
            )}
          </div>
        </main>
      </div>

      {isPolling ? (
        <button
          type="button"
          className="fixed bottom-6 left-6 inline-flex items-center gap-2 rounded-full border border-slate-300 bg-white/90 px-3 py-2 text-xs font-medium text-slate-700 shadow-sm"
          onClick={() => {
            if (!runId) {
              return;
            }
            void fetchRunStatus(runId).then(setRunStatus).catch(() => {
              setRunError("Manual refresh failed.");
            });
          }}
        >
          <RefreshCw className="h-3.5 w-3.5" />
          Polling run status
        </button>
      ) : null}
    </div>
  );
}
