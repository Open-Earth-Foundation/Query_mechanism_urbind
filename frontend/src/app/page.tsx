"use client";

import {
  useCallback,
  useDeferredValue,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  AlertTriangle,
  BookOpen,
  CheckCircle2,
  CircleDashed,
  FileText,
  Loader2,
  MessageSquareText,
  ChevronDown,
  PanelLeftClose,
  PanelLeftOpen,
  RefreshCw,
  ScanSearch,
  Settings2,
  Sparkles,
} from "lucide-react";

import { EnrichmentProcessWorkspace } from "@/components/enrichment-process-workspace";
import { AssumptionsWorkspace } from "@/components/assumptions-workspace";
import { CccDocumentRail } from "@/components/ccc-document-rail";
import { LiveBuildTimeline } from "@/components/pipeline/live-build-timeline";
import { ContextChatWorkspace } from "@/components/context-chat/context-chat-workspace";
import { DevModeToggle } from "@/components/dev-mode-toggle";
import { DevToolsPanel } from "@/components/dev-tools-panel";
import { DocumentExportControls } from "@/components/document-export-controls";
import { MarkdownWithReferences } from "@/components/markdown-with-references";
import { RunDiagnosticsPanel } from "@/components/run-diagnostics-panel";
import { SearchableCityPicker } from "@/components/searchable-city-picker";
import { SearchableRunPicker } from "@/components/searchable-run-picker";
import { LogoutButton } from "@/components/logout-button";
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
  CityMarkdownResponse,
  CreateRunResponse,
  RunArtifactsResponse,
  RunContextResponse,
  RunOutputResponse,
  RunSummary,
  RunStatus,
  RunStatusResponse,
  VectorStoreWarmupResponse,
  fetchCities,
  fetchCityMarkdown,
  fetchCityGroups,
  fetchRuns,
  fetchRunArtifacts,
  fetchRunContext,
  fetchRunOutput,
  fetchRunStatus,
  fetchVectorStoreStatus,
  getApiBaseUrl,
  startRun,
} from "@/lib/api";

const TERMINAL_STATUSES: RunStatus[] = [
  "completed",
  "completed_with_gaps",
  "failed",
  "stopped",
];
const RUN_STATUS_POLL_INTERVAL_MS = 2500;

type CityScopeMode = "all" | "group" | "manual";
type AnalysisMode = "aggregate" | "city_by_city";
type TabKey = "document" | "enrichment" | "chat" | "ccc" | "assumptions";
const LAST_RUN_ID_STORAGE_KEY = "last_run_id";
const LAST_CCC_CITY_STORAGE_KEY = "last_ccc_city";
const CONTROLS_COLLAPSED_STORAGE_KEY = "build_controls_collapsed";

function formatRunOptionLabel(run: RunSummary): string {
  const compactQuestion = run.question.replace(/\s+/g, " ").trim();
  const preview =
    compactQuestion.length > 56 ? `${compactQuestion.slice(0, 53)}...` : compactQuestion;
  const statusPrefix =
    run.status === "completed" || run.status === "completed_with_gaps"
      ? ""
      : `[${run.status}] `;
  const pickerLabel = run.picker_timestamp || run.run_id;
  return `${statusPrefix}${pickerLabel} | ${preview || "No question"}`;
}

function normalizeCitySelectionKey(value: string): string {
  const cleaned = value.trim().toLowerCase();
  if (!cleaned) {
    return "";
  }
  return cleaned.replace(/[^\p{L}\p{N}]+/gu, "_").replace(/^_+|_+$/g, "");
}

function isObjectRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function readStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .filter((item): item is string => typeof item === "string")
    .map((item) => item.trim())
    .filter((item) => item.length > 0);
}

function extractRunContextCityNames(runContext: RunContextResponse | null): string[] {
  if (!runContext) {
    return [];
  }
  const rootBundle = isObjectRecord(runContext.context_bundle)
    ? runContext.context_bundle
    : null;

  const orderedCandidates = [
    ...readStringArray(rootBundle?.selected_city_names),
    ...readStringArray(rootBundle?.inspected_city_names),
    ...readStringArray(rootBundle?.selected_cities),
    ...readStringArray(rootBundle?.inspected_cities),
  ];
  const uniqueCityNames: string[] = [];
  const seen = new Set<string>();

  orderedCandidates.forEach((city) => {
    const key = normalizeCitySelectionKey(city);
    if (!key || seen.has(key)) {
      return;
    }
    seen.add(key);
    uniqueCityNames.push(city);
  });

  return uniqueCityNames;
}

function pickFirstAvailableCity(candidates: string[], availableCities: string[]): string | null {
  if (candidates.length === 0 || availableCities.length === 0) {
    return null;
  }

  const availableByKey = new Map(
    availableCities.map((city) => [normalizeCitySelectionKey(city), city]),
  );
  for (const candidate of candidates) {
    const resolved = availableByKey.get(normalizeCitySelectionKey(candidate));
    if (resolved) {
      return resolved;
    }
  }
  return null;
}

export default function Home() {
  const [question, setQuestion] = useState("");
  const [query2, setQuery2] = useState("");
  const [query3, setQuery3] = useState("");
  const [scopeMode, setScopeMode] = useState<CityScopeMode>("all");
  const [analysisMode, setAnalysisMode] = useState<AnalysisMode>("aggregate");
  const [enrichmentEnabled, setEnrichmentEnabled] = useState(true);
  const [webResearchEnabled, setWebResearchEnabled] = useState(false);
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
  const [liveArtifacts, setLiveArtifacts] = useState<RunArtifactsResponse | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isPolling, setIsPolling] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [availableRuns, setAvailableRuns] = useState<RunSummary[]>([]);
  const [knownRunsById, setKnownRunsById] = useState<Record<string, RunSummary>>({});
  const [selectedExistingRunId, setSelectedExistingRunId] = useState("");
  const [runSearchQuery, setRunSearchQuery] = useState("");
  const [isLoadingRuns, setIsLoadingRuns] = useState(false);
  const [runsError, setRunsError] = useState<string | null>(null);
  const [isLoadingSelectedRun, setIsLoadingSelectedRun] = useState(false);
  const [vectorStoreStatus, setVectorStoreStatus] =
    useState<VectorStoreWarmupResponse | null>(null);
  const [vectorStoreStatusError, setVectorStoreStatusError] = useState<string | null>(
    null,
  );

  const [activeTab, setActiveTab] = useState<TabKey>("document");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [isControlsCollapsed, setIsControlsCollapsed] = useState(false);
  const [selectedCccCity, setSelectedCccCity] = useState<string | null>(null);
  const [cccDocumentCache, setCccDocumentCache] = useState<
    Record<string, CityMarkdownResponse>
  >({});
  const [isLoadingCccDocument, setIsLoadingCccDocument] = useState(false);
  const [cccDocumentError, setCccDocumentError] = useState<string | null>(null);
  const [frontendMode, setFrontendMode] = useState<FrontendMode>(getDefaultFrontendMode());
  const [hasHydratedFrontendMode, setHasHydratedFrontendMode] = useState(false);
  const cccRunDefaultRef = useRef<string | null>(null);
  const runListAbortControllerRef = useRef<AbortController | null>(null);

  const runId = runResponse?.run_id ?? null;
  const statusValue = runStatus?.status ?? runResponse?.status ?? null;
  const canFetchArtifacts = statusValue === "completed" || statusValue === "completed_with_gaps";
  const documentReady = !!runOutput?.content && canFetchArtifacts;
  const devFeatures = useMemo(() => getDevFeatureFlags(frontendMode), [frontendMode]);
  const isDevMode = frontendMode === "dev";
  const advancedExpanded = isDevMode || showAdvanced;
  // Right-panel tab is the single source of truth for which view is shown.
  const chatOpen = activeTab === "chat";
  const auditOpen = activeTab === "enrichment";
  const assumptionsOpen = activeTab === "assumptions";
  const cccTabActive = activeTab === "ccc";

  const selectedExistingRunSummary = useMemo(() => {
    if (!selectedExistingRunId) {
      return null;
    }
    return (
      knownRunsById[selectedExistingRunId] ??
      availableRuns.find((run) => run.run_id === selectedExistingRunId) ??
      null
    );
  }, [availableRuns, knownRunsById, selectedExistingRunId]);

  const runContextCityNames = useMemo(
    () => extractRunContextCityNames(runContext),
    [runContext],
  );
  const preferredRunCccCity = useMemo(
    () => pickFirstAvailableCity(runContextCityNames, cities),
    [cities, runContextCityNames],
  );
  const selectedCccCityKey = selectedCccCity
    ? normalizeCitySelectionKey(selectedCccCity)
    : "";
  const deferredRunSearchQuery = useDeferredValue(runSearchQuery);

  const railToggleLabel = isControlsCollapsed ? "Show controls" : "Hide controls";

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
    const stored = window.localStorage.getItem(CONTROLS_COLLAPSED_STORAGE_KEY);
    setIsControlsCollapsed(stored === "1");
  }, []);

  useEffect(() => {
    window.localStorage.setItem(
      CONTROLS_COLLAPSED_STORAGE_KEY,
      isControlsCollapsed ? "1" : "0",
    );
  }, [isControlsCollapsed]);

  useEffect(() => {
    const controller = new AbortController();
    let intervalId: number | null = null;
    const isVectorStoreTransitional = ["checking", "stale", "running"].includes(
      vectorStoreStatus?.status ?? "",
    );
    const shouldPollContinuously = isSubmitting || isVectorStoreTransitional;

    async function refreshVectorStoreStatus(): Promise<void> {
      try {
        const status = await fetchVectorStoreStatus({ signal: controller.signal });
        setVectorStoreStatus(status);
        setVectorStoreStatusError(null);
      } catch (error) {
        if (!controller.signal.aborted) {
          setVectorStoreStatusError(
            error instanceof Error
              ? error.message
              : "Could not load vector store status.",
          );
        }
      }
    }

    void refreshVectorStoreStatus();
    if (shouldPollContinuously) {
      intervalId = window.setInterval(() => {
        void refreshVectorStoreStatus();
      }, isSubmitting ? 1000 : 3000);
    }

    return () => {
      controller.abort();
      if (intervalId !== null) {
        window.clearInterval(intervalId);
      }
    };
  }, [isSubmitting, vectorStoreStatus?.status]);

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
    window.localStorage.setItem(LAST_RUN_ID_STORAGE_KEY, trimmedRunId);
    setSelectedExistingRunId(trimmedRunId);
  }, []);

  const refreshRunList = useCallback(
    async (preferredRunId?: string): Promise<RunSummary[] | null> => {
      runListAbortControllerRef.current?.abort();
      const controller = new AbortController();
      runListAbortControllerRef.current = controller;
      setIsLoadingRuns(true);
      setRunsError(null);
      try {
        const payload = await fetchRuns({
          includeAll: devFeatures.showIncompleteRuns,
          search: deferredRunSearchQuery,
          signal: controller.signal,
        });
        if (runListAbortControllerRef.current !== controller) {
          return null;
        }
        setKnownRunsById((current) => {
          const next = { ...current };
          payload.runs.forEach((run) => {
            next[run.run_id] = run;
          });
          return next;
        });
        setAvailableRuns(payload.runs);
        setSelectedExistingRunId((current) => {
          const preferred = (preferredRunId ?? current).trim();
          if (preferred) {
            return preferred;
          }
          if (payload.runs.length > 0) {
            return payload.runs[0].run_id;
          }
          return "";
        });
        return payload.runs;
      } catch (error) {
        if (controller.signal.aborted) {
          return null;
        }
        setRunsError(error instanceof Error ? error.message : "Failed to load runs.");
        return null;
      } finally {
        if (runListAbortControllerRef.current === controller) {
          setIsLoadingRuns(false);
        }
      }
    },
    [deferredRunSearchQuery, devFeatures.showIncompleteRuns],
  );

  async function handleLoadExistingRun(): Promise<void> {
    const trimmed = selectedExistingRunId.trim();
    if (!trimmed || isLoadingSelectedRun) {
      return;
    }
    setIsLoadingSelectedRun(true);
    setRunError(null);
    setActiveTab("document");
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
    if (!hasHydratedFrontendMode) {
      return;
    }
    let cancelled = false;
    const storedRunId = (window.localStorage.getItem(LAST_RUN_ID_STORAGE_KEY) ?? "").trim();
    if (!storedRunId) {
      void refreshRunList();
      return () => {
        cancelled = true;
      };
    }

    const loadInitialRunState = async (): Promise<void> => {
      const runs = await refreshRunList(storedRunId || undefined);
      if (cancelled || !storedRunId) {
        return;
      }
      const canHydrateStoredRun = runs?.some((run) => run.run_id === storedRunId) ?? false;
      if (!canHydrateStoredRun) {
        window.localStorage.removeItem(LAST_RUN_ID_STORAGE_KEY);
        return;
      }

      setSelectedExistingRunId(storedRunId);
      setIsLoadingSelectedRun(true);
      try {
        await hydrateRunById(storedRunId);
      } catch {
        // Ignore stale stored run ids on startup; user can load another run manually.
      } finally {
        if (!cancelled) {
          setIsLoadingSelectedRun(false);
        }
      }
    };

    void loadInitialRunState();
    return () => {
      cancelled = true;
    };
  }, [hasHydratedFrontendMode, refreshRunList, hydrateRunById]);

  useEffect(() => {
    return () => {
      runListAbortControllerRef.current?.abort();
    };
  }, []);

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
    if (cities.length === 0) {
      return;
    }
    if (selectedCccCity && pickFirstAvailableCity([selectedCccCity], cities)) {
      return;
    }

    const storedCity =
      typeof window === "undefined"
        ? null
        : window.sessionStorage.getItem(LAST_CCC_CITY_STORAGE_KEY);
    const fallbackCity =
      pickFirstAvailableCity(storedCity ? [storedCity] : [], cities) ?? cities[0];
    setSelectedCccCity(fallbackCity);
  }, [cities, selectedCccCity]);

  useEffect(() => {
    if (!selectedCccCity || typeof window === "undefined") {
      return;
    }
    window.sessionStorage.setItem(LAST_CCC_CITY_STORAGE_KEY, selectedCccCity);
  }, [selectedCccCity]);

  useEffect(() => {
    if (!runId || !preferredRunCccCity) {
      return;
    }
    const appliedKey = `${runId}:${normalizeCitySelectionKey(preferredRunCccCity)}`;
    if (cccRunDefaultRef.current === appliedKey) {
      return;
    }
    cccRunDefaultRef.current = appliedKey;
    setSelectedCccCity(preferredRunCccCity);
  }, [preferredRunCccCity, runId]);

  useEffect(() => {
    if (!documentReady || !cccTabActive || !selectedCccCity) {
      return;
    }

    if (!selectedCccCityKey) {
      return;
    }
    if (cccDocumentCache[selectedCccCityKey]) {
      setIsLoadingCccDocument(false);
      setCccDocumentError(null);
      return;
    }

    let cancelled = false;
    const controller = new AbortController();
    setIsLoadingCccDocument(true);
    setCccDocumentError(null);

    fetchCityMarkdown(selectedCccCity, { signal: controller.signal })
      .then((payload) => {
        if (cancelled) {
          return;
        }
        const cacheKey = normalizeCitySelectionKey(payload.city_name);
        if (!cacheKey) {
          throw new Error("CCC markdown response did not include a valid city name.");
        }
        setCccDocumentCache((current) => ({
          ...current,
          [cacheKey]: payload,
        }));
      })
      .catch((error) => {
        if (cancelled) {
          return;
        }
        if (error instanceof DOMException && error.name === "AbortError") {
          return;
        }
        setCccDocumentError(
          error instanceof Error ? error.message : "Failed to load CCC markdown.",
        );
      })
      .finally(() => {
        if (!cancelled) {
          setIsLoadingCccDocument(false);
        }
      });

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [
    cccDocumentCache,
    documentReady,
    selectedCccCity,
    selectedCccCityKey,
    cccTabActive,
  ]);

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
          }, RUN_STATUS_POLL_INTERVAL_MS);
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

  // While building, also poll the (partial) artifacts so the live timeline can
  // stream each stage's detail (gap rationale, evidence) as it lands.
  useEffect(() => {
    if (!runId || !statusValue || !["queued", "running"].includes(statusValue)) {
      setLiveArtifacts(null);
      return;
    }
    let cancelled = false;
    let handle: ReturnType<typeof setTimeout> | null = null;
    let controller: AbortController | null = null;

    const pollOnce = (): void => {
      if (cancelled) {
        return;
      }
      controller = new AbortController();
      fetchRunArtifacts(runId, { signal: controller.signal })
        .then((payload) => {
          if (!cancelled) {
            setLiveArtifacts(payload);
          }
        })
        .catch(() => {
          // Partial/early reads can fail transiently; ignore and retry.
        })
        .finally(() => {
          controller = null;
          if (!cancelled) {
            handle = setTimeout(pollOnce, 3000);
          }
        });
    };
    pollOnce();

    return () => {
      cancelled = true;
      if (handle) {
        clearTimeout(handle);
      }
      controller?.abort();
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

  function selectCccCity(city: string): void {
    setSelectedCccCity(city);
    setCccDocumentError(null);
  }

  async function handleBuildDocument(): Promise<void> {
    const trimmed = question.trim();
    if (!trimmed || isSubmitting) {
      return;
    }
    if (["checking", "stale", "running", "failed"].includes(vectorStoreStatus?.status ?? "")) {
      setRunError(
        vectorStoreStatus?.message ||
          "Vector store is not ready. Please retry after the update completes.",
      );
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
    setActiveTab("document");

    try {
      const payload = await startRun({
        question: trimmed,
        query_mode: isDevMode ? "dev" : "standard",
        query_2: query2,
        query_3: query3,
        cities: scopeMode === "all" ? undefined : scopedCities,
        analysis_mode: analysisMode,
        enrichment_enabled: enrichmentEnabled,
        web_research_enabled: enrichmentEnabled && webResearchEnabled,
      });
      setRunResponse(payload);
      setSelectedExistingRunId(payload.run_id);
      window.localStorage.setItem(LAST_RUN_ID_STORAGE_KEY, payload.run_id);
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

  const isTerminal = !!statusValue && TERMINAL_STATUSES.includes(statusValue);
  const isLongWait = !!statusValue && ["queued", "running"].includes(statusValue);
  const runFailureMessage =
    runStatus?.error == null
      ? null
      : devFeatures.showRunDiagnostics
        ? `${runStatus.error.code}: ${runStatus.error.message}`
        : "This run did not complete, so no document is available.";
  const runPartialCoverageMessage =
    runStatus?.status === "completed_with_gaps"
      ? "This answer was returned with partial coverage."
      : null;
  const hasValidScope =
    scopeMode === "all" ||
    (scopeMode === "group"
      ? (selectedGroup?.cities.length ?? 0) > 0
      : selectedCities.length > 0);
  const isVectorStoreUpdating = ["checking", "running"].includes(
    vectorStoreStatus?.status ?? "",
  );
  const isVectorStoreBlocked = ["checking", "stale", "running", "failed"].includes(
    vectorStoreStatus?.status ?? "",
  );
  const showVectorStoreBanner =
    isVectorStoreBlocked || vectorStoreStatusError !== null;
  const vectorStoreBannerText =
    vectorStoreStatus?.message ||
    (isVectorStoreUpdating
      ? "Vector store update in progress. Report generation is paused until the index is ready."
      : vectorStoreStatus?.status === "stale"
        ? "Vector store is stale and needs an update before reports can run."
        : vectorStoreStatus?.status === "failed"
          ? vectorStoreStatus.error || "Vector store update failed. Admin action required."
          : vectorStoreStatusError);
  const vectorStoreBannerTitle = isVectorStoreUpdating
    ? "Preparing vector search"
    : vectorStoreStatus?.status === "stale"
      ? "Manual vector-store update required"
      : vectorStoreStatus?.status === "failed"
        ? "Vector store update failed"
        : "Vector store status needs attention";
  const hasApiKeyIssue =
    /api key|authentication|unauthorized|401|403/i.test(runError ?? "") ||
    /api key|authentication|unauthorized|401|403/i.test(
      runStatus?.error?.message ?? "",
    );
  const hasProviderReadinessIssue =
    /serper|firecrawl|openrouter|web research is unavailable|cannot start because no openrouter api key/i.test(
      runError ?? "",
    );
  const activeCccDocument = selectedCccCityKey
    ? cccDocumentCache[selectedCccCityKey] ?? null
    : null;
  const buildDisabledReason = isSubmitting
    ? "Report generation is already in progress."
    : isVectorStoreBlocked
      ? vectorStoreBannerText || "Vector store is not ready yet."
      : !question.trim()
        ? "Enter a question to generate a report."
        : !hasValidScope
          ? scopeMode === "group"
            ? "Select a predefined city group before generating the report."
            : "Select at least one city before generating the report."
          : null;

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_20%_20%,#f8edd6_0%,#f2f6f6_45%,#eef2ff_100%)] px-4 py-8 md:px-8">
      <div className="mx-auto max-w-[96rem] space-y-6">
        <header className="rounded-xl border border-slate-200 bg-white/80 p-6 shadow-sm backdrop-blur-sm">
          <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
            <div>
              <p className="mb-1 text-xs font-semibold uppercase tracking-[0.18em] text-amber-700">
                Query Engine
              </p>
              <h1 className="text-3xl font-semibold text-slate-900 md:text-4xl">
                Query demand signals from city plans &amp; public commitments.
              </h1>
              <p className="mt-2 max-w-3xl text-sm text-slate-600 md:text-base">
                Ask a question and the engine reads the cities&apos; climate
                contracts, fills data gaps, and builds a sourced report you can
                explore, audit, and chat with.
              </p>
            </div>
            <div className="flex items-center gap-3 self-start md:self-auto">
              <DevModeToggle mode={frontendMode} onModeChange={setFrontendMode} />
              <LogoutButton />
            </div>
          </div>
        </header>

        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={() => setIsControlsCollapsed((current) => !current)}
          aria-label={isControlsCollapsed ? "Show query controls" : "Hide query controls"}
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
            {railToggleLabel}
          </span>
        </Button>

        <main
          className="flex flex-col gap-6 lg:flex-row"
        >
          <div
            className={`overflow-hidden lg:shrink-0 transition-[width,opacity,transform] duration-300 ease-in-out ${
              isControlsCollapsed
                ? "lg:w-0 lg:-translate-x-4 lg:opacity-0 lg:pointer-events-none"
                : "lg:w-[26rem] lg:translate-x-0 lg:opacity-100"
            }`}
          >
            <Card className="h-fit border-slate-300">
              <CardHeader>
                <CardTitle>Query Controls</CardTitle>
                <CardDescription>
                  Set your query, pick the cities, and run.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-5">
                    {showVectorStoreBanner ? (
                      <div
                        className={`flex items-start gap-3 rounded-md border p-3 text-sm ${
                          isVectorStoreUpdating
                            ? "border-sky-200 bg-sky-50 text-sky-900"
                            : "border-amber-200 bg-amber-50 text-amber-900"
                        }`}
                      >
                        {isVectorStoreUpdating ? (
                          <Loader2 className="mt-0.5 h-4 w-4 shrink-0 animate-spin" />
                        ) : (
                          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
                        )}
                        <div className="min-w-0 space-y-1">
                          <p className="font-medium">
                            {vectorStoreBannerTitle}
                          </p>
                          <p>{vectorStoreBannerText}</p>
                        </div>
                      </div>
                    ) : null}

                    <div className="space-y-2">
                      <Label htmlFor="question">
                        What demand signal or target are you looking for?
                      </Label>
                      <Textarea
                        id="question"
                        placeholder="e.g. What are Aachen's rooftop-solar capacity targets and the public commitments behind them? (You can paste a longer, structured prompt too.)"
                        value={question}
                        onChange={(event) => setQuestion(event.target.value)}
                        disabled={isVectorStoreBlocked}
                        className="min-h-32"
                      />
                      <p className="text-xs text-slate-600">
                        This is the primary query against the city climate contracts. Be
                        specific about the metric, target, or commitment you want — you can
                        paste a long, LLM-prepared prompt as well.
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

              {!isDevMode ? (
                <button
                  type="button"
                  onClick={() => setShowAdvanced((v) => !v)}
                  aria-expanded={showAdvanced}
                  className="flex w-full items-center justify-between rounded-md border border-slate-200 bg-slate-50 px-3 py-2.5 text-sm font-medium text-slate-700 transition hover:bg-slate-100"
                >
                  <span className="flex items-center gap-2">
                    <Settings2 className="h-4 w-4 text-slate-400" />
                    Advanced options
                  </span>
                  <ChevronDown
                    className={`h-4 w-4 text-slate-400 transition-transform ${showAdvanced ? "rotate-180" : ""}`}
                  />
                </button>
              ) : null}

              {advancedExpanded ? (
                <div className="space-y-4">
                  <div className="space-y-3 rounded-md border border-slate-200 bg-slate-50 p-3">
                    <div className="space-y-1">
                      <div className="flex items-center justify-between">
                        <Label className="text-sm font-medium text-slate-800">
                          Follow-up retrieval queries
                        </Label>
                        {isDevMode ? <Badge variant="outline">Dev Mode</Badge> : null}
                      </div>
                      <p className="text-xs text-slate-600">
                        Optional extra queries run alongside your main question to pull more
                        evidence. Blank fields are ignored.
                      </p>
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="query-2">Follow-up query 2 (optional)</Label>
                      <Textarea
                        id="query-2"
                        placeholder="e.g. Planned rollout milestones, deadlines, and responsible owners."
                        value={query2}
                        onChange={(event) => setQuery2(event.target.value)}
                        disabled={isVectorStoreBlocked}
                        className="min-h-20"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="query-3">Follow-up query 3 (optional)</Label>
                      <Textarea
                        id="query-3"
                        placeholder="e.g. Tables or numeric references for current values, 2030 targets, and budget commitments."
                        value={query3}
                        onChange={(event) => setQuery3(event.target.value)}
                        disabled={isVectorStoreBlocked}
                        className="min-h-20"
                      />
                    </div>
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
                    : "Answering one city section at a time; similarities at the end."}
                </p>
              </div>

              <div className="space-y-3 rounded-md border border-slate-200 p-3">
                <div className="flex items-center justify-between">
                  <Label>Enrichment layer</Label>
                  <Badge variant={enrichmentEnabled ? "secondary" : "outline"}>
                    {enrichmentEnabled ? "On" : "Off"}
                  </Badge>
                </div>
                <div className="flex items-center justify-between gap-2">
                  <span className="text-sm text-slate-700">Assumptions + web research step</span>
                  <Button
                    type="button"
                    size="sm"
                    variant={enrichmentEnabled ? "default" : "outline"}
                    onClick={() => setEnrichmentEnabled((v) => !v)}
                  >
                    {enrichmentEnabled ? "Disable" : "Enable"}
                  </Button>
                </div>
                <div className="flex items-center justify-between gap-2">
                  <span
                    className={`text-sm ${enrichmentEnabled ? "text-slate-700" : "text-slate-400"}`}
                  >
                    Web research sub-step
                  </span>
                  <Button
                    type="button"
                    size="sm"
                    variant={enrichmentEnabled && webResearchEnabled ? "default" : "outline"}
                    disabled={!enrichmentEnabled}
                    onClick={() => setWebResearchEnabled((v) => !v)}
                  >
                    {webResearchEnabled ? "Disable" : "Enable"}
                  </Button>
                </div>
                <p className="text-xs text-slate-600">
                  {enrichmentEnabled
                    ? webResearchEnabled
                      ? "Gap analysis, web research, and assumption estimates will run after CCC research."
                      : "Gap analysis and assumption estimates will run; live web search is skipped."
                    : "CCC excerpts only — no gap analysis, web research, or assumption estimates."}
                </p>
              </div>
                </div>
              ) : null}

              <Button
                onClick={handleBuildDocument}
                disabled={isSubmitting || isVectorStoreBlocked || !question.trim() || !hasValidScope}
                className="w-full"
              >
                {isSubmitting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
                Run query
              </Button>
              {buildDisabledReason ? (
                <p className="text-xs text-slate-500">{buildDisabledReason}</p>
              ) : null}

              <Separator />

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label htmlFor="existing-run">Open a previous run</Label>
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
                <div className="flex items-start gap-2">
                  <SearchableRunPicker
                    id="existing-run"
                    runs={availableRuns}
                    selectedRun={selectedExistingRunSummary}
                    selectedRunId={selectedExistingRunId}
                    searchQuery={runSearchQuery}
                    onSearchQueryChange={setRunSearchQuery}
                    onSelectRun={setSelectedExistingRunId}
                    formatRunLabel={formatRunOptionLabel}
                    isLoading={isLoadingRuns}
                    popupClassName="w-[calc(100%+5.5rem)]"
                  />
                  <Button
                    type="button"
                    variant="outline"
                    className="h-11 w-20 shrink-0"
                    onClick={() => void handleLoadExistingRun()}
                    disabled={isLoadingSelectedRun || !selectedExistingRunId.trim()}
                  >
                    {isLoadingSelectedRun ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
                    Open
                  </Button>
                </div>
                <p className="text-xs text-slate-500">
                  View a past answer without re-running — search the list by question, date,
                  run ID, or city (minor typos are tolerated).
                </p>
                {runsError ? <p className="text-xs text-red-600">{runsError}</p> : null}
                {devFeatures.showIncompleteRuns ? (
                  <p className="text-xs text-slate-500">
                    Dev mode keeps failed and in-progress runs in the picker.
                  </p>
                ) : null}
              </div>

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
                    <div className="flex items-center gap-2 font-medium">
                      <CircleDashed className="h-4 w-4 animate-spin" />
                      Build in progress
                    </div>
                    <p className="mt-1 text-xs text-amber-800/80">
                      Live stage-by-stage progress is shown in the document panel.
                    </p>
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
                    {runPartialCoverageMessage ? (
                      <p className="text-xs text-amber-700">{runPartialCoverageMessage}</p>
                    ) : null}
                    {runFailureMessage ? <p className="text-xs text-red-700">{runFailureMessage}</p> : null}
                    {hasApiKeyIssue ? (
                      <p className="mt-1 text-xs text-amber-700">
                        API key issue detected. Verify backend OpenRouter credentials and retry the run.
                      </p>
                    ) : null}
                  </div>
                ) : null}
                {runError ? <p className="text-sm text-red-600">{runError}</p> : null}
                {hasProviderReadinessIssue ? (
                  <p className="text-xs text-amber-700">
                    Provider readiness check failed before the run was queued. Verify the
                    configured provider keys or disable the affected feature and retry.
                  </p>
                ) : null}
                {devFeatures.showRunDiagnostics && runId ? (
                  <RunDiagnosticsPanel runId={runId} runStatus={runStatus} />
                ) : null}
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

          <div className="min-w-0 flex-1 space-y-4">
            {documentReady && runId ? (
              <>
                <div className="inline-flex flex-wrap gap-1 rounded-full border border-slate-200 bg-slate-100 p-1">
                  {(
                    [
                      { key: "document", label: "Document", icon: FileText, show: true },
                      { key: "enrichment", label: "Enrichment Process", icon: ScanSearch, show: true },
                      { key: "chat", label: "Chat", icon: MessageSquareText, show: true },
                      { key: "ccc", label: "CCC", icon: BookOpen, show: isDevMode },
                      { key: "assumptions", label: "Assumptions", icon: Sparkles, show: devFeatures.showAssumptionsEntry },
                    ] as { key: TabKey; label: string; icon: typeof FileText; show: boolean }[]
                  )
                    .filter((tab) => tab.show)
                    .map((tab) => {
                      const Icon = tab.icon;
                      return (
                        <button
                          key={tab.key}
                          type="button"
                          onClick={() => setActiveTab(tab.key)}
                          className={`inline-flex items-center gap-1.5 rounded-full px-3.5 py-1.5 text-sm font-medium transition ${
                            activeTab === tab.key
                              ? "bg-white text-slate-900 shadow-sm"
                              : "text-slate-600 hover:text-slate-900"
                          }`}
                        >
                          <Icon className="h-4 w-4" />
                          {tab.label}
                        </button>
                      );
                    })}
                </div>

                {auditOpen ? (
                  <EnrichmentProcessWorkspace runId={runId} />
                ) : chatOpen ? (
                  <ContextChatWorkspace
                    runId={runId}
                    enabled={documentReady}
                    showContextManager={devFeatures.showContextManager}
                    showDevDiagnostics={frontendMode === "dev"}
                    showTokenMetrics={devFeatures.showChatTokenMetrics}
                  />
                ) : cccTabActive ? (
                  <Card className="border-slate-300">
                    <CardContent className="pt-6">
                      <CccDocumentRail
                        cities={cities}
                        selectedCity={selectedCccCity}
                        onSelectCity={selectCccCity}
                        content={activeCccDocument?.content ?? null}
                        sourcePaths={activeCccDocument?.source_paths ?? []}
                        isLoadingCities={isLoadingCities}
                        citiesError={citiesError}
                        isLoadingContent={isLoadingCccDocument}
                        contentError={cccDocumentError}
                      />
                    </CardContent>
                  </Card>
                ) : assumptionsOpen ? (
                  <AssumptionsWorkspace runId={runId} enabled={documentReady} />
                ) : (
                  <Card className="border-slate-300">
                    <CardContent className="space-y-3 pt-6">
                      <DocumentExportControls
                        runId={runId}
                        content={runOutput.content}
                        showWriterContextExport={devFeatures.showRunDiagnostics}
                      />
                      <article className="document-markdown rounded-md border border-slate-200 bg-white p-5 shadow-inner">
                        <MarkdownWithReferences content={runOutput.content} runId={runId} />
                      </article>
                      {runContext ? (
                        <p className="text-xs text-slate-500">
                          Context bundle loaded from: {runContext.context_bundle_path}
                        </p>
                      ) : null}
                    </CardContent>
                  </Card>
                )}
              </>
            ) : isLongWait ? (
              <LiveBuildTimeline
                steps={runStatus?.steps}
                runStatus={statusValue}
                artifacts={liveArtifacts}
              />
            ) : (
              <Card className="border-slate-300">
                <CardHeader className="pb-4">
                  <CardTitle>Report</CardTitle>
                  <CardDescription>
                    Your answer, written as a sourced report. Audit the data behind it or chat to dig deeper.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="rounded-md border border-dashed border-slate-300 bg-white p-8 text-center text-slate-600">
                    <p className="text-base font-medium">Your report will appear here.</p>
                    <p className="mt-1 text-sm">
                      {isControlsCollapsed
                        ? "Open Query Controls to run a query."
                        : "Write a query on the left and hit Run query to start."}
                    </p>
                  </div>
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
