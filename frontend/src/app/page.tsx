"use client";

import {
  type CSSProperties,
  type PointerEvent as ReactPointerEvent,
  useCallback,
  useDeferredValue,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
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
import { CccDocumentRail } from "@/components/ccc-document-rail";
import { PipelineProgress } from "@/components/pipeline-progress";
import { ContextChatWorkspace } from "@/components/context-chat/context-chat-workspace";
import { DevModeToggle } from "@/components/dev-mode-toggle";
import { DevToolsPanel } from "@/components/dev-tools-panel";
import { DocumentExportControls } from "@/components/document-export-controls";
import { MarkdownWithReferences } from "@/components/markdown-with-references";
import { SearchableCityPicker } from "@/components/searchable-city-picker";
import { SearchableRunPicker } from "@/components/searchable-run-picker";
import { WriterDocumentRail } from "@/components/writer-document-rail";
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
import { filterImmediateRunMatches } from "@/lib/run-picker-search";
import { formatCityLabel } from "@/lib/utils";
import {
  CityGroup,
  CityMarkdownResponse,
  CreateRunResponse,
  RunContextResponse,
  RunOutputResponse,
  RunSummary,
  RunStatus,
  RunStatusResponse,
  fetchCities,
  fetchCityMarkdown,
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
const RUN_STATUS_POLL_INTERVAL_MS = 2500;

type CityScopeMode = "all" | "group" | "manual";
type AnalysisMode = "aggregate" | "city_by_city";
type WorkspaceRailMode = "controls" | "document" | "ccc";
const LAST_RUN_ID_STORAGE_KEY = "last_run_id";
const LAST_CCC_CITY_STORAGE_KEY = "last_ccc_city";
const CONTROLS_COLLAPSED_STORAGE_KEY = "build_controls_collapsed";
const DEFAULT_WRITER_RAIL_WIDTH_PX = 416;
const MIN_WRITER_RAIL_WIDTH_PX = 320;
const MAX_WRITER_RAIL_WIDTH_PX = 760;
const MIN_WORKSPACE_CONTENT_WIDTH_PX = 480;

function clampWriterRailWidth(width: number, viewportWidth: number): number {
  const maxWidth = Math.min(
    MAX_WRITER_RAIL_WIDTH_PX,
    Math.max(
      MIN_WRITER_RAIL_WIDTH_PX,
      viewportWidth - MIN_WORKSPACE_CONTENT_WIDTH_PX - 96,
    ),
  );
  return Math.min(Math.max(width, MIN_WRITER_RAIL_WIDTH_PX), maxWidth);
}

function formatRunOptionLabel(run: RunSummary): string {
  const compactQuestion = run.question.replace(/\s+/g, " ").trim();
  const preview =
    compactQuestion.length > 56 ? `${compactQuestion.slice(0, 53)}...` : compactQuestion;
  const pickerLabel = run.picker_timestamp || run.run_id;
  return `${pickerLabel} | ${preview || "No question"}`;
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
  const markdownBundle = runContext.context_bundle.markdown;
  if (!isObjectRecord(markdownBundle)) {
    return [];
  }

  const orderedCandidates = [
    ...readStringArray(markdownBundle.selected_city_names),
    ...readStringArray(markdownBundle.inspected_city_names),
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
  const [webResearchEnabled, setWebResearchEnabled] = useState(true);
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
  const [knownRunsById, setKnownRunsById] = useState<Record<string, RunSummary>>({});
  const [selectedExistingRunId, setSelectedExistingRunId] = useState("");
  const [runSearchQuery, setRunSearchQuery] = useState("");
  const [isLoadingRuns, setIsLoadingRuns] = useState(false);
  const [runsError, setRunsError] = useState<string | null>(null);
  const [isLoadingSelectedRun, setIsLoadingSelectedRun] = useState(false);

  const [chatOpen, setChatOpen] = useState(false);
  const [assumptionsOpen, setAssumptionsOpen] = useState(false);
  const [isControlsCollapsed, setIsControlsCollapsed] = useState(false);
  const [workspaceRailMode, setWorkspaceRailMode] =
    useState<WorkspaceRailMode>("controls");
  const [selectedCccCity, setSelectedCccCity] = useState<string | null>(null);
  const [cccDocumentCache, setCccDocumentCache] = useState<
    Record<string, CityMarkdownResponse>
  >({});
  const [isLoadingCccDocument, setIsLoadingCccDocument] = useState(false);
  const [cccDocumentError, setCccDocumentError] = useState<string | null>(null);
  const [writerRailWidth, setWriterRailWidth] = useState(DEFAULT_WRITER_RAIL_WIDTH_PX);
  const [isWriterRailResizing, setIsWriterRailResizing] = useState(false);
  const [frontendMode, setFrontendMode] = useState<FrontendMode>(getDefaultFrontendMode());
  const [hasHydratedFrontendMode, setHasHydratedFrontendMode] = useState(false);
  const writerRailResizeRef = useRef<{ startX: number; startWidth: number } | null>(
    null,
  );
  const cccRunDefaultRef = useRef<string | null>(null);
  const runListAbortControllerRef = useRef<AbortController | null>(null);

  const runId = runResponse?.run_id ?? null;
  const statusValue = runStatus?.status ?? runResponse?.status ?? null;
  const canFetchArtifacts = statusValue === "completed" || statusValue === "completed_with_gaps";
  const documentReady = !!runOutput?.content && canFetchArtifacts;
  const devFeatures = useMemo(() => getDevFeatureFlags(frontendMode), [frontendMode]);
  const showDirectQueryControls = frontendMode === "dev";
  const workspaceUsesDocumentRail = documentReady && (chatOpen || assumptionsOpen);
  const isWriterRailResizable =
    workspaceUsesDocumentRail &&
    workspaceRailMode !== "controls" &&
    !isControlsCollapsed;
  const writerRailStyle = useMemo<CSSProperties | undefined>(() => {
    if (!isWriterRailResizable) {
      return undefined;
    }
    return {
      "--workspace-rail-width": `${writerRailWidth}px`,
    } as CSSProperties;
  }, [isWriterRailResizable, writerRailWidth]);

  const activeRunSummary = useMemo(() => {
    if (!runId) {
      return null;
    }
    return knownRunsById[runId] ?? availableRuns.find((run) => run.run_id === runId) ?? null;
  }, [availableRuns, knownRunsById, runId]);

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

  const activeRunQuestion = useMemo(() => {
    const summaryQuestion = activeRunSummary?.question?.trim();
    if (summaryQuestion) {
      return summaryQuestion;
    }
    const draftQuestion = question.trim();
    if (draftQuestion && runId === runResponse?.run_id) {
      return draftQuestion;
    }
    return null;
  }, [activeRunSummary, question, runId, runResponse?.run_id]);

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
  const visibleRunOptions = useMemo(
    () => filterImmediateRunMatches(availableRuns, runSearchQuery),
    [availableRuns, runSearchQuery],
  );

  const workspaceRailTitle =
    workspaceUsesDocumentRail && workspaceRailMode === "document"
      ? "Writer Document"
      : workspaceUsesDocumentRail && workspaceRailMode === "ccc"
        ? "CCC Source"
        : "Build Controls";
  const workspaceRailDescription =
    workspaceUsesDocumentRail && workspaceRailMode === "document"
      ? "Keep the generated report open while you chat or review assumptions."
      : workspaceUsesDocumentRail && workspaceRailMode === "ccc"
        ? "Browse raw Climate City Contracts without dropping the active workspace."
        : "Select scope, trigger a build, or load a previous answer.";
  const railToggleLabel = workspaceUsesDocumentRail
    ? isControlsCollapsed
      ? "Show Rail"
      : "Hide Rail"
    : isControlsCollapsed
      ? "Show Controls"
      : "Hide Controls";

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
    setWriterRailWidth(
      clampWriterRailWidth(DEFAULT_WRITER_RAIL_WIDTH_PX, window.innerWidth),
    );
  }, []);

  useEffect(() => {
    const handleResize = (): void => {
      setWriterRailWidth((current) =>
        clampWriterRailWidth(current, window.innerWidth),
      );
    };
    window.addEventListener("resize", handleResize);
    return () => {
      window.removeEventListener("resize", handleResize);
    };
  }, []);

  useEffect(() => {
    if (!isWriterRailResizing) {
      return;
    }

    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";

    const handlePointerMove = (event: PointerEvent): void => {
      const origin = writerRailResizeRef.current;
      if (!origin) {
        return;
      }
      const nextWidth = origin.startWidth + event.clientX - origin.startX;
      setWriterRailWidth(
        clampWriterRailWidth(nextWidth, window.innerWidth),
      );
    };

    const stopResizing = (): void => {
      writerRailResizeRef.current = null;
      setIsWriterRailResizing(false);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };

    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", stopResizing);
    window.addEventListener("pointercancel", stopResizing);
    return () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", stopResizing);
      window.removeEventListener("pointercancel", stopResizing);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
  }, [isWriterRailResizing]);

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
    async (preferredRunId?: string): Promise<void> => {
      runListAbortControllerRef.current?.abort();
      const controller = new AbortController();
      runListAbortControllerRef.current = controller;
      setIsLoadingRuns(true);
      setRunsError(null);
      try {
        const payload = await fetchRuns({
          search: deferredRunSearchQuery,
          signal: controller.signal,
        });
        if (runListAbortControllerRef.current !== controller) {
          return;
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
      } catch (error) {
        if (controller.signal.aborted) {
          return;
        }
        setRunsError(error instanceof Error ? error.message : "Failed to load runs.");
      } finally {
        if (runListAbortControllerRef.current === controller) {
          setIsLoadingRuns(false);
        }
      }
    },
    [deferredRunSearchQuery],
  );

  async function handleLoadExistingRun(): Promise<void> {
    const trimmed = selectedExistingRunId.trim();
    if (!trimmed || isLoadingSelectedRun) {
      return;
    }
    setIsLoadingSelectedRun(true);
    setRunError(null);
    openDocumentWorkspace();
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
    return () => {
      runListAbortControllerRef.current?.abort();
    };
  }, []);

  useEffect(() => {
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
    if (
      !documentReady ||
      !workspaceUsesDocumentRail ||
      workspaceRailMode !== "ccc" ||
      !selectedCccCity
    ) {
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
    workspaceRailMode,
    workspaceUsesDocumentRail,
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

  function openDocumentWorkspace(): void {
    setChatOpen(false);
    setAssumptionsOpen(false);
    setWorkspaceRailMode("controls");
    setIsWriterRailResizing(false);
    writerRailResizeRef.current = null;
  }

  function openChatWorkspace(): void {
    setAssumptionsOpen(false);
    setChatOpen(true);
    setWorkspaceRailMode("document");
    setIsControlsCollapsed(false);
  }

  function openAssumptionsWorkspace(): void {
    setChatOpen(false);
    setAssumptionsOpen(true);
    setWorkspaceRailMode("document");
    setIsControlsCollapsed(false);
  }

  function startWriterRailResize(
    event: ReactPointerEvent<HTMLButtonElement>,
  ): void {
    if (!isWriterRailResizable) {
      return;
    }
    event.preventDefault();
    writerRailResizeRef.current = {
      startX: event.clientX,
      startWidth: writerRailWidth,
    };
    setIsWriterRailResizing(true);
  }

  function resetWriterRailWidth(): void {
    setWriterRailWidth(
      clampWriterRailWidth(DEFAULT_WRITER_RAIL_WIDTH_PX, window.innerWidth),
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
    openDocumentWorkspace();

    try {
      const payload = await startRun({
        question: trimmed,
        query_mode: showDirectQueryControls ? "dev" : "standard",
        query_2: showDirectQueryControls ? query2 : undefined,
        query_3: showDirectQueryControls ? query3 : undefined,
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
  const activeCccDocument = selectedCccCityKey
    ? cccDocumentCache[selectedCccCityKey] ?? null
    : null;

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_20%_20%,#f8edd6_0%,#f2f6f6_45%,#eef2ff_100%)] px-4 py-8 md:px-8">
      <div className="mx-auto max-w-[96rem] space-y-6">
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
                This flow is document-first. You submit a build run, wait for
                completion, review the generated document, then switch into
                context chat workspace.
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
          aria-label={isControlsCollapsed ? `Show ${workspaceRailTitle.toLowerCase()}` : `Hide ${workspaceRailTitle.toLowerCase()}`}
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
          className={`flex flex-col gap-6 lg:flex-row ${
            isWriterRailResizing ? "lg:select-none" : ""
          }`}
        >
          <div
            style={writerRailStyle}
            className={`overflow-hidden lg:shrink-0 ${
              isWriterRailResizing
                ? "lg:transition-none"
                : "transition-[width,opacity,transform] duration-300 ease-in-out"
            } ${
              isControlsCollapsed
                ? "lg:w-0 lg:-translate-x-4 lg:opacity-0 lg:pointer-events-none"
                : isWriterRailResizable
                  ? "lg:w-[var(--workspace-rail-width)] lg:translate-x-0 lg:opacity-100"
                  : "lg:w-[26rem] lg:translate-x-0 lg:opacity-100"
            }`}
          >
            <Card className="h-fit border-slate-300">
              <CardHeader>
                <div className="flex flex-col gap-3">
                  <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                    <div>
                      <CardTitle>{workspaceRailTitle}</CardTitle>
                      <CardDescription>{workspaceRailDescription}</CardDescription>
                    </div>
                    {workspaceUsesDocumentRail ? (
                      <div className="inline-flex rounded-full border border-slate-200 bg-slate-100 p-1">
                        <button
                          type="button"
                          onClick={() => setWorkspaceRailMode("document")}
                          className={`rounded-full px-3 py-1 text-xs font-medium transition ${
                            workspaceRailMode === "document"
                              ? "bg-white text-slate-900 shadow-sm"
                              : "text-slate-600 hover:text-slate-900"
                          }`}
                        >
                          Writer Doc
                        </button>
                        <button
                          type="button"
                          onClick={() => setWorkspaceRailMode("ccc")}
                          className={`rounded-full px-3 py-1 text-xs font-medium transition ${
                            workspaceRailMode === "ccc"
                              ? "bg-white text-slate-900 shadow-sm"
                              : "text-slate-600 hover:text-slate-900"
                          }`}
                        >
                          CCC
                        </button>
                        <button
                          type="button"
                          onClick={() => setWorkspaceRailMode("controls")}
                          className={`rounded-full px-3 py-1 text-xs font-medium transition ${
                            workspaceRailMode === "controls"
                              ? "bg-white text-slate-900 shadow-sm"
                              : "text-slate-600 hover:text-slate-900"
                          }`}
                        >
                          Controls
                        </button>
                      </div>
                    ) : null}
                  </div>
                  {workspaceUsesDocumentRail && workspaceRailMode !== "ccc" ? (
                    <p className="text-xs text-slate-500">
                      {workspaceRailMode === "document"
                        ? "Switch the rail without dropping the chat session, then drag the divider to resize the writer view."
                        : "Switch the rail between the generated answer, source CCCs, and build inputs without dropping the chat session."}
                    </p>
                  ) : null}
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                {workspaceUsesDocumentRail && workspaceRailMode === "document" && runOutput && runId ? (
                  <WriterDocumentRail
                    runId={runId}
                    content={runOutput.content}
                    question={activeRunQuestion}
                    statusLabel={statusValue}
                    onOpenFullDocument={openDocumentWorkspace}
                  />
                ) : workspaceUsesDocumentRail && workspaceRailMode === "ccc" ? (
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
                ) : (
                  <div className="space-y-5">
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

              {showDirectQueryControls ? (
                <div className="space-y-3 rounded-md border border-slate-200 bg-slate-50 p-3">
                  <div className="space-y-1">
                    <div className="flex items-center justify-between">
                      <Label className="text-sm font-medium text-slate-800">
                        Direct retrieval queries
                      </Label>
                      <Badge variant="outline">Dev Mode</Badge>
                    </div>
                    <p className="text-xs text-slate-600">
                      Optional inputs for the retriever. Leave either field blank to ignore it.
                      In dev mode these can be any useful retrieval phrasings, not just keyword or
                      metrics-only queries.
                    </p>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="query-2">Retrieval query 2 (optional)</Label>
                    <Textarea
                      id="query-2"
                      placeholder="Example: a narrower or complementary phrasing of the main question"
                      value={query2}
                      onChange={(event) => setQuery2(event.target.value)}
                      className="min-h-20"
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="query-3">Retrieval query 3 (optional)</Label>
                    <Textarea
                      id="query-3"
                      placeholder="Example: another retrieval angle you want to test directly"
                      value={query3}
                      onChange={(event) => setQuery3(event.target.value)}
                      className="min-h-20"
                    />
                  </div>
                </div>
              ) : null}

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
                    Load
                  </Button>
                </div>
                <p className="text-xs text-slate-500">
                  {runSearchQuery.trim()
                    ? `${visibleRunOptions.length} matching runs.`
                    : `${availableRuns.length} runs discovered in backend storage.`}
                </p>
                {selectedExistingRunSummary ? (
                  <p className="text-xs text-slate-600">
                    Selected run:{" "}
                    <span className="font-medium text-slate-800">
                      {formatRunOptionLabel(selectedExistingRunSummary)}
                    </span>
                    {runSearchQuery.trim() &&
                    !visibleRunOptions.some(
                      (run) => run.run_id === selectedExistingRunSummary.run_id,
                    )
                      ? " (kept selected while search is filtered)"
                      : ""}
                  </p>
                ) : null}
                {runsError ? <p className="text-xs text-red-600">{runsError}</p> : null}
                <p className="text-xs text-slate-500">
                  Open the list to search by question, date, run ID, or city. Minor city typos
                  are tolerated.
                </p>
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
                  <div className="space-y-3">
                    <div className="rounded-md border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
                      <div className="mb-2 flex items-center gap-2 font-medium">
                        <CircleDashed className="h-4 w-4 animate-spin" />
                        Build in progress
                      </div>
                      <p>Leave this page open. Document generation may take several minutes for broad questions.</p>
                    </div>
                    {devFeatures.showPipelineProgress && runStatus?.steps ? (
                      <PipelineProgress steps={runStatus.steps} compact />
                    ) : null}
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
                {devFeatures.showPipelineProgress && isTerminal && runStatus?.steps ? (
                  <PipelineProgress steps={runStatus.steps} compact />
                ) : null}
              </div>

              {devFeatures.showRunId || devFeatures.showApiKeyControls ? (
                <>
                  <Separator />
                  <DevToolsPanel apiKeyIssue={hasApiKeyIssue} runId={runId} />
                </>
              ) : null}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {isWriterRailResizable ? (
            <div className="hidden lg:flex lg:w-4 lg:shrink-0 lg:items-stretch lg:justify-center">
              <button
                type="button"
                aria-label="Resize writer document panel"
                onPointerDown={startWriterRailResize}
                onDoubleClick={resetWriterRailWidth}
                className="group flex h-full w-4 cursor-col-resize items-center justify-center bg-transparent"
              >
                <span
                  className={`h-full w-px rounded-full bg-slate-300 transition-colors ${
                    isWriterRailResizing
                      ? "bg-amber-500"
                      : "group-hover:bg-slate-400"
                  }`}
                />
              </button>
            </div>
          ) : null}

          <div className="min-w-0 flex-1">
            {devFeatures.showAssumptionsEntry && assumptionsOpen && documentReady && runId ? (
              <AssumptionsWorkspace
                runId={runId}
                enabled={documentReady}
                onClose={openDocumentWorkspace}
              />
            ) : chatOpen && documentReady && runId ? (
              <ContextChatWorkspace
                runId={runId}
                enabled={documentReady}
                onClose={openDocumentWorkspace}
                showContextManager={devFeatures.showContextManager}
                showDevDiagnostics={frontendMode === "dev"}
                showTokenMetrics={devFeatures.showChatTokenMetrics}
              />
            ) : (
              <Card className="border-slate-300">
                <CardHeader className="pb-4">
                  <div>
                    <div>
                      <CardTitle>Generated Document</CardTitle>
                      <CardDescription>
                        The main answer is rendered as a report. Context chat keeps this report docked in the workspace rail.
                      </CardDescription>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  {documentReady ? (
                    <>
                      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                        {runId ? (
                          <DocumentExportControls
                            runId={runId}
                            content={runOutput.content}
                          />
                        ) : null}
                        <div className="flex flex-wrap gap-2">
                          {devFeatures.showAssumptionsEntry ? (
                            <Button
                              type="button"
                              variant="outline"
                              onClick={openAssumptionsWorkspace}
                              disabled={!runId}
                            >
                              <Sparkles className="h-4 w-4" />
                              Assumptions Review
                            </Button>
                          ) : null}
                          <Button
                            type="button"
                            onClick={openChatWorkspace}
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
                      {devFeatures.showPipelineProgress && runStatus?.steps ? (
                        <PipelineProgress steps={runStatus.steps} />
                      ) : (
                        <div className="space-y-2">
                          <div className="h-2 animate-pulse rounded bg-slate-200" />
                          <div className="h-2 w-11/12 animate-pulse rounded bg-slate-200" />
                          <div className="h-2 w-10/12 animate-pulse rounded bg-slate-200" />
                          <div className="h-2 w-8/12 animate-pulse rounded bg-slate-200" />
                        </div>
                      )}
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
