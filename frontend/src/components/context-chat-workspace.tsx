"use client";

import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowLeft,
  Loader2,
  SendHorizonal,
  Sparkles,
} from "lucide-react";

import {
  fetchCities,
  ChatContextSummary,
  ChatJobHandle,
  ChatMessage,
  ChatRoutingMetadata,
  ChatSessionResponse,
  ChatSessionContextsResponse,
  createChatSession,
  fetchChatContextCatalog,
  fetchChatJobStatus,
  fetchChatSession,
  fetchChatSessionContexts,
  listChatSessions,
  sendChatMessage,
  updateChatSessionContexts,
} from "@/lib/api";
import { MarkdownWithReferences } from "@/components/markdown-with-references";
import { SearchableCityPicker } from "@/components/searchable-city-picker";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn, formatCityLabel } from "@/lib/utils";

interface ContextChatWorkspaceProps {
  runId: string;
  enabled: boolean;
  onClose: () => void;
  showContextManager: boolean;
  showDevDiagnostics: boolean;
  showTokenMetrics: boolean;
}

const DEFAULT_CONTEXT_TOKEN_CAP = 250000;
const CHAT_JOB_POLL_INTERVAL_MS = 2000;
const CHAT_SEND_RECOVERY_POLL_INTERVAL_MS = 2000;
const CHAT_SEND_RECOVERY_TIMEOUT_MS = 180000;

function isRequestTimeoutError(error: unknown): boolean {
  return error instanceof Error && /^Request timeout after \d+s\.$/.test(error.message);
}

function formatAdditionalResearchLabel(city: string | null | undefined): string {
  const cityLabel = formatCityLabel(city ?? "");
  return cityLabel ? `Additional ${cityLabel} research` : "Additional city research";
}

function getPromptContextTokens(value: {
  total_tokens: number;
  prompt_context_tokens?: number | null;
}): number {
  return typeof value.prompt_context_tokens === "number"
    ? value.prompt_context_tokens
    : value.total_tokens;
}

function describeRouting(
  routing: ChatRoutingMetadata,
): { label: string; className: string } {
  switch (routing.action) {
    case "search_single_city":
      return {
        label: formatAdditionalResearchLabel(routing.target_city),
        className: "border-sky-200 bg-sky-50 text-sky-900",
      };
    case "needs_city_clarification":
      return {
        label: "Choose one city",
        className: "border-amber-200 bg-amber-50 text-amber-900",
      };
    case "out_of_scope":
      return {
        label: "Outside current scope",
        className: "border-slate-300 bg-slate-100 text-slate-800",
      };
    default:
      return {
        label: "Answered from saved context",
        className: "border-teal-200 bg-teal-50 text-teal-900",
      };
  }
}

export function ContextChatWorkspace({
  runId,
  enabled,
  onClose,
  showContextManager,
  showDevDiagnostics,
  showTokenMetrics,
}: ContextChatWorkspaceProps) {
  const messageScrollAreaRef = useRef<HTMLDivElement | null>(null);
  const handledClarificationKeyRef = useRef<string | null>(null);
  const sendLockRef = useRef(false);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sessionContexts, setSessionContexts] =
    useState<ChatSessionContextsResponse | null>(null);
  const [pendingJob, setPendingJob] = useState<ChatJobHandle | null>(null);
  const [pendingJobRouting, setPendingJobRouting] = useState<ChatRoutingMetadata | null>(null);

  const [inputValue, setInputValue] = useState("");
  const [pendingPrompt, setPendingPrompt] = useState<string | null>(null);
  const [pendingClarificationCity, setPendingClarificationCity] = useState<string | null>(null);
  const [pendingClarificationQuestion, setPendingClarificationQuestion] = useState<string | null>(
    null,
  );
  const [isBootstrapping, setIsBootstrapping] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [isRecoveringSend, setIsRecoveringSend] = useState(false);
  const [isLoadingContexts, setIsLoadingContexts] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [recoveryMessage, setRecoveryMessage] = useState<string | null>(null);
  const [recoveryPollAttempt, setRecoveryPollAttempt] = useState(0);

  const [isContextManagerOpen, setIsContextManagerOpen] = useState(false);
  const [isLoadingCatalog, setIsLoadingCatalog] = useState(false);
  const [isSavingContexts, setIsSavingContexts] = useState(false);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [contextCatalog, setContextCatalog] = useState<ChatContextSummary[]>([]);
  const [managerSelection, setManagerSelection] = useState<string[]>([]);
  const [isCityClarificationOpen, setIsCityClarificationOpen] = useState(false);
  const [clarificationQuestion, setClarificationQuestion] = useState<string | null>(null);
  const [clarificationCities, setClarificationCities] = useState<string[]>([]);
  const [selectedClarificationCity, setSelectedClarificationCity] = useState<string | null>(null);
  const [isLoadingClarificationCities, setIsLoadingClarificationCities] = useState(false);
  const [clarificationError, setClarificationError] = useState<string | null>(null);

  const canChat = enabled && !!runId;

  useEffect(() => {
    setConversationId(null);
    setMessages([]);
    setSessionContexts(null);
    setPendingJob(null);
    setPendingJobRouting(null);
    setInputValue("");
    setPendingPrompt(null);
    setPendingClarificationCity(null);
    setPendingClarificationQuestion(null);
    setErrorMessage(null);
    setRecoveryMessage(null);
    setRecoveryPollAttempt(0);
    setContextCatalog([]);
    setManagerSelection([]);
    setCatalogError(null);
    setIsCityClarificationOpen(false);
    setClarificationQuestion(null);
    setClarificationCities([]);
    setSelectedClarificationCity(null);
    setIsLoadingClarificationCities(false);
    setClarificationError(null);
    setIsRecoveringSend(false);
    handledClarificationKeyRef.current = null;
    sendLockRef.current = false;
  }, [runId]);

  function applySessionState(session: ChatSessionResponse): void {
    setMessages(session.messages);
    setPendingJob(session.pending_job ?? null);
    setPendingJobRouting(null);
  }

  async function loadSessionContexts(
    activeRunId: string,
    sessionId: string,
  ): Promise<void> {
    setIsLoadingContexts(true);
    try {
      const payload = await fetchChatSessionContexts(activeRunId, sessionId);
      setSessionContexts(payload);
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : "Failed to load session contexts.",
      );
    } finally {
      setIsLoadingContexts(false);
    }
  }

  async function recoverTimedOutSend(options: {
    activeRunId: string;
    sessionId: string;
    baselineMessageCount: number;
  }): Promise<boolean> {
    setIsRecoveringSend(true);
    setRecoveryMessage("Polling backend session state...");
    setRecoveryPollAttempt(0);
    const deadline = Date.now() + CHAT_SEND_RECOVERY_TIMEOUT_MS;
    let attempt = 0;
    try {
      while (Date.now() < deadline) {
        attempt += 1;
        setRecoveryPollAttempt(attempt);
        try {
          const sessionPayload = await fetchChatSession(options.activeRunId, options.sessionId);
          if (
            sessionPayload.pending_job ||
            sessionPayload.messages.length > options.baselineMessageCount
          ) {
            const contextsPayload = await fetchChatSessionContexts(
              options.activeRunId,
              options.sessionId,
            );
            applySessionState(sessionPayload);
            setSessionContexts(contextsPayload);
            setErrorMessage(null);
            return true;
          }
        } catch {
          // Keep polling while the original request is still settling on the backend.
        }
        await new Promise<void>((resolve) => {
          window.setTimeout(resolve, CHAT_SEND_RECOVERY_POLL_INTERVAL_MS);
        });
      }
      return false;
    } finally {
      setIsRecoveringSend(false);
      setRecoveryMessage(null);
      setRecoveryPollAttempt(0);
    }
  }

  useEffect(() => {
    if (!canChat || !runId) {
      return;
    }
    const activeRunId = runId;
    let cancelled = false;

    async function bootstrap(): Promise<void> {
      setIsBootstrapping(true);
      setErrorMessage(null);
      try {
        const existing = await listChatSessions(activeRunId);
        if (cancelled) {
          return;
        }
        let sessionId: string | null = null;
        if (existing.conversations.length > 0) {
          // Fetch all sessions to sort by updated_at (most recent first)
          // Use allSettled to handle partial failures (e.g., corrupted session files)
          const results = await Promise.allSettled(
            existing.conversations.map((id) => fetchChatSession(activeRunId, id)),
          );
          if (cancelled) {
            return;
          }
          // Filter to only successful fetches and extract the session data
          const sessionsWithMetadata = results
            .filter(
              (result): result is PromiseFulfilledResult<Awaited<ReturnType<typeof fetchChatSession>>> =>
                result.status === "fulfilled",
            )
            .map((result) => result.value);
          if (sessionsWithMetadata.length > 0) {
            // Sort by updated_at descending (most recent first)
            sessionsWithMetadata.sort((a, b) =>
              b.updated_at.localeCompare(a.updated_at),
            );
            const mostRecent = sessionsWithMetadata[0];
            sessionId = mostRecent.conversation_id;
            applySessionState(mostRecent);
          }
          // If all sessions failed to load, fall through to create a new one
        }
        if (!sessionId) {
          const created = await createChatSession(activeRunId);
          if (cancelled) {
            return;
          }
          sessionId = created.conversation_id;
          applySessionState(created);
        }
        setConversationId(sessionId);
        if (sessionId) {
          await loadSessionContexts(activeRunId, sessionId);
        }
      } catch (error) {
        if (cancelled) {
          return;
        }
        setErrorMessage(
          error instanceof Error ? error.message : "Failed to open chat session.",
        );
      } finally {
        if (!cancelled) {
          setIsBootstrapping(false);
        }
      }
    }

    void bootstrap();
    return () => {
      cancelled = true;
    };
  }, [canChat, runId]);

  const sortedMessages = useMemo(
    () =>
      [...messages].sort((a, b) => a.created_at.localeCompare(b.created_at)),
    [messages],
  );

  const scrollMessagesToBottom = useCallback((): void => {
    const root = messageScrollAreaRef.current;
    if (!root) {
      return;
    }
    const viewport = root.querySelector<HTMLDivElement>(
      "[data-radix-scroll-area-viewport]",
    );
    if (!viewport) {
      return;
    }
    viewport.scrollTop = viewport.scrollHeight;
  }, []);

  useEffect(() => {
    const handle = window.requestAnimationFrame(() => {
      scrollMessagesToBottom();
    });
    return () => window.cancelAnimationFrame(handle);
  }, [scrollMessagesToBottom, sortedMessages.length, isSending, pendingJob, pendingPrompt]);

  const contextById = useMemo(() => {
    const mapping = new Map<string, ChatContextSummary>();
    contextCatalog.forEach((context) => {
      mapping.set(context.run_id, context);
    });
    return mapping;
  }, [contextCatalog]);
  const pendingJobId = pendingJob?.job_id ?? null;

  const managerTokenCap = sessionContexts?.token_cap ?? DEFAULT_CONTEXT_TOKEN_CAP;
  const selectedContextTokens = useMemo(
    () =>
      managerSelection.reduce((sum, runIdValue) => {
        const context = contextById.get(runIdValue);
        return sum + (context ? getPromptContextTokens(context) : 0);
      }, 0),
    [contextById, managerSelection],
  );
  const selectionExceedsDirectCap = selectedContextTokens > managerTokenCap;

  useEffect(() => {
    const latestMessage = sortedMessages.at(-1);
    const pendingQuestion = latestMessage?.routing?.pending_user_message?.trim();
    if (
      latestMessage?.role !== "assistant" ||
      latestMessage.routing?.action !== "needs_city_clarification" ||
      !pendingQuestion
    ) {
      return;
    }
    const clarificationKey = `${latestMessage.created_at}:${pendingQuestion}`;
    if (handledClarificationKeyRef.current === clarificationKey) {
      return;
    }
    handledClarificationKeyRef.current = clarificationKey;
    setClarificationQuestion(pendingQuestion);
    setSelectedClarificationCity(null);
    setClarificationError(null);
    setIsCityClarificationOpen(true);
  }, [sortedMessages]);

  useEffect(() => {
    if (!isCityClarificationOpen || clarificationCities.length > 0) {
      return;
    }
    let cancelled = false;
    setIsLoadingClarificationCities(true);
    setClarificationError(null);
    fetchCities()
      .then((payload) => {
        if (cancelled) {
          return;
        }
        setClarificationCities(payload.cities);
      })
      .catch((error) => {
        if (cancelled) {
          return;
        }
        setClarificationError(
          error instanceof Error ? error.message : "Failed to load city list.",
        );
      })
      .finally(() => {
        if (!cancelled) {
          setIsLoadingClarificationCities(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [clarificationCities.length, isCityClarificationOpen]);

  useEffect(() => {
    if (!runId || !conversationId || !pendingJobId) {
      return;
    }
    let cancelled = false;
    let timeoutHandle: number | null = null;

    const poll = async (): Promise<void> => {
      try {
        const jobStatus = await fetchChatJobStatus(runId, conversationId, pendingJobId);
        if (cancelled) {
          return;
        }
        setPendingJob((current) => {
          if (!current || current.job_id !== jobStatus.job_id) {
            return current;
          }
          return {
            ...current,
            status: jobStatus.status,
          };
        });
        if (jobStatus.status === "completed" || jobStatus.status === "failed") {
          const [sessionPayload, contextsPayload] = await Promise.all([
            fetchChatSession(runId, conversationId),
            fetchChatSessionContexts(runId, conversationId),
          ]);
          if (cancelled) {
            return;
          }
          applySessionState(sessionPayload);
          setSessionContexts(contextsPayload);
          setErrorMessage(null);
          return;
        }
      } catch (error) {
        if (cancelled) {
          return;
        }
        setErrorMessage(
          error instanceof Error ? error.message : "Failed to poll long answer.",
        );
      }
      timeoutHandle = window.setTimeout(() => {
        void poll();
      }, CHAT_JOB_POLL_INTERVAL_MS);
    };

    void poll();
    return () => {
      cancelled = true;
      if (timeoutHandle) {
        window.clearTimeout(timeoutHandle);
      }
    };
  }, [conversationId, pendingJobId, runId]);

  async function submitMessage(options: {
    content: string;
    clarificationCity?: string;
    clarificationQuestion?: string;
    restoreInputOnError?: boolean;
  }): Promise<boolean> {
    if (!runId || !conversationId || isSending || sendLockRef.current) {
      return false;
    }
    const content = options.content.trim();
    if (!content) {
      return false;
    }
    sendLockRef.current = true;
    setIsSending(true);
    setErrorMessage(null);
    setRecoveryMessage(null);
    setRecoveryPollAttempt(0);
    setClarificationError(null);
    setPendingPrompt(content);
    setPendingClarificationCity(options.clarificationCity?.trim() || null);
    setPendingClarificationQuestion(options.clarificationQuestion?.trim() || null);
    const baselineMessageCount = sortedMessages.length;
    try {
      const response = await sendChatMessage(runId, conversationId, content, {
        clarificationCity: options.clarificationCity,
        clarificationQuestion: options.clarificationQuestion,
      });
      if (response.mode === "completed") {
        setMessages((current) => [
          ...current,
          response.user_message,
          response.assistant_message,
        ]);
        setPendingJob(null);
        setPendingJobRouting(null);
      } else {
        setMessages((current) => [
          ...current,
          response.user_message,
        ]);
        setPendingJob(response.job);
        setPendingJobRouting(response.routing ?? null);
      }
      await loadSessionContexts(runId, conversationId);
      return true;
    } catch (error) {
      if (isRequestTimeoutError(error)) {
        const recovered = await recoverTimedOutSend({
          activeRunId: runId,
          sessionId: conversationId,
          baselineMessageCount,
        });
        if (recovered) {
          return true;
        }
        setErrorMessage(
          "The request is still running on the backend, but the chat state did not update in time. Refresh to recover the saved result.",
        );
      } else {
        setErrorMessage(
          error instanceof Error ? error.message : "Message send failed.",
        );
      }
      if (options.restoreInputOnError) {
        setInputValue((current) => (current.trim() ? current : content));
      }
      return false;
    } finally {
      setPendingPrompt(null);
      setPendingClarificationCity(null);
      setPendingClarificationQuestion(null);
      setIsRecoveringSend(false);
      setRecoveryMessage(null);
      setRecoveryPollAttempt(0);
      setIsSending(false);
      sendLockRef.current = false;
    }
  }

  async function handleSend(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    const value = inputValue.trim();
    if (!value) {
      return;
    }
    setInputValue("");
    const succeeded = await submitMessage({
      content: value,
      restoreInputOnError: true,
    });
    if (!succeeded) {
      return;
    }
  }

  async function handleClarificationSubmit(): Promise<void> {
    if (!selectedClarificationCity || !clarificationQuestion) {
      return;
    }
    const chosenCity = selectedClarificationCity;
    const originalQuestion = clarificationQuestion;
    setIsCityClarificationOpen(false);
    setSelectedClarificationCity(null);
    setClarificationError(null);
    const succeeded = await submitMessage({
      content: `Focus only on ${chosenCity}.`,
      clarificationCity: chosenCity,
      clarificationQuestion: originalQuestion,
    });
    if (!succeeded) {
      setClarificationQuestion(originalQuestion);
      setSelectedClarificationCity(chosenCity);
      setIsCityClarificationOpen(true);
      return;
    }
    setClarificationQuestion(null);
  }

  function closeCityClarification(): void {
    if (isSending) {
      return;
    }
    setIsCityClarificationOpen(false);
    setSelectedClarificationCity(null);
    setClarificationError(null);
  }

  function toggleContextSelection(targetRunId: string): void {
    if (targetRunId === runId) {
      return;
    }
    setManagerSelection((current) => {
      if (current.includes(targetRunId)) {
        if (current.length <= 1) {
          return current;
        }
        return current.filter((runIdValue) => runIdValue !== targetRunId);
      }

      const context = contextById.get(targetRunId);
      if (!context) {
        return current;
      }
      setCatalogError(null);
      return [...current, targetRunId];
    });
  }

  useEffect(() => {
    if (!isContextManagerOpen || !runId) {
      return;
    }
    let cancelled = false;
    setIsLoadingCatalog(true);
    setCatalogError(null);
    fetchChatContextCatalog()
      .then((payload) => {
        if (cancelled) {
          return;
        }
        setContextCatalog(payload.contexts);
        const currentSelection = sessionContexts?.context_run_ids ?? [runId];
        setManagerSelection(currentSelection);
      })
      .catch((error) => {
        if (cancelled) {
          return;
        }
        setCatalogError(
          error instanceof Error ? error.message : "Failed to load context catalog.",
        );
      })
      .finally(() => {
        if (!cancelled) {
          setIsLoadingCatalog(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [isContextManagerOpen, runId, sessionContexts]);

  useEffect(() => {
    if (!showContextManager) {
      setIsContextManagerOpen(false);
    }
  }, [showContextManager]);

  async function saveContextSelection(): Promise<void> {
    if (!runId || !conversationId || managerSelection.length === 0 || pendingJob) {
      return;
    }
    setIsSavingContexts(true);
    setCatalogError(null);
    try {
      const payload = await updateChatSessionContexts(
        runId,
        conversationId,
        managerSelection,
      );
      setSessionContexts(payload);
      setIsContextManagerOpen(false);
    } catch (error) {
      setCatalogError(
        error instanceof Error ? error.message : "Failed to update contexts.",
      );
    } finally {
      setIsSavingContexts(false);
    }
  }

  const disabledInput = !canChat || !conversationId || isBootstrapping || !!pendingJob;
  const disabledSend =
    !canChat || !conversationId || isSending || isBootstrapping || !!pendingJob;
  const hasVisibleMessages =
    sortedMessages.length > 0 || (isSending && !!pendingPrompt) || !!pendingJob;

  return (
    <>
      <Card className="border-slate-300">
        <CardHeader className="pb-3">
          <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Sparkles className="h-4 w-4 text-teal-600" />
                Context Chat Workspace
              </CardTitle>
              <CardDescription>
                Multi-context chat grounded in saved run outputs and context bundles.
              </CardDescription>
            </div>
            <div className="ml-auto flex flex-col items-end gap-2">
              {conversationId ? (
                <Badge variant="outline">Session: {conversationId.slice(0, 8)}</Badge>
              ) : null}
              <div className="flex flex-wrap justify-end gap-2">
                {showContextManager ? (
                  <Button
                    type="button"
                    size="sm"
                    variant="outline"
                    onClick={() => setIsContextManagerOpen(true)}
                    disabled={
                      !sessionContexts || isLoadingContexts || isBootstrapping || !!pendingJob
                    }
                  >
                    Manage Contexts
                  </Button>
                ) : null}
                <Button type="button" size="sm" variant="outline" onClick={onClose}>
                  <ArrowLeft className="h-4 w-4" />
                  Back to Document
                </Button>
              </div>
            </div>
          </div>

          {sessionContexts ? (
            <div className="space-y-2 rounded-md border border-slate-200 bg-slate-50 p-3 text-xs text-slate-700">
              <p>
                Active sources: {sessionContexts.contexts.length + sessionContexts.followup_bundles.length}
              </p>
              {showTokenMetrics ? (
                <p>
                  Prompt estimate: {getPromptContextTokens(sessionContexts).toLocaleString()} /{" "}
                  {sessionContexts.token_cap.toLocaleString()}
                </p>
              ) : null}
              <div className="flex flex-wrap gap-2">
                {sessionContexts.contexts.map((context) => (
                  <Badge key={context.run_id} variant="secondary">
                    {showTokenMetrics
                      ? `${context.run_id} · ${getPromptContextTokens(context).toLocaleString()}`
                      : context.run_id}
                  </Badge>
                ))}
                {sessionContexts.followup_bundles.map((bundle) => (
                  <Badge key={bundle.bundle_id} variant="outline">
                    {showTokenMetrics
                      ? `${formatAdditionalResearchLabel(bundle.target_city)} · ${getPromptContextTokens(bundle).toLocaleString()}`
                      : formatAdditionalResearchLabel(bundle.target_city)}
                  </Badge>
                ))}
              </div>
              {sessionContexts.is_capped ? (
                [...sessionContexts.excluded_context_run_ids, ...sessionContexts.excluded_followup_bundle_ids]
                  .length > 0 ? (
                    <p className="text-amber-700">
                      Some sources are excluded due to token cap or missing artifacts:{" "}
                      {[
                        ...sessionContexts.excluded_context_run_ids,
                        ...sessionContexts.excluded_followup_bundle_ids,
                      ].join(", ")}
                    </p>
                  ) : (
                    <p className="text-amber-700">
                      The selected context is very large. AI may take longer to answer.
                    </p>
                  )
              ) : null}
            </div>
          ) : isLoadingContexts ? (
            <p className="text-xs text-slate-500">Loading session context metadata...</p>
          ) : null}
        </CardHeader>

        <CardContent className="space-y-4">
          <ScrollArea
            ref={messageScrollAreaRef}
            className="h-[52vh] rounded-md border border-slate-200 px-5 py-4"
          >
            {isBootstrapping ? (
              <div className="flex items-center gap-2 text-sm text-slate-600">
                <Loader2 className="h-4 w-4 animate-spin" />
                Preparing conversation memory...
              </div>
            ) : !hasVisibleMessages ? (
              <p className="text-sm text-slate-600">
                No chat messages yet. Ask about assumptions, compare runs, or request a narrower summary.
              </p>
            ) : (
              <div className="space-y-3">
                {sortedMessages.map((message, index) => {
                  const routingDescriptor = message.routing
                    ? describeRouting(message.routing)
                    : null;
                  return (
                    <div
                      key={`${message.created_at}-${index}`}
                      className={
                        message.role === "user"
                          ? "ml-8 rounded-lg bg-slate-900 p-3 text-sm text-white"
                          : "mr-8 rounded-lg bg-amber-50 p-3 text-sm text-slate-900"
                      }
                    >
                      <p className="mb-1 text-xs font-semibold uppercase tracking-wide opacity-75">
                        {message.role}
                      </p>
                      {message.role === "assistant" ? (
                        <div className="chat-markdown text-sm leading-relaxed">
                          {routingDescriptor ? (
                            <Badge
                              variant="outline"
                              className={cn("mb-3", routingDescriptor.className)}
                            >
                              {routingDescriptor.label}
                            </Badge>
                          ) : null}
                          <MarkdownWithReferences
                            content={message.content}
                            runId={runId}
                            conversationId={conversationId}
                            chatCitations={message.citations}
                            prefetchRunReferences={false}
                          />
                          {message.citation_warning ? (
                            <p className="mt-2 text-xs text-amber-700">
                              {message.citation_warning}
                            </p>
                          ) : null}
                        </div>
                      ) : (
                        <p className="whitespace-pre-wrap leading-relaxed">{message.content}</p>
                      )}
                    </div>
                  );
                })}
                {isSending && pendingPrompt ? (
                  <div className="ml-8 rounded-lg bg-slate-900 p-3 text-sm text-white">
                    <p className="mb-1 text-xs font-semibold uppercase tracking-wide opacity-75">
                      user
                    </p>
                    <p className="whitespace-pre-wrap leading-relaxed">{pendingPrompt}</p>
                  </div>
                ) : null}
                {isSending ? (
                  pendingClarificationCity ? (
                    <div className="mr-8 rounded-lg border border-sky-200 bg-sky-50 p-3 text-slate-900">
                      <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-sky-700">
                        assistant
                      </p>
                      <p className="mb-1 text-base font-semibold text-sky-950">
                        {formatAdditionalResearchLabel(pendingClarificationCity)} in progress...
                      </p>
                      {pendingClarificationQuestion ? (
                        <p className="mb-2 text-sm leading-relaxed text-sky-900">
                          {pendingClarificationQuestion}
                        </p>
                      ) : null}
                      <div className="inline-flex items-center gap-2 text-xs text-sky-800">
                        <span>Searching and reading city context</span>
                        <span className="chat-thinking-dots" aria-hidden="true">
                          <span className="chat-thinking-dot" />
                          <span className="chat-thinking-dot" />
                          <span className="chat-thinking-dot" />
                        </span>
                      </div>
                    </div>
                  ) : (
                    <div className="mr-8 rounded-lg border border-teal-100 bg-teal-50 p-3 text-slate-900">
                      <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-teal-800">
                        assistant
                      </p>
                      <p className="mb-2 text-base font-semibold text-teal-900">
                        {showDevDiagnostics && isRecoveringSend
                          ? "Reconnecting to backend..."
                          : "Processing your question..."}
                      </p>
                      <div className="inline-flex items-center gap-2 text-xs text-teal-800">
                        <span>
                          {showDevDiagnostics && isRecoveringSend
                            ? `Polling backend session state (check ${recoveryPollAttempt})`
                            : "Thinking"}
                        </span>
                        <span className="chat-thinking-dots" aria-hidden="true">
                          <span className="chat-thinking-dot" />
                          <span className="chat-thinking-dot" />
                          <span className="chat-thinking-dot" />
                        </span>
                      </div>
                    </div>
                  )
                ) : null}
                {!isSending && pendingJob ? (
                  <div className="mr-8 rounded-lg border border-teal-100 bg-teal-50 p-3 text-slate-900">
                    <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-teal-800">
                      assistant
                    </p>
                    <p className="mb-2 text-base font-semibold text-teal-900">
                      {pendingJobRouting?.action === "search_single_city"
                        ? `${formatAdditionalResearchLabel(pendingJobRouting.target_city)} in progress...`
                        : "Processing long answer..."}
                    </p>
                    <div className="inline-flex items-center gap-2 text-xs text-teal-800">
                      <span>
                        {pendingJobRouting?.action === "search_single_city"
                          ? "Finishing the answer with the new city context"
                          : "Working through a very large context"}
                      </span>
                      <span className="chat-thinking-dots" aria-hidden="true">
                        <span className="chat-thinking-dot" />
                        <span className="chat-thinking-dot" />
                        <span className="chat-thinking-dot" />
                      </span>
                    </div>
                  </div>
                ) : null}
              </div>
            )}
          </ScrollArea>

          <form className="space-y-3" onSubmit={handleSend}>
            <Input
              placeholder="Ask follow-up questions grounded in selected contexts..."
              value={inputValue}
              onChange={(event) => setInputValue(event.target.value)}
              disabled={disabledInput}
            />
            <div className="flex items-center justify-between gap-3">
              {errorMessage ? (
                <p className="text-xs text-red-600">{errorMessage}</p>
              ) : showDevDiagnostics && recoveryMessage ? (
                <p className="text-xs text-amber-700">
                  {recoveryPollAttempt > 0
                    ? `${recoveryMessage} Check ${recoveryPollAttempt}.`
                    : recoveryMessage}
                </p>
              ) : (
                <p className="text-xs text-slate-500">
                  Chat memory and selected contexts are persisted on backend.
                </p>
              )}
              <Button type="submit" size="sm" disabled={disabledSend}>
                {isSending ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
                <SendHorizonal className="h-4 w-4" />
                Send
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>

      <Dialog open={isContextManagerOpen} onOpenChange={setIsContextManagerOpen}>
        <DialogContent className="max-h-[85vh] max-w-3xl overflow-hidden p-0">
          <DialogHeader className="border-b border-slate-200 p-5">
            <DialogTitle>Context Manager</DialogTitle>
            <DialogDescription>
              Select or combine multiple run contexts. Direct prompt cap before overflow:{" "}
              {managerTokenCap.toLocaleString()} tokens.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-3 p-5">
            <div className="rounded-md border border-slate-200 bg-slate-50 p-3 text-xs text-slate-700">
              Selected prompt estimate: {selectedContextTokens.toLocaleString()} /{" "}
              {managerTokenCap.toLocaleString()}
              {selectionExceedsDirectCap ? (
                <p className="mt-2 text-amber-700">
                  This context is very large. AI may take longer to answer.
                </p>
              ) : null}
            </div>

            <ScrollArea className="h-[45vh] rounded-md border border-slate-200 p-3">
              {isLoadingCatalog ? (
                <div className="flex items-center gap-2 text-sm text-slate-600">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Loading available contexts...
                </div>
              ) : contextCatalog.length === 0 ? (
                <p className="text-sm text-slate-600">No completed run contexts available.</p>
              ) : (
                <div className="space-y-2">
                  {contextCatalog.map((context) => {
                    const selected = managerSelection.includes(context.run_id);
                    const isPinnedBaseRun = context.run_id === runId;
                    return (
                      <button
                        key={context.run_id}
                        type="button"
                        onClick={() => toggleContextSelection(context.run_id)}
                        disabled={isPinnedBaseRun}
                        className={
                          selected
                            ? "w-full rounded-md border border-teal-300 bg-teal-50 p-3 text-left"
                            : "w-full rounded-md border border-slate-200 bg-white p-3 text-left hover:border-slate-300 disabled:cursor-not-allowed disabled:opacity-45"
                        }
                      >
                        <div className="mb-1 flex items-center justify-between gap-2">
                          <p className="text-sm font-semibold text-slate-900">{context.run_id}</p>
                          <Badge variant="outline">
                            {getPromptContextTokens(context).toLocaleString()} tokens
                          </Badge>
                        </div>
                        <p className="line-clamp-2 text-xs text-slate-600">{context.question}</p>
                        {isPinnedBaseRun ? (
                          <p className="mt-1 text-xs text-teal-700">Pinned parent run</p>
                        ) : null}
                      </button>
                    );
                  })}
                </div>
              )}
            </ScrollArea>

            <div className="flex items-center justify-between gap-3">
              {catalogError ? (
                <p className="text-xs text-red-600">{catalogError}</p>
              ) : (
                <p className="text-xs text-slate-500">
                  The parent run stays pinned; select additional runs as needed. Selections above
                  the direct prompt cap are allowed and will use overflow mode.
                </p>
              )}
              <Button
                type="button"
                size="sm"
                onClick={() => void saveContextSelection()}
                disabled={
                  !!pendingJob ||
                  isSavingContexts ||
                  isLoadingCatalog ||
                  managerSelection.length === 0
                }
              >
                {isSavingContexts ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
                Save Selection
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      <Dialog
        open={isCityClarificationOpen}
        onOpenChange={(open) => {
          if (isSending) {
            return;
          }
          if (!open) {
            closeCityClarification();
            return;
          }
          setIsCityClarificationOpen(true);
        }}
      >
        <DialogContent className="flex max-h-[85vh] max-w-xl flex-col overflow-hidden p-0">
          <DialogHeader className="border-b border-slate-200 px-6 pb-4 pt-6 pr-14">
            <DialogTitle>Choose One City</DialogTitle>
            <DialogDescription>
              Pick one city for additional search.
            </DialogDescription>
          </DialogHeader>
          <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-hidden px-6 py-5">
            {clarificationQuestion ? (
              <div className="rounded-md border border-slate-200 bg-slate-50 p-3 text-sm text-slate-700">
                <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                  Original question
                </p>
                <p className="mt-1 whitespace-pre-wrap">{clarificationQuestion}</p>
              </div>
            ) : null}
            <SearchableCityPicker
              cities={clarificationCities}
              selectedCities={selectedClarificationCity ? [selectedClarificationCity] : []}
              onSelectCity={setSelectedClarificationCity}
              className="min-h-0 flex-1"
              disabled={isSending}
              errorMessage={clarificationError}
              isLoading={isLoadingClarificationCities}
              emptyMessage="No cities match the current filter."
              loadingMessage="Loading cities..."
            />
          </div>
          <DialogFooter className="border-t border-slate-200 bg-slate-50 px-6 py-4 sm:justify-between sm:space-x-0">
            <p className="text-xs text-slate-600">
              {selectedClarificationCity
                ? `Selected city: ${formatCityLabel(selectedClarificationCity)}`
                : "Select one city to continue with a focused follow-up search."}
            </p>
            <div className="flex items-center justify-end gap-2">
              <Button
                type="button"
                variant="outline"
                disabled={isSending}
                onClick={closeCityClarification}
              >
                Cancel
              </Button>
              <Button
                type="button"
                disabled={!selectedClarificationCity || isSending}
                onClick={() => void handleClarificationSubmit()}
              >
                {isSending ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
                Continue with {selectedClarificationCity ? formatCityLabel(selectedClarificationCity) : "City"}
              </Button>
            </div>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
