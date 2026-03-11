"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { ArrowLeft, Loader2, SendHorizonal, Sparkles } from "lucide-react";

import { CityClarificationDialog } from "@/components/context-chat/city-clarification-dialog";
import { ChatMessageList } from "@/components/context-chat/chat-message-list";
import { ContextManagerDialog } from "@/components/context-chat/context-manager-dialog";
import {
  buildChatTurns,
  buildDisplayMessages,
  formatAdditionalResearchLabel,
  formatExcludedBundleLabel,
} from "@/components/context-chat/chat-utils";
import { useChatJobPolling } from "@/components/context-chat/hooks/use-chat-job-polling";
import { useChatSend } from "@/components/context-chat/hooks/use-chat-send";
import { useChatSession } from "@/components/context-chat/hooks/use-chat-session";
import { DEFAULT_CONTEXT_TOKEN_CAP } from "@/components/context-chat/chat-ui-config";
import {
  ChatContextSummary,
  fetchChatContextCatalog,
  fetchCities,
  updateChatSessionContexts,
} from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";

interface ContextChatWorkspaceProps {
  runId: string;
  enabled: boolean;
  onClose: () => void;
  showContextManager: boolean;
  showDevDiagnostics: boolean;
  showTokenMetrics: boolean;
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
  const {
    canChat,
    conversationId,
    messages,
    sessionContexts,
    pendingJob,
    pendingJobRouting,
    isBootstrapping,
    isLoadingContexts,
    errorMessage,
    setErrorMessage,
    setMessages,
    setPendingJob,
    setPendingJobRouting,
    setSessionContexts,
    applySessionState,
    loadSessionContexts,
  } = useChatSession({ runId, enabled });
  const [inputValue, setInputValue] = useState("");
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
  const {
    isSending,
    isRecoveringSend,
    recoveryMessage,
    recoveryPollAttempt,
    pendingPrompt,
    pendingClarificationCity,
    submitMessage,
  } = useChatSend({
    runId,
    conversationId,
    sortedMessages: [...messages].sort((a, b) => a.created_at.localeCompare(b.created_at)),
    applySessionState,
    loadSessionContexts,
    setMessages,
    setPendingJob,
    setPendingJobRouting,
    setSessionContexts,
    setErrorMessage,
  });

  useEffect(() => {
    setInputValue("");
    setContextCatalog([]);
    setManagerSelection([]);
    setCatalogError(null);
    setIsContextManagerOpen(false);
    setIsCityClarificationOpen(false);
    setClarificationQuestion(null);
    setClarificationCities([]);
    setSelectedClarificationCity(null);
    setClarificationError(null);
    setIsLoadingClarificationCities(false);
    handledClarificationKeyRef.current = null;
  }, [runId]);

  const sortedMessages = useMemo(
    () => [...messages].sort((a, b) => a.created_at.localeCompare(b.created_at)),
    [messages],
  );
  const displayMessages = useMemo(() => buildDisplayMessages(sortedMessages), [sortedMessages]);
  const chatTurns = useMemo(() => buildChatTurns(displayMessages), [displayMessages]);

  useEffect(() => {
    const root = messageScrollAreaRef.current;
    if (!root) {
      return;
    }
    const handle = window.requestAnimationFrame(() => {
      const viewport = root.querySelector<HTMLDivElement>("[data-radix-scroll-area-viewport]");
      if (viewport) {
        viewport.scrollTop = viewport.scrollHeight;
      }
    });
    return () => window.cancelAnimationFrame(handle);
  }, [chatTurns.length, isSending, pendingJob, pendingPrompt]);

  const contextById = useMemo(() => {
    const mapping = new Map<string, ChatContextSummary>();
    contextCatalog.forEach((context) => {
      mapping.set(context.run_id, context);
    });
    return mapping;
  }, [contextCatalog]);
  const managerTokenCap = sessionContexts?.token_cap ?? DEFAULT_CONTEXT_TOKEN_CAP;
  const selectedContextTokens = useMemo(
    () =>
      managerSelection.reduce((sum, contextRunId) => {
        const context = contextById.get(contextRunId);
        return sum + (context ? context.prompt_context_tokens : 0);
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
        if (!cancelled) {
          setClarificationCities(payload.cities);
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setClarificationError(
            error instanceof Error ? error.message : "Failed to load city list.",
          );
        }
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

  useChatJobPolling({
    runId,
    conversationId,
    pendingJob,
    applySessionState,
    setSessionContexts,
    setPendingJob,
    setErrorMessage,
  });

  useEffect(() => {
    if (!isContextManagerOpen || !runId) {
      return;
    }
    let cancelled = false;
    setIsLoadingCatalog(true);
    setCatalogError(null);
    fetchChatContextCatalog()
      .then((payload) => {
        if (!cancelled) {
          setContextCatalog(payload.contexts);
          setManagerSelection(sessionContexts?.context_run_ids ?? [runId]);
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setCatalogError(
            error instanceof Error ? error.message : "Failed to load context catalog.",
          );
        }
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

  async function handleSend(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    const value = inputValue.trim();
    if (!value) {
      return;
    }
    setInputValue("");
    const succeeded = await submitMessage({ content: value });
    if (!succeeded) {
      setInputValue(value);
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
      content: originalQuestion,
      clarificationCity: chosenCity,
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
        return current.length <= 1 ? current : current.filter((value) => value !== targetRunId);
      }
      if (!contextById.get(targetRunId)) {
        return current;
      }
      setCatalogError(null);
      return [...current, targetRunId];
    });
  }

  async function saveContextSelection(): Promise<void> {
    if (!runId || !conversationId || managerSelection.length === 0 || pendingJob) {
      return;
    }
    setIsSavingContexts(true);
    setCatalogError(null);
    try {
      const payload = await updateChatSessionContexts(runId, conversationId, managerSelection);
      setSessionContexts(payload);
      setIsContextManagerOpen(false);
    } catch (error) {
      setCatalogError(error instanceof Error ? error.message : "Failed to update contexts.");
    } finally {
      setIsSavingContexts(false);
    }
  }

  const disabledInput = !canChat || !conversationId || isBootstrapping || !!pendingJob;
  const disabledSend = !canChat || !conversationId || isSending || isBootstrapping || !!pendingJob;
  const hasVisibleMessages = displayMessages.length > 0 || (isSending && !!pendingPrompt) || !!pendingJob;
  const shouldRenderStandalonePendingJob =
    !isSending && !!pendingJob && (!chatTurns.at(-1) || !chatTurns.at(-1)?.userMessage);

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
                    disabled={!sessionContexts || isLoadingContexts || isBootstrapping || !!pendingJob}
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
                  Context tokens: {sessionContexts.prompt_context_tokens.toLocaleString()} /{" "}
                  {sessionContexts.token_cap.toLocaleString()}
                </p>
              ) : null}
              <div className="flex flex-wrap gap-2">
                {sessionContexts.contexts.map((context) => (
                  <Badge key={context.run_id} variant="secondary">
                    {showTokenMetrics
                      ? `${context.run_id} · ${context.prompt_context_tokens.toLocaleString()}`
                      : context.run_id}
                  </Badge>
                ))}
                {sessionContexts.followup_bundles.map((bundle) => (
                  <Badge key={bundle.bundle_id} variant="outline">
                    {showTokenMetrics
                      ? `${formatAdditionalResearchLabel(bundle.target_city)} · ${bundle.prompt_context_tokens.toLocaleString()}`
                      : formatAdditionalResearchLabel(bundle.target_city)}
                  </Badge>
                ))}
              </div>
              {sessionContexts.excluded_followup_bundle_ids.length > 0 ? (
                <p className="text-amber-700">
                  {sessionContexts.excluded_followup_bundle_ids.map(formatExcludedBundleLabel).join(", ")}{" "}
                  excluded — context already exceeds the {sessionContexts.token_cap.toLocaleString()}-token cap.
                  Reduce the selection to re-attach city research.
                </p>
              ) : null}
              {sessionContexts.excluded_context_run_ids.length > 0 ? (
                <p className="text-amber-700">
                  Some runs could not be loaded (broken or missing artifacts):{" "}
                  {sessionContexts.excluded_context_run_ids.join(", ")}
                </p>
              ) : null}
              {sessionContexts.is_capped &&
              sessionContexts.excluded_followup_bundle_ids.length === 0 &&
              sessionContexts.excluded_context_run_ids.length === 0 ? (
                <p className="text-amber-700">
                  The selected context is very large. AI may take longer to answer.
                </p>
              ) : null}
            </div>
          ) : isLoadingContexts ? (
            <p className="text-xs text-slate-500">Loading session context metadata...</p>
          ) : null}
        </CardHeader>

        <CardContent className="space-y-4">
          <ChatMessageList
            runId={runId}
            conversationId={conversationId}
            messageScrollAreaRef={messageScrollAreaRef}
            isBootstrapping={isBootstrapping}
            chatTurns={chatTurns}
            hasVisibleMessages={hasVisibleMessages}
            isSending={isSending}
            pendingPrompt={pendingPrompt}
            pendingClarificationCity={pendingClarificationCity}
            pendingJobRouting={pendingJobRouting}
            pendingJob={pendingJob}
            shouldRenderStandalonePendingJob={shouldRenderStandalonePendingJob}
            showDevDiagnostics={showDevDiagnostics}
            isRecoveringSend={isRecoveringSend}
            recoveryPollAttempt={recoveryPollAttempt}
          />

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

      <ContextManagerDialog
        open={isContextManagerOpen}
        runId={runId}
        managerTokenCap={managerTokenCap}
        selectedContextTokens={selectedContextTokens}
        selectionExceedsDirectCap={selectionExceedsDirectCap}
        isLoadingCatalog={isLoadingCatalog}
        contextCatalog={contextCatalog}
        managerSelection={managerSelection}
        onToggleSelection={toggleContextSelection}
        onOpenChange={setIsContextManagerOpen}
        onSave={() => void saveContextSelection()}
        isSavingContexts={isSavingContexts}
        pendingJob={pendingJob}
        catalogError={catalogError}
      />

      <CityClarificationDialog
        open={isCityClarificationOpen}
        isSending={isSending}
        question={clarificationQuestion}
        cities={clarificationCities}
        selectedCity={selectedClarificationCity}
        onSelectCity={setSelectedClarificationCity}
        onClose={closeCityClarification}
        onConfirm={() => void handleClarificationSubmit()}
        errorMessage={clarificationError}
        isLoading={isLoadingClarificationCities}
      />
    </>
  );
}
