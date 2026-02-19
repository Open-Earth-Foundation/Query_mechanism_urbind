"use client";

import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  ArrowLeft,
  Loader2,
  SendHorizonal,
  Settings2,
  Sparkles,
} from "lucide-react";

import {
  ChatContextSummary,
  ChatMessage,
  ChatSessionContextsResponse,
  createChatSession,
  fetchChatContextCatalog,
  fetchChatSession,
  fetchChatSessionContexts,
  listChatSessions,
  sendChatMessage,
  updateChatSessionContexts,
} from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";

interface ContextChatWorkspaceProps {
  runId: string;
  enabled: boolean;
  onClose: () => void;
}

const DEFAULT_CONTEXT_TOKEN_CAP = 250000;

export function ContextChatWorkspace({
  runId,
  enabled,
  onClose,
}: ContextChatWorkspaceProps) {
  const messageScrollAreaRef = useRef<HTMLDivElement | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sessionContexts, setSessionContexts] =
    useState<ChatSessionContextsResponse | null>(null);

  const [inputValue, setInputValue] = useState("");
  const [pendingPrompt, setPendingPrompt] = useState<string | null>(null);
  const [isBootstrapping, setIsBootstrapping] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [isLoadingContexts, setIsLoadingContexts] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const [isContextManagerOpen, setIsContextManagerOpen] = useState(false);
  const [isLoadingCatalog, setIsLoadingCatalog] = useState(false);
  const [isSavingContexts, setIsSavingContexts] = useState(false);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [contextCatalog, setContextCatalog] = useState<ChatContextSummary[]>([]);
  const [managerSelection, setManagerSelection] = useState<string[]>([]);

  const canChat = enabled && !!runId;

  useEffect(() => {
    setConversationId(null);
    setMessages([]);
    setSessionContexts(null);
    setInputValue("");
    setPendingPrompt(null);
    setErrorMessage(null);
    setContextCatalog([]);
    setManagerSelection([]);
    setCatalogError(null);
  }, [runId]);

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
            setMessages(mostRecent.messages);
          }
          // If all sessions failed to load, fall through to create a new one
        }
        if (!sessionId) {
          const created = await createChatSession(activeRunId);
          if (cancelled) {
            return;
          }
          sessionId = created.conversation_id;
          setMessages(created.messages);
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
  }, [scrollMessagesToBottom, sortedMessages.length, isSending, pendingPrompt]);

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
      managerSelection.reduce((sum, runIdValue) => {
        const context = contextById.get(runIdValue);
        return sum + (context?.total_tokens ?? 0);
      }, 0),
    [contextById, managerSelection],
  );

  async function handleSend(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    if (!runId || !conversationId || isSending) {
      return;
    }
    const value = inputValue.trim();
    if (!value) {
      return;
    }
    setIsSending(true);
    setErrorMessage(null);
    setInputValue("");
    setPendingPrompt(value);
    try {
      const response = await sendChatMessage(runId, conversationId, value);
      setMessages((current) => [
        ...current,
        response.user_message,
        response.assistant_message,
      ]);
      await loadSessionContexts(runId, conversationId);
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : "Message send failed.",
      );
      setInputValue(value);
    } finally {
      setPendingPrompt(null);
      setIsSending(false);
    }
  }

  function toggleContextSelection(targetRunId: string): void {
    setManagerSelection((current) => {
      if (current.includes(targetRunId)) {
        if (current.length <= 1) {
          return current;
        }
        return current.filter((runIdValue) => runIdValue !== targetRunId);
      }

      const context = contextById.get(targetRunId);
      const additionalTokens = context?.total_tokens ?? 0;
      if (selectedContextTokens + additionalTokens > managerTokenCap) {
        setCatalogError(
          `Selection exceeds token cap ${managerTokenCap.toLocaleString()} tokens.`,
        );
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

  async function saveContextSelection(): Promise<void> {
    if (!runId || !conversationId || managerSelection.length === 0) {
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

  const disabledSend = !canChat || !conversationId || isSending || isBootstrapping;

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
            <div className="flex flex-wrap items-center gap-2">
              {conversationId ? (
                <Badge variant="outline">Session: {conversationId.slice(0, 8)}</Badge>
              ) : null}
              <Button type="button" size="sm" variant="outline" onClick={onClose}>
                <ArrowLeft className="h-4 w-4" />
                Back to Document
              </Button>
              <Button
                type="button"
                size="sm"
                variant="outline"
                onClick={() => setIsContextManagerOpen(true)}
                disabled={!conversationId || isBootstrapping}
              >
                <Settings2 className="h-4 w-4" />
                Manage Contexts
              </Button>
            </div>
          </div>

          {sessionContexts ? (
            <div className="space-y-2 rounded-md border border-slate-200 bg-slate-50 p-3 text-xs text-slate-700">
              <p>
                Active contexts: {sessionContexts.contexts.length} | Tokens:{" "}
                {sessionContexts.total_tokens.toLocaleString()} /{" "}
                {sessionContexts.token_cap.toLocaleString()}
              </p>
              <div className="flex flex-wrap gap-2">
                {sessionContexts.contexts.map((context) => (
                  <Badge key={context.run_id} variant="secondary">
                    {context.run_id} ({context.total_tokens.toLocaleString()})
                  </Badge>
                ))}
              </div>
              {sessionContexts.is_capped ? (
                <p className="text-amber-700">
                  Some selected contexts are excluded due to token cap or missing artifacts:{" "}
                  {sessionContexts.excluded_context_run_ids.join(", ")}
                </p>
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
            ) : sortedMessages.length === 0 ? (
              <p className="text-sm text-slate-600">
                No chat messages yet. Ask about assumptions, compare runs, or request a narrower summary.
              </p>
            ) : (
              <div className="space-y-3">
                {sortedMessages.map((message, index) => (
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
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {message.content}
                        </ReactMarkdown>
                      </div>
                    ) : (
                      <p className="whitespace-pre-wrap leading-relaxed">{message.content}</p>
                    )}
                  </div>
                ))}
                {isSending ? (
                  <div className="mr-8 rounded-lg border border-teal-100 bg-teal-50 p-3 text-sm text-slate-900">
                    <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-teal-800">
                      assistant
                    </p>
                    <div className="mb-2 inline-flex items-center gap-2 text-teal-800">
                      <span>Thinking</span>
                      <span className="chat-thinking-dots" aria-hidden="true">
                        <span className="chat-thinking-dot" />
                        <span className="chat-thinking-dot" />
                        <span className="chat-thinking-dot" />
                      </span>
                    </div>
                    {pendingPrompt ? (
                      <p className="text-xs text-slate-600">
                        Working on: {pendingPrompt}
                      </p>
                    ) : null}
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
              disabled={disabledSend}
            />
            <div className="flex items-center justify-between gap-3">
              {errorMessage ? (
                <p className="text-xs text-red-600">{errorMessage}</p>
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
              Switch or combine multiple run contexts. Hard cap:{" "}
              {managerTokenCap.toLocaleString()} tokens.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-3 p-5">
            <div className="rounded-md border border-slate-200 bg-slate-50 p-3 text-xs text-slate-700">
              Selected token estimate: {selectedContextTokens.toLocaleString()} /{" "}
              {managerTokenCap.toLocaleString()}
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
                    const wouldOverflow =
                      !selected &&
                      selectedContextTokens + context.total_tokens > managerTokenCap;
                    return (
                      <button
                        key={context.run_id}
                        type="button"
                        onClick={() => toggleContextSelection(context.run_id)}
                        disabled={wouldOverflow}
                        className={
                          selected
                            ? "w-full rounded-md border border-teal-300 bg-teal-50 p-3 text-left"
                            : "w-full rounded-md border border-slate-200 bg-white p-3 text-left hover:border-slate-300 disabled:cursor-not-allowed disabled:opacity-45"
                        }
                      >
                        <div className="mb-1 flex items-center justify-between gap-2">
                          <p className="text-sm font-semibold text-slate-900">{context.run_id}</p>
                          <Badge variant="outline">
                            {context.total_tokens.toLocaleString()} tokens
                          </Badge>
                        </div>
                        <p className="line-clamp-2 text-xs text-slate-600">{context.question}</p>
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
                  Keep at least one context selected.
                </p>
              )}
              <Button
                type="button"
                size="sm"
                onClick={() => void saveContextSelection()}
                disabled={
                  isSavingContexts ||
                  isLoadingCatalog ||
                  managerSelection.length === 0 ||
                  selectedContextTokens > managerTokenCap
                }
              >
                {isSavingContexts ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
                Save Selection
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
