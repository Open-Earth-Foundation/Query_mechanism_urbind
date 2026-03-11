import type { Dispatch, SetStateAction } from "react";
import { useCallback, useEffect, useState } from "react";

import {
  ChatJobHandle,
  ChatMessage,
  ChatRoutingMetadata,
  ChatSessionContextsResponse,
  ChatSessionResponse,
  createChatSession,
  fetchChatSession,
  fetchChatSessionContexts,
  listChatSessions,
} from "@/lib/api";

interface UseChatSessionArgs {
  runId: string;
  enabled: boolean;
}

interface UseChatSessionResult {
  canChat: boolean;
  conversationId: string | null;
  messages: ChatMessage[];
  sessionContexts: ChatSessionContextsResponse | null;
  pendingJob: ChatJobHandle | null;
  pendingJobRouting: ChatRoutingMetadata | null;
  isBootstrapping: boolean;
  isLoadingContexts: boolean;
  errorMessage: string | null;
  setErrorMessage: Dispatch<SetStateAction<string | null>>;
  setMessages: Dispatch<SetStateAction<ChatMessage[]>>;
  setPendingJob: Dispatch<SetStateAction<ChatJobHandle | null>>;
  setPendingJobRouting: Dispatch<SetStateAction<ChatRoutingMetadata | null>>;
  setSessionContexts: Dispatch<SetStateAction<ChatSessionContextsResponse | null>>;
  applySessionState: (session: ChatSessionResponse) => void;
  loadSessionContexts: (activeRunId: string, sessionId: string) => Promise<void>;
}

export function useChatSession({
  runId,
  enabled,
}: UseChatSessionArgs): UseChatSessionResult {
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sessionContexts, setSessionContexts] = useState<ChatSessionContextsResponse | null>(null);
  const [pendingJob, setPendingJob] = useState<ChatJobHandle | null>(null);
  const [pendingJobRouting, setPendingJobRouting] = useState<ChatRoutingMetadata | null>(null);
  const [isBootstrapping, setIsBootstrapping] = useState(false);
  const [isLoadingContexts, setIsLoadingContexts] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const canChat = enabled && !!runId;

  useEffect(() => {
    setConversationId(null);
    setMessages([]);
    setSessionContexts(null);
    setPendingJob(null);
    setPendingJobRouting(null);
    setErrorMessage(null);
  }, [runId]);

  const applySessionState = useCallback((session: ChatSessionResponse): void => {
    setMessages(session.messages);
    setPendingJob(session.pending_job ?? null);
    setPendingJobRouting(null);
  }, []);

  const loadSessionContexts = useCallback(
    async (activeRunId: string, sessionId: string): Promise<void> => {
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
    },
    [],
  );

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
          const results = await Promise.allSettled(
            existing.conversations.map((id) => fetchChatSession(activeRunId, id)),
          );
          if (cancelled) {
            return;
          }
          const sessionsWithMetadata = results
            .filter(
              (
                result,
              ): result is PromiseFulfilledResult<Awaited<ReturnType<typeof fetchChatSession>>> =>
                result.status === "fulfilled",
            )
            .map((result) => result.value);
          if (sessionsWithMetadata.length > 0) {
            sessionsWithMetadata.sort((a, b) => b.updated_at.localeCompare(a.updated_at));
            const mostRecent = sessionsWithMetadata[0];
            sessionId = mostRecent.conversation_id;
            applySessionState(mostRecent);
          }
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
        if (!cancelled) {
          setErrorMessage(error instanceof Error ? error.message : "Failed to open chat session.");
        }
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
  }, [applySessionState, canChat, loadSessionContexts, runId]);

  return {
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
  };
}
