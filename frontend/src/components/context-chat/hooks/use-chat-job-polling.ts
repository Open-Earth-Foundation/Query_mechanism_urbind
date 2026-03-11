import type { Dispatch, SetStateAction } from "react";
import { useEffect } from "react";

import {
  ChatJobHandle,
  ChatSessionContextsResponse,
  ChatSessionResponse,
  fetchChatJobStatus,
  fetchChatSession,
  fetchChatSessionContexts,
} from "@/lib/api";
import { CHAT_JOB_POLL_INTERVAL_MS } from "@/components/context-chat/chat-ui-config";

interface UseChatJobPollingArgs {
  runId: string;
  conversationId: string | null;
  pendingJob: ChatJobHandle | null;
  applySessionState: (session: ChatSessionResponse) => void;
  setSessionContexts: Dispatch<SetStateAction<ChatSessionContextsResponse | null>>;
  setPendingJob: Dispatch<SetStateAction<ChatJobHandle | null>>;
  setErrorMessage: Dispatch<SetStateAction<string | null>>;
}

export function useChatJobPolling({
  runId,
  conversationId,
  pendingJob,
  applySessionState,
  setSessionContexts,
  setPendingJob,
  setErrorMessage,
}: UseChatJobPollingArgs): void {
  const pendingJobId = pendingJob?.job_id ?? null;

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
        if (!cancelled) {
          setErrorMessage(error instanceof Error ? error.message : "Failed to poll long answer.");
        }
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
  }, [
    applySessionState,
    conversationId,
    pendingJobId,
    runId,
    setErrorMessage,
    setPendingJob,
    setSessionContexts,
  ]);
}
