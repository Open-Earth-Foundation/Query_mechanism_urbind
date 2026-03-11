import type { Dispatch, SetStateAction } from "react";
import { useEffect, useRef, useState } from "react";

import {
  ChatJobHandle,
  ChatMessage,
  ChatRoutingMetadata,
  ChatSessionContextsResponse,
  ChatSessionResponse,
  fetchChatSession,
  fetchChatSessionContexts,
  sendChatMessage,
} from "@/lib/api";
import {
  CHAT_SEND_RECOVERY_POLL_INTERVAL_MS,
  CHAT_SEND_RECOVERY_TIMEOUT_MS,
} from "@/components/context-chat/chat-ui-config";
import { isRequestTimeoutError } from "@/components/context-chat/chat-utils";

interface UseChatSendArgs {
  runId: string;
  conversationId: string | null;
  sortedMessages: ChatMessage[];
  applySessionState: (session: ChatSessionResponse) => void;
  loadSessionContexts: (runId: string, sessionId: string) => Promise<void>;
  setMessages: Dispatch<SetStateAction<ChatMessage[]>>;
  setPendingJob: Dispatch<SetStateAction<ChatJobHandle | null>>;
  setPendingJobRouting: Dispatch<SetStateAction<ChatRoutingMetadata | null>>;
  setSessionContexts: Dispatch<SetStateAction<ChatSessionContextsResponse | null>>;
  setErrorMessage: Dispatch<SetStateAction<string | null>>;
}

interface SubmitMessageOptions {
  content: string;
  clarificationCity?: string;
}

interface UseChatSendResult {
  isSending: boolean;
  isRecoveringSend: boolean;
  recoveryMessage: string | null;
  recoveryPollAttempt: number;
  pendingPrompt: string | null;
  pendingClarificationCity: string | null;
  submitMessage: (options: SubmitMessageOptions) => Promise<boolean>;
}

export function useChatSend({
  runId,
  conversationId,
  sortedMessages,
  applySessionState,
  loadSessionContexts,
  setMessages,
  setPendingJob,
  setPendingJobRouting,
  setSessionContexts,
  setErrorMessage,
}: UseChatSendArgs): UseChatSendResult {
  const sendLockRef = useRef(false);
  const [isSending, setIsSending] = useState(false);
  const [isRecoveringSend, setIsRecoveringSend] = useState(false);
  const [recoveryMessage, setRecoveryMessage] = useState<string | null>(null);
  const [recoveryPollAttempt, setRecoveryPollAttempt] = useState(0);
  const [pendingPrompt, setPendingPrompt] = useState<string | null>(null);
  const [pendingClarificationCity, setPendingClarificationCity] = useState<string | null>(null);

  useEffect(() => {
    sendLockRef.current = false;
    setIsSending(false);
    setIsRecoveringSend(false);
    setRecoveryMessage(null);
    setRecoveryPollAttempt(0);
    setPendingPrompt(null);
    setPendingClarificationCity(null);
  }, [runId]);

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

  async function submitMessage(options: SubmitMessageOptions): Promise<boolean> {
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
    setPendingPrompt(content);
    setPendingClarificationCity(options.clarificationCity?.trim() || null);
    const baselineMessageCount = sortedMessages.length;
    try {
      const response = await sendChatMessage(runId, conversationId, content, {
        clarificationCity: options.clarificationCity,
      });
      if (response.mode === "completed") {
        setMessages((current) => [...current, response.user_message, response.assistant_message]);
        setPendingJob(null);
        setPendingJobRouting(null);
      } else {
        setMessages((current) => [...current, response.user_message]);
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
        setErrorMessage(error instanceof Error ? error.message : "Message send failed.");
      }
      return false;
    } finally {
      setPendingPrompt(null);
      setPendingClarificationCity(null);
      setIsRecoveringSend(false);
      setRecoveryMessage(null);
      setRecoveryPollAttempt(0);
      setIsSending(false);
      sendLockRef.current = false;
    }
  }

  return {
    isSending,
    isRecoveringSend,
    recoveryMessage,
    recoveryPollAttempt,
    pendingPrompt,
    pendingClarificationCity,
    submitMessage,
  };
}
