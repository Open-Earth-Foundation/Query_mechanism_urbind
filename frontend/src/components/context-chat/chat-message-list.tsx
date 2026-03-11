import type { RefObject } from "react";
import { Loader2 } from "lucide-react";

import { MarkdownWithReferences } from "@/components/markdown-with-references";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ChatRoutingMetadata } from "@/lib/api";
import { cn } from "@/lib/utils";

import {
  ChatTurn,
  PendingAssistantCardOptions,
  buildPendingJobCardOptions,
  describeRouting,
  formatAdditionalResearchLabel,
} from "@/components/context-chat/chat-utils";

function UserMessageBubble({ content }: { content: string }) {
  return (
    <div className="flex justify-end">
      <div className="min-w-0 w-full max-w-[88%] rounded-[1.25rem] bg-slate-900 px-4 py-3 text-sm text-white shadow-sm">
        <p className="mb-2 text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-300">
          User
        </p>
        <p className="whitespace-pre-wrap break-words leading-relaxed">{content}</p>
      </div>
    </div>
  );
}

function PendingAssistantCard({ options }: { options: PendingAssistantCardOptions }) {
  return (
    <div className="flex justify-start">
      <div
        className={cn(
          "min-w-0 w-full max-w-[88%] rounded-[1.25rem] border px-4 py-3 text-slate-900 shadow-sm",
          options.accentClassName,
        )}
      >
        <p
          className={cn(
            "mb-2 text-[11px] font-semibold uppercase tracking-[0.18em]",
            options.labelClassName,
          )}
        >
          Assistant
        </p>
        <p className={cn("mb-2 text-base font-semibold", options.titleClassName)}>
          {options.title}
        </p>
        {options.description ? (
          <p className={cn("mb-2 text-sm leading-relaxed", options.detailClassName)}>
            {options.description}
          </p>
        ) : null}
        <div className={cn("inline-flex items-center gap-2 text-xs", options.detailClassName)}>
          <span>{options.detail}</span>
          <span className="chat-thinking-dots" aria-hidden="true">
            <span className="chat-thinking-dot" />
            <span className="chat-thinking-dot" />
            <span className="chat-thinking-dot" />
          </span>
        </div>
      </div>
    </div>
  );
}

interface ChatMessageListProps {
  runId: string;
  conversationId: string | null;
  messageScrollAreaRef: RefObject<HTMLDivElement | null>;
  isBootstrapping: boolean;
  chatTurns: ChatTurn[];
  hasVisibleMessages: boolean;
  isSending: boolean;
  pendingPrompt: string | null;
  pendingClarificationCity: string | null;
  pendingJobRouting: ChatRoutingMetadata | null;
  pendingJob: { job_id: string } | null;
  shouldRenderStandalonePendingJob: boolean;
  showDevDiagnostics: boolean;
  isRecoveringSend: boolean;
  recoveryPollAttempt: number;
}

export function ChatMessageList({
  runId,
  conversationId,
  messageScrollAreaRef,
  isBootstrapping,
  chatTurns,
  hasVisibleMessages,
  isSending,
  pendingPrompt,
  pendingClarificationCity,
  pendingJobRouting,
  pendingJob,
  shouldRenderStandalonePendingJob,
  showDevDiagnostics,
  isRecoveringSend,
  recoveryPollAttempt,
}: ChatMessageListProps) {
  const lastChatTurn = chatTurns.at(-1) ?? null;

  return (
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
          No chat messages yet. Ask about assumptions, compare runs, or request a narrower
          summary.
        </p>
      ) : (
        <div className="space-y-5">
          {chatTurns.map((turn, turnIndex) => (
            <div key={turn.key} className="space-y-3">
              {turn.userMessage ? <UserMessageBubble content={turn.userMessage.content} /> : null}

              {turn.assistantMessages.map((message, assistantIndex) => {
                const routingDescriptor = message.routing ? describeRouting(message.routing) : null;
                return (
                  <div key={`${message.created_at}-${assistantIndex}`} className="flex justify-start">
                    <div className="min-w-0 w-full max-w-[88%] rounded-[1.25rem] border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-slate-900 shadow-sm">
                      <p className="mb-2 text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">
                        Assistant
                      </p>
                      <div className="chat-markdown break-words text-sm leading-relaxed">
                        {routingDescriptor ? (
                          <Badge variant="outline" className={cn("mb-3", routingDescriptor.className)}>
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
                          <p className="mt-2 text-xs text-amber-700">{message.citation_warning}</p>
                        ) : null}
                      </div>
                    </div>
                  </div>
                );
              })}

              {!isSending && pendingJob && turnIndex === chatTurns.length - 1 && turn.userMessage ? (
                <PendingAssistantCard options={buildPendingJobCardOptions(pendingJobRouting)} />
              ) : null}
            </div>
          ))}
          {isSending && pendingPrompt ? (
            <div className="space-y-3">
              <UserMessageBubble content={pendingPrompt} />
              {pendingClarificationCity ? (
                <PendingAssistantCard
                  options={{
                    accentClassName: "border-sky-200 bg-sky-50",
                    labelClassName: "text-sky-700",
                    titleClassName: "text-sky-950",
                    detailClassName: "text-sky-800",
                    title: `${formatAdditionalResearchLabel(pendingClarificationCity)} in progress...`,
                    detail: "Searching and reading city context",
                  }}
                />
              ) : (
                <PendingAssistantCard
                  options={{
                    accentClassName: "border-teal-100 bg-teal-50",
                    labelClassName: "text-teal-800",
                    titleClassName: "text-teal-900",
                    detailClassName: "text-teal-800",
                    title:
                      showDevDiagnostics && isRecoveringSend
                        ? "Reconnecting to backend..."
                        : "Processing your question...",
                    detail:
                      showDevDiagnostics && isRecoveringSend
                        ? `Polling backend session state (check ${recoveryPollAttempt})`
                        : "Thinking",
                  }}
                />
              )}
            </div>
          ) : null}
          {shouldRenderStandalonePendingJob && !isSending && !lastChatTurn?.userMessage ? (
            <PendingAssistantCard options={buildPendingJobCardOptions(pendingJobRouting)} />
          ) : null}
        </div>
      )}
    </ScrollArea>
  );
}
