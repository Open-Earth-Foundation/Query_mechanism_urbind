import { ChatMessage, ChatRoutingMetadata } from "@/lib/api";
import { formatCityLabel } from "@/lib/utils";

export interface ChatTurn {
  key: string;
  userMessage: ChatMessage | null;
  assistantMessages: ChatMessage[];
}

export interface PendingAssistantCardOptions {
  accentClassName: string;
  labelClassName: string;
  titleClassName: string;
  detailClassName: string;
  title: string;
  detail: string;
  description?: string | null;
}

const CHAT_RETRY_FAILURE_MESSAGES = new Set<string>([
  "The long-context answer did not complete. Please retry.",
  "The long-context answer was interrupted before completion. Please retry.",
]);

export function isRequestTimeoutError(error: unknown): boolean {
  return error instanceof Error && /^Request timeout after \d+s\.$/.test(error.message);
}

export function formatAdditionalResearchLabel(city: string | null | undefined): string {
  const cityLabel = formatCityLabel(city ?? "");
  return cityLabel ? `Additional ${cityLabel} research` : "Additional city research";
}

export function formatExcludedBundleLabel(bundleId: string): string {
  const parts = bundleId.split("_");
  const cityKey = parts.length >= 4 ? parts.slice(3).join("_") : bundleId;
  return formatAdditionalResearchLabel(cityKey);
}

export function describeRouting(
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

function normalizeMessageFingerprint(content: string): string {
  return content.trim().replace(/\s+/g, " ").toLocaleLowerCase();
}

function isRetryFailureMessage(content: string): boolean {
  return CHAT_RETRY_FAILURE_MESSAGES.has(content.trim());
}

export function buildDisplayMessages(messages: ChatMessage[]): ChatMessage[] {
  const nonEmptyMessages = messages.filter((message) => message.content.trim().length > 0);
  const displayMessages: ChatMessage[] = [];
  let skippedRetryFingerprint: string | null = null;

  nonEmptyMessages.forEach((message, index) => {
    if (message.role === "assistant" && isRetryFailureMessage(message.content)) {
      const previousMessage = displayMessages.at(-1);
      const nextMessage = nonEmptyMessages[index + 1];
      if (
        previousMessage?.role === "user" &&
        nextMessage?.role === "user" &&
        normalizeMessageFingerprint(previousMessage.content) ===
          normalizeMessageFingerprint(nextMessage.content)
      ) {
        skippedRetryFingerprint = normalizeMessageFingerprint(previousMessage.content);
        return;
      }
    }

    if (
      skippedRetryFingerprint &&
      message.role === "user" &&
      normalizeMessageFingerprint(message.content) === skippedRetryFingerprint
    ) {
      skippedRetryFingerprint = null;
      return;
    }

    skippedRetryFingerprint = null;
    displayMessages.push(message);
  });

  return displayMessages;
}

export function buildChatTurns(messages: ChatMessage[]): ChatTurn[] {
  const turns: ChatTurn[] = [];
  let currentTurn: ChatTurn | null = null;

  messages.forEach((message, index) => {
    if (message.role === "user") {
      if (currentTurn) {
        turns.push(currentTurn);
      }
      currentTurn = {
        key: `${message.created_at}-${index}`,
        userMessage: message,
        assistantMessages: [],
      };
      return;
    }

    if (!currentTurn) {
      turns.push({
        key: `${message.created_at}-${index}`,
        userMessage: null,
        assistantMessages: [message],
      });
      return;
    }

    currentTurn.assistantMessages.push(message);
  });

  if (currentTurn) {
    turns.push(currentTurn);
  }

  return turns;
}

export function buildPendingJobCardOptions(
  routing: ChatRoutingMetadata | null,
): PendingAssistantCardOptions {
  return {
    accentClassName: "border-teal-100 bg-teal-50",
    labelClassName: "text-teal-800",
    titleClassName: "text-teal-900",
    detailClassName: "text-teal-800",
    title:
      routing?.action === "search_single_city"
        ? `${formatAdditionalResearchLabel(routing.target_city)} in progress...`
        : "Processing long answer...",
    detail:
      routing?.action === "search_single_city"
        ? "Finishing the answer with the new city context"
        : "Working through a very large context",
  };
}
