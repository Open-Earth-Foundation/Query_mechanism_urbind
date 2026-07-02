export function reportContentForDisplay(content: string, prompt?: string | null): string {
  const expectedPrompt = normalizePromptText(prompt);
  if (!expectedPrompt) {
    return content;
  }

  const normalizedContent = content.replace(/\r\n/g, "\n");
  const leadingPrompt = normalizedContent.match(
    /^#\s+(?:Question|Prompt)\s*\n([\s\S]*?)(?=\n#{1,6}\s+\S|$)/i,
  );
  if (!leadingPrompt) {
    return content;
  }

  const renderedPrompt = normalizePromptText(leadingPrompt[1]);
  if (renderedPrompt !== expectedPrompt) {
    return content;
  }

  const reportBody = normalizedContent.slice(leadingPrompt[0].length).replace(/^\n+/, "");
  return reportBody || content;
}

function normalizePromptText(value?: string | null): string {
  return (value ?? "").trim().replace(/\s+/g, " ");
}
