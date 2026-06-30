interface PromptBlockProps {
  prompt: string | null | undefined;
}

export function PromptBlock({ prompt }: PromptBlockProps) {
  const trimmedPrompt = prompt?.trim();
  if (!trimmedPrompt) {
    return null;
  }

  return (
    <div className="document-prompt-block">
      <p className="document-prompt-heading">Prompt</p>
      <div className="document-prompt-text">{trimmedPrompt}</div>
    </div>
  );
}
