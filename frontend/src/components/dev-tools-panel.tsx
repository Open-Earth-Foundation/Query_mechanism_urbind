"use client";

import { useState } from "react";
import { Copy, KeyRound } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { getUserApiKey, setUserApiKey } from "@/lib/api";

interface DevToolsPanelProps {
  apiKeyIssue: boolean;
  runId: string | null;
}

function maskApiKey(value: string): string {
  if (value.length <= 8) {
    return "Applied in this tab";
  }
  return `${value.slice(0, 4)}...${value.slice(-4)}`;
}

export function DevToolsPanel({ apiKeyIssue, runId }: DevToolsPanelProps) {
  const [apiKeyInput, setApiKeyInput] = useState(() => getUserApiKey() ?? "");
  const [activeApiKey, setActiveApiKey] = useState<string | null>(() => getUserApiKey());
  const [copyFeedback, setCopyFeedback] = useState<string | null>(null);
  const [apiKeyFeedback, setApiKeyFeedback] = useState<string | null>(null);

  async function handleCopyRunId(): Promise<void> {
    if (!runId) {
      return;
    }
    try {
      await navigator.clipboard.writeText(runId);
      setCopyFeedback("Copied.");
    } catch {
      setCopyFeedback("Clipboard unavailable.");
    }
  }

  function handleApplyApiKey(): void {
    const cleaned = apiKeyInput.trim();
    setUserApiKey(cleaned);
    setActiveApiKey(cleaned || null);
    if (cleaned) {
      setApiKeyFeedback("Custom OpenRouter key applied for this tab.");
      return;
    }
    setApiKeyFeedback("Custom OpenRouter key cleared.");
  }

  function handleClearApiKey(): void {
    setApiKeyInput("");
    setUserApiKey(null);
    setActiveApiKey(null);
    setApiKeyFeedback("Custom OpenRouter key cleared.");
  }

  return (
    <div className="space-y-4 rounded-md border border-slate-200 bg-slate-50 p-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="text-sm font-medium text-slate-900">Developer Tools</p>
          <p className="text-xs text-slate-600">Dev-only controls. Mode stays persistent in this browser.</p>
        </div>
        <Badge variant="outline">Dev Only</Badge>
      </div>

      <div className="space-y-2 rounded-md border border-slate-200 bg-white p-3">
        <div className="flex items-center justify-between gap-2">
          <Label className="text-xs uppercase tracking-[0.16em] text-slate-500">Run ID</Label>
          <Button
            type="button"
            size="sm"
            variant="outline"
            onClick={() => void handleCopyRunId()}
            disabled={!runId}
          >
            <Copy className="h-3.5 w-3.5" />
            Copy
          </Button>
        </div>
        <p className="rounded bg-slate-50 px-2 py-1 font-mono text-xs text-slate-700">
          {runId ?? "No active run loaded."}
        </p>
        {copyFeedback ? <p className="text-xs text-slate-500">{copyFeedback}</p> : null}
      </div>

      <div className="space-y-3 rounded-md border border-slate-200 bg-white p-3">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <KeyRound className="h-4 w-4 text-slate-500" />
            <Label htmlFor="user-api-key" className="text-sm font-medium text-slate-900">
              OpenRouter API Key (Optional)
            </Label>
          </div>
          {activeApiKey ? <Badge variant="secondary">{maskApiKey(activeApiKey)}</Badge> : null}
        </div>

        <Input
          id="user-api-key"
          type="password"
          autoComplete="off"
          placeholder="sk-or-v1-..."
          value={apiKeyInput}
          onChange={(event) => setApiKeyInput(event.target.value)}
        />

        <div className="flex flex-wrap gap-2">
          <Button type="button" size="sm" onClick={handleApplyApiKey}>
            Use This Key
          </Button>
          <Button type="button" size="sm" variant="outline" onClick={handleClearApiKey}>
            Clear
          </Button>
        </div>

        {apiKeyFeedback ? (
          <p className="text-xs text-slate-500">{apiKeyFeedback}</p>
        ) : (
          <p className="text-xs text-slate-500">Session-only override. This key is not stored in localStorage.</p>
        )}
        {apiKeyIssue ? (
          <p className="text-xs text-amber-700">
            Recent run activity suggests an API key or authorization issue.
          </p>
        ) : null}
      </div>
    </div>
  );
}
