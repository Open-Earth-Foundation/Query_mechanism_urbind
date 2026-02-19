"use client";

import { useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import { ArrowLeft, Loader2, RefreshCw, Sparkles } from "lucide-react";

import {
  AssumptionsDiscoverResponse,
  MissingDataItem,
  applyRunAssumptions,
  discoverRunAssumptions,
} from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Textarea } from "@/components/ui/textarea";

interface AssumptionsWorkspaceProps {
  runId: string;
  enabled: boolean;
  onClose: () => void;
}

interface EditableMissingDataItem {
  city: string;
  missing_description: string;
  proposed_number_input: string;
}

function toEditableItem(item: MissingDataItem): EditableMissingDataItem {
  return {
    city: item.city,
    missing_description: item.missing_description,
    proposed_number_input:
      item.proposed_number === null || item.proposed_number === undefined
        ? ""
        : String(item.proposed_number),
  };
}

function _looksLikePlainNumber(value: string): boolean {
  return /^[-+]?\d+(\.\d+)?$/.test(value);
}

export function AssumptionsWorkspace({
  runId,
  enabled,
  onClose,
}: AssumptionsWorkspaceProps) {
  const [isDiscovering, setIsDiscovering] = useState(false);
  const [isRegenerating, setIsRegenerating] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [discoverPayload, setDiscoverPayload] =
    useState<AssumptionsDiscoverResponse | null>(null);
  const [editableItems, setEditableItems] = useState<EditableMissingDataItem[]>([]);
  const [rewriteInstructions, setRewriteInstructions] = useState("");
  const [revisedContent, setRevisedContent] = useState<string | null>(null);
  const [revisedOutputPath, setRevisedOutputPath] = useState<string | null>(null);
  const [assumptionsPath, setAssumptionsPath] = useState<string | null>(null);

  const groupedItems = useMemo(() => {
    const grouped = new Map<
      string,
      Array<{ index: number; item: EditableMissingDataItem }>
    >();
    editableItems.forEach((item, index) => {
      const city = item.city.trim();
      const current = grouped.get(city) ?? [];
      current.push({ index, item });
      grouped.set(city, current);
    });
    return [...grouped.entries()].sort(([left], [right]) =>
      left.localeCompare(right),
    );
  }, [editableItems]);

  async function handleDiscover(): Promise<void> {
    if (!enabled || !runId || isDiscovering) {
      return;
    }
    setIsDiscovering(true);
    setErrorMessage(null);
    setRevisedContent(null);
    setRevisedOutputPath(null);
    setAssumptionsPath(null);
    try {
      const payload = await discoverRunAssumptions(runId);
      setDiscoverPayload(payload);
      setEditableItems(payload.items.map(toEditableItem));
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : "Failed to discover missing data.",
      );
    } finally {
      setIsDiscovering(false);
    }
  }

  function updateItemNumber(index: number, value: string): void {
    setEditableItems((current) => {
      const next = [...current];
      next[index] = { ...next[index], proposed_number_input: value };
      return next;
    });
  }

  function buildAssumptionsPayload(): MissingDataItem[] {
    return editableItems.map((item) => {
      const trimmed = item.proposed_number_input.trim();
      if (!trimmed) {
        return {
          city: item.city,
          missing_description: item.missing_description,
          proposed_number: null,
        };
      }
      const parsed = Number(trimmed);
      const proposedValue =
        _looksLikePlainNumber(trimmed) && Number.isFinite(parsed)
          ? parsed
          : trimmed;
      return {
        city: item.city,
        missing_description: item.missing_description,
        proposed_number: proposedValue,
      };
    });
  }

  async function handleRegenerate(): Promise<void> {
    if (!enabled || !runId || isRegenerating || editableItems.length === 0) {
      return;
    }
    setIsRegenerating(true);
    setErrorMessage(null);
    try {
      const items = buildAssumptionsPayload();
      const response = await applyRunAssumptions(runId, {
        items,
        rewrite_instructions: rewriteInstructions.trim() || undefined,
        persist_artifacts: false,
      });
      setRevisedContent(response.revised_content);
      setRevisedOutputPath(response.revised_output_path ?? null);
      setAssumptionsPath(response.assumptions_path ?? null);
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : "Failed to regenerate document.",
      );
    } finally {
      setIsRegenerating(false);
    }
  }

  const canRegenerate = enabled && editableItems.length > 0 && !isRegenerating;

  return (
    <Card className="border-slate-300">
      <CardHeader className="pb-4">
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-teal-600" />
              Assumptions Review
            </CardTitle>
            <CardDescription>
              Two-pass missing data extraction with editable assumptions before regeneration.
            </CardDescription>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Button type="button" size="sm" variant="outline" onClick={onClose}>
              <ArrowLeft className="h-4 w-4" />
              Back to Document
            </Button>
            <Button
              type="button"
              size="sm"
              onClick={() => void handleDiscover()}
              disabled={!enabled || !runId || isDiscovering}
            >
              {isDiscovering ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
              Find Missing Data
            </Button>
          </div>
        </div>

        {discoverPayload ? (
          <div className="rounded-md border border-slate-200 bg-slate-50 p-3 text-xs text-slate-700">
            <p>
              Pass 1: {discoverPayload.verification_summary.first_pass_count} | Pass 2 additions:{" "}
              {discoverPayload.verification_summary.added_in_verification} | Final:{" "}
              {discoverPayload.verification_summary.merged_count}
            </p>
          </div>
        ) : null}
      </CardHeader>

      <CardContent className="space-y-4">
        {errorMessage ? <p className="text-sm text-red-600">{errorMessage}</p> : null}

        {editableItems.length === 0 ? (
          <div className="rounded-md border border-dashed border-slate-300 bg-white p-6 text-sm text-slate-600">
            Click <span className="font-medium">Find Missing Data</span> to run extraction and verification.
          </div>
        ) : (
          <ScrollArea className="h-[38vh] rounded-md border border-slate-200 p-3">
            <div className="space-y-4">
              {groupedItems.map(([city, entries]) => (
                <section key={city} className="space-y-2">
                  <div className="flex items-center gap-2">
                    <Badge variant="outline">{city}</Badge>
                    <span className="text-xs text-slate-500">{entries.length} items</span>
                  </div>
                  <div className="space-y-2">
                    {entries.map(({ index, item }) => {
                      return (
                        <div
                          key={`${item.city}-${item.missing_description}-${index}`}
                          className="rounded-md border border-slate-200 bg-white p-3"
                        >
                          <p className="mb-2 text-sm text-slate-800">{item.missing_description}</p>
                          <Label className="mb-1 block text-xs text-slate-600">
                            Proposed value (number or text)
                          </Label>
                          <Input
                            value={item.proposed_number_input}
                            onChange={(event) =>
                              updateItemNumber(index, event.target.value)
                            }
                            placeholder="Example: 1200 or about 90%"
                          />
                        </div>
                      );
                    })}
                  </div>
                </section>
              ))}
            </div>
          </ScrollArea>
        )}

        <div className="space-y-2">
          <Label htmlFor="rewrite-instructions">Optional rewrite instructions</Label>
          <Textarea
            id="rewrite-instructions"
            value={rewriteInstructions}
            onChange={(event) => setRewriteInstructions(event.target.value)}
            placeholder="Example: keep answer shorter and emphasize budget uncertainty."
            className="min-h-24"
          />
        </div>

        <Button
          type="button"
          onClick={() => void handleRegenerate()}
          disabled={!canRegenerate}
        >
          {isRegenerating ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
          Regenerate
        </Button>
        <p className="text-xs text-slate-500">
          Assumptions are ephemeral by default and are not stored in run artifacts.
        </p>

        {revisedContent ? (
          <div className="space-y-2">
            <p className="text-xs text-slate-500">
              Revised output: {revisedOutputPath}
              {assumptionsPath ? ` | Assumptions: ${assumptionsPath}` : ""}
            </p>
            <article className="document-markdown rounded-md border border-slate-200 bg-white p-5 shadow-inner">
              <ReactMarkdown>{revisedContent}</ReactMarkdown>
            </article>
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
