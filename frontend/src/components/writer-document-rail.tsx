"use client";

import { Maximize2, ScrollText } from "lucide-react";

import { DocumentExportControls } from "@/components/document-export-controls";
import { MarkdownWithReferences } from "@/components/markdown-with-references";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";

interface WriterDocumentRailProps {
  runId: string;
  content: string;
  question?: string | null;
  statusLabel?: string | null;
  onOpenFullDocument: () => void;
}

export function WriterDocumentRail({
  runId,
  content,
  question,
  statusLabel,
  onOpenFullDocument,
}: WriterDocumentRailProps) {
  return (
    <div className="space-y-4">
      <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
        <div className="flex items-start justify-between gap-3">
          <div className="space-y-2">
            <div className="inline-flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">
              <ScrollText className="h-3.5 w-3.5" />
              Writer Document
            </div>
            <p className="text-sm leading-relaxed text-slate-700">
              {question?.trim()
                ? question.trim()
                : "Keep the generated answer visible here while you question it in chat."}
            </p>
          </div>
          {statusLabel ? <Badge variant="outline">{statusLabel}</Badge> : null}
        </div>

        <div className="mt-4 flex flex-wrap items-center justify-between gap-2">
          <Badge variant="secondary">Run {runId.slice(0, 8)}</Badge>
          <div className="flex flex-wrap items-center justify-end gap-2">
            <DocumentExportControls runId={runId} content={content} />
            <Button type="button" size="sm" variant="outline" onClick={onOpenFullDocument}>
              <Maximize2 className="h-4 w-4" />
              Open Full View
            </Button>
          </div>
        </div>
      </div>

      <ScrollArea className="h-[62vh] rounded-xl border border-slate-200 bg-white">
        <article className="document-markdown document-markdown-rail p-4">
          <MarkdownWithReferences content={content} runId={runId} />
        </article>
      </ScrollArea>
    </div>
  );
}
