"use client";

import { type ReactNode, useEffect, useRef, useState } from "react";
import { Check, ChevronDown, Copy, Download, FileText } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { Button } from "@/components/ui/button";
import { downloadRunWordExport } from "@/lib/api";
import { cn } from "@/lib/utils";

interface DocumentExportControlsProps {
  runId: string;
  content: string;
  className?: string;
}

const EXPORT_FEEDBACK_RESET_MS = 2200;
const EXPORT_REFERENCE_PATTERN = /(?:\s*\[ref_[^\]\s]+\](?!\())+/g;
const EXPORT_MARKDOWN_COMPONENTS = {
  table: ({ children }: { children?: ReactNode }) => (
    <div className="markdown-table-wrap">
      <table className="markdown-table">{children}</table>
    </div>
  ),
  th: ({ children }: { children?: ReactNode }) => (
    <th className="markdown-table-head">{children}</th>
  ),
  td: ({ children }: { children?: ReactNode }) => (
    <td className="markdown-table-cell">{children}</td>
  ),
  a: ({
    href,
    children,
  }: {
    href?: string;
    children?: ReactNode;
  }) => (
    <a
      href={href}
      target={/^https?:\/\//i.test(href ?? "") ? "_blank" : undefined}
      rel={/^https?:\/\//i.test(href ?? "") ? "noreferrer" : undefined}
    >
      {children}
    </a>
  ),
  img: () => null,
};
const RICH_CLIPBOARD_STYLES = `
  body {
    margin: 0;
    color: #0f172a;
    font-family: Aptos, Calibri, Segoe UI, sans-serif;
    font-size: 11pt;
    line-height: 1.55;
  }
  article {
    padding: 0;
  }
  h1, h2, h3, h4, h5, h6 {
    color: #0f172a;
    line-height: 1.2;
    margin: 1.1em 0 0.45em;
  }
  p, ul, ol, blockquote, pre {
    margin: 0.6em 0;
  }
  ul, ol {
    padding-left: 1.25em;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    margin: 1em 0;
  }
  th, td {
    border: 1px solid #cbd5e1;
    padding: 6px 8px;
    vertical-align: top;
    text-align: left;
  }
  th {
    background: #f8fafc;
    font-weight: 700;
  }
  code {
    font-family: "Cascadia Code", "SFMono-Regular", Consolas, monospace;
  }
  pre {
    white-space: pre-wrap;
  }
`;

function buildRichClipboardHtml(fragmentHtml: string): string {
  return [
    "<!doctype html>",
    "<html>",
    "<head>",
    '<meta charset="utf-8" />',
    `<style>${RICH_CLIPBOARD_STYLES}</style>`,
    "</head>",
    "<body>",
    `<article>${fragmentHtml}</article>`,
    "</body>",
    "</html>",
  ].join("");
}

function stripExportReferences(markdown: string): string {
  return markdown.replace(EXPORT_REFERENCE_PATTERN, "").trim();
}

function triggerBlobDownload(blob: Blob, filename: string): void {
  const objectUrl = window.URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = objectUrl;
  anchor.download = filename;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => {
    window.URL.revokeObjectURL(objectUrl);
  }, 0);
}

export function DocumentExportControls({
  runId,
  content,
  className,
}: DocumentExportControlsProps) {
  const exportContent = stripExportReferences(content);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [feedback, setFeedback] = useState<string | null>(null);
  const [activeAction, setActiveAction] = useState<"word" | "notion" | null>(null);
  const menuRef = useRef<HTMLDivElement | null>(null);
  const richSnapshotRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!feedback) {
      return;
    }
    const timeoutId = window.setTimeout(() => {
      setFeedback(null);
    }, EXPORT_FEEDBACK_RESET_MS);
    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [feedback]);

  useEffect(() => {
    if (!isMenuOpen) {
      return;
    }
    const handlePointerDown = (event: PointerEvent): void => {
      if (!menuRef.current?.contains(event.target as Node)) {
        setIsMenuOpen(false);
      }
    };
    window.addEventListener("pointerdown", handlePointerDown);
    return () => {
      window.removeEventListener("pointerdown", handlePointerDown);
    };
  }, [isMenuOpen]);

  async function handleCopyMarkdown(): Promise<void> {
    try {
      await navigator.clipboard.writeText(content);
      setFeedback("Markdown copied");
    } catch (error) {
      setFeedback(error instanceof Error ? error.message : "Copy failed.");
    }
  }

  async function handleCopyForNotion(): Promise<void> {
    setActiveAction("notion");
    const richSnapshot = richSnapshotRef.current;
    if (!richSnapshot) {
      setFeedback("Rich copy is not ready yet.");
      setActiveAction(null);
      return;
    }

    const richHtml = buildRichClipboardHtml(richSnapshot.innerHTML);
    const plainText = richSnapshot.textContent?.trim() || exportContent;

    try {
      if (
        typeof ClipboardItem !== "undefined" &&
        typeof navigator.clipboard.write === "function"
      ) {
        await navigator.clipboard.write([
          new ClipboardItem({
            "text/html": new Blob([richHtml], { type: "text/html" }),
            "text/plain": new Blob([plainText], { type: "text/plain" }),
          }),
        ]);
      } else {
        await navigator.clipboard.writeText(plainText);
      }
      setFeedback("Rich copy ready");
      setIsMenuOpen(false);
    } catch (error) {
      setFeedback(error instanceof Error ? error.message : "Rich copy failed.");
    } finally {
      setActiveAction(null);
    }
  }

  async function handleDownloadWord(): Promise<void> {
    setActiveAction("word");
    try {
      const blob = await downloadRunWordExport(runId);
      triggerBlobDownload(blob, `${runId}.docx`);
      setFeedback("Word download ready");
      setIsMenuOpen(false);
    } catch (error) {
      setFeedback(error instanceof Error ? error.message : "Download failed.");
    } finally {
      setActiveAction(null);
    }
  }

  return (
    <div className={cn("flex flex-wrap items-center justify-end gap-2", className)}>
      <div ref={menuRef} className="relative">
        <Button
          type="button"
          size="sm"
          variant="outline"
          onClick={() => setIsMenuOpen((current) => !current)}
          aria-expanded={isMenuOpen}
          aria-label="Open export options"
        >
          Export
          <ChevronDown
            className={cn(
              "h-4 w-4 transition-transform",
              isMenuOpen ? "rotate-180" : "rotate-0",
            )}
          />
        </Button>
        {isMenuOpen ? (
          <div className="absolute right-0 top-full z-30 mt-2 w-56 rounded-xl border border-slate-200 bg-white p-1.5 shadow-lg">
            <button
              type="button"
              className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-sm text-slate-700 transition hover:bg-slate-100"
              onClick={() => void handleDownloadWord()}
              disabled={activeAction === "word"}
            >
              <Download className="h-4 w-4" />
              <span>
                {activeAction === "word" ? "Preparing Word..." : "Download Word (.docx)"}
              </span>
            </button>
            <button
              type="button"
              className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-sm text-slate-700 transition hover:bg-slate-100"
              onClick={() => void handleCopyForNotion()}
              disabled={activeAction === "notion"}
            >
              <FileText className="h-4 w-4" />
              <span>
                {activeAction === "notion" ? "Copying rich text..." : "Copy for Notion"}
              </span>
            </button>
          </div>
        ) : null}
      </div>

      <Button
        type="button"
        size="icon"
        variant="outline"
        onClick={() => void handleCopyMarkdown()}
        aria-label="Copy markdown"
        title="Copy markdown"
      >
        {feedback === "Markdown copied" ? (
          <Check className="h-4 w-4" />
        ) : (
          <Copy className="h-4 w-4" />
        )}
      </Button>

      {feedback ? (
        <span className="text-xs font-medium text-slate-500">{feedback}</span>
      ) : null}

      <div className="hidden" aria-hidden="true">
        <article ref={richSnapshotRef} className="document-markdown">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={EXPORT_MARKDOWN_COMPONENTS}
          >
            {exportContent}
          </ReactMarkdown>
        </article>
      </div>
    </div>
  );
}
