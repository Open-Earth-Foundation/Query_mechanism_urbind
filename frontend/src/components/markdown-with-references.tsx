"use client";

import { MouseEvent, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { RunReferenceResponse, fetchRunReference } from "@/lib/api";

interface ReferencePopoverState {
  refId: string;
  top: number;
  left: number;
}

interface MarkdownWithReferencesProps {
  content: string;
  runId: string | null;
  className?: string;
}

const REFERENCE_TOKEN_PATTERN = /\[(ref_\d+)\](?!\()|(?<![\w[/])(ref_\d+)\b/g;
const REFERENCE_HREF_PREFIX = "#ref-";
const POPOVER_WIDTH_PX = 360;
const POPOVER_MARGIN_PX = 12;
const POPOVER_ESTIMATED_HEIGHT_PX = 280;

function _toReferenceMarkdown(content: string): string {
  return content.replace(
    REFERENCE_TOKEN_PATTERN,
    (_match, bracketedRefId: string | undefined, bareRefId: string | undefined) => {
      const refId = bracketedRefId ?? bareRefId;
      if (!refId) {
        return _match;
      }
      return `[${refId}](${REFERENCE_HREF_PREFIX}${refId})`;
    },
  );
}

function _resolvePopoverLeft(left: number): number {
  const maxLeft = Math.max(
    window.innerWidth - POPOVER_WIDTH_PX - POPOVER_MARGIN_PX,
    POPOVER_MARGIN_PX,
  );
  return Math.min(Math.max(left, POPOVER_MARGIN_PX), maxLeft);
}

function _resolvePopoverTop(top: number, bottom: number): number {
  const preferredTop = bottom + 8;
  const maxTop = Math.max(
    window.innerHeight - POPOVER_ESTIMATED_HEIGHT_PX - POPOVER_MARGIN_PX,
    POPOVER_MARGIN_PX,
  );
  if (preferredTop <= maxTop) {
    return preferredTop;
  }
  return Math.max(top - POPOVER_ESTIMATED_HEIGHT_PX - 8, POPOVER_MARGIN_PX);
}

export function MarkdownWithReferences({
  content,
  runId,
  className,
}: MarkdownWithReferencesProps) {
  const popoverRef = useRef<HTMLDivElement | null>(null);
  const [referenceCache, setReferenceCache] = useState<
    Record<string, RunReferenceResponse>
  >({});
  const [activePopover, setActivePopover] = useState<ReferencePopoverState | null>(
    null,
  );
  const [loadingRefId, setLoadingRefId] = useState<string | null>(null);
  const [referenceError, setReferenceError] = useState<string | null>(null);

  const markdownContent = useMemo(() => _toReferenceMarkdown(content), [content]);
  const activeReference =
    activePopover !== null ? referenceCache[activePopover.refId] : undefined;

  useEffect(() => {
    if (!activePopover) {
      return;
    }

    function handleDocumentClick(event: globalThis.MouseEvent): void {
      const target = event.target;
      if (!(target instanceof HTMLElement)) {
        return;
      }
      if (popoverRef.current?.contains(target)) {
        return;
      }
      if (target.closest("[data-reference-link='true']")) {
        return;
      }
      setActivePopover(null);
      setReferenceError(null);
    }

    function handleKeydown(event: KeyboardEvent): void {
      if (event.key !== "Escape") {
        return;
      }
      setActivePopover(null);
      setReferenceError(null);
    }

    document.addEventListener("mousedown", handleDocumentClick);
    document.addEventListener("keydown", handleKeydown);
    return () => {
      document.removeEventListener("mousedown", handleDocumentClick);
      document.removeEventListener("keydown", handleKeydown);
    };
  }, [activePopover]);

  async function handleReferenceClick(
    event: MouseEvent<HTMLButtonElement>,
    refId: string,
  ): Promise<void> {
    event.preventDefault();

    if (activePopover?.refId === refId) {
      setActivePopover(null);
      setReferenceError(null);
      return;
    }

    const targetRect = event.currentTarget.getBoundingClientRect();
    setActivePopover({
      refId,
      top: _resolvePopoverTop(targetRect.top, targetRect.bottom),
      left: _resolvePopoverLeft(targetRect.left),
    });
    setReferenceError(null);

    if (!runId) {
      setReferenceError("Run id is missing, so this reference cannot be loaded.");
      return;
    }
    if (referenceCache[refId]) {
      return;
    }

    setLoadingRefId(refId);
    try {
      const reference = await fetchRunReference(runId, refId);
      setReferenceCache((current) => ({ ...current, [refId]: reference }));
    } catch (error) {
      setReferenceError(
        error instanceof Error ? error.message : "Failed to load reference.",
      );
    } finally {
      setLoadingRefId((current) => (current === refId ? null : current));
    }
  }

  return (
    <>
      <div className={className}>
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            a: ({ href, children }) => {
              if (
                typeof href === "string" &&
                href.startsWith(REFERENCE_HREF_PREFIX)
              ) {
                const refId = href.slice(REFERENCE_HREF_PREFIX.length);
                return (
                  <button
                    type="button"
                    className="citation-ref-link"
                    data-reference-link="true"
                    onClick={(event) => void handleReferenceClick(event, refId)}
                  >
                    {children}
                  </button>
                );
              }
              return (
                <a
                  href={href}
                  target={/^https?:\/\//i.test(href ?? "") ? "_blank" : undefined}
                  rel={/^https?:\/\//i.test(href ?? "") ? "noreferrer" : undefined}
                >
                  {children}
                </a>
              );
            },
          }}
        >
          {markdownContent}
        </ReactMarkdown>
      </div>

      {activePopover
        ? createPortal(
            <div
              ref={popoverRef}
              className="citation-popover"
              style={{ top: `${activePopover.top}px`, left: `${activePopover.left}px` }}
            >
              <div className="citation-popover-header">
                <strong>{activePopover.refId}</strong>
                <button
                  type="button"
                  className="citation-popover-close"
                  onClick={() => {
                    setActivePopover(null);
                    setReferenceError(null);
                  }}
                >
                  Close
                </button>
              </div>

              {loadingRefId === activePopover.refId && !activeReference ? (
                <p className="citation-popover-muted">Loading reference...</p>
              ) : null}

              {referenceError ? (
                <p className="citation-popover-error">{referenceError}</p>
              ) : null}

              {!referenceError && activeReference ? (
                <div className="citation-popover-body">
                  <p>
                    <span className="citation-popover-label">City:</span>{" "}
                    {activeReference.city_name || "(unknown)"}
                  </p>
                  <p>
                    <span className="citation-popover-label">Quote:</span>{" "}
                    {activeReference.quote || "(empty quote)"}
                  </p>
                </div>
              ) : null}
            </div>,
            document.body,
          )
        : null}
    </>
  );
}
