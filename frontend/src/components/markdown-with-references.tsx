"use client";

import {
  Children,
  MouseEvent,
  ReactNode,
  isValidElement,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { createPortal } from "react-dom";
import { ChevronRight } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import {
  ChatCitation,
  RunReferenceListItem,
  fetchRunReferences,
} from "@/lib/api";
import { formatCityLabel } from "@/lib/utils";

interface ReferencePopoverState {
  refId: string;
  sourceRunId: string | null;
  sourceRefId: string | null;
  top: number;
  left: number;
}

interface CitationPointer {
  cityName: string;
  sourceRunId: string | null;
  sourceRefId: string | null;
}

interface MarkdownWithReferencesProps {
  content: string;
  runId: string | null;
  className?: string;
  chatCitations?: ChatCitation[] | null;
  prefetchRunReferences?: boolean;
}

const REFERENCE_TOKEN_PATTERN = /\[(ref_\d+)\](?!\()|(?<![\w[/])(ref_\d+)\b/g;
const REFERENCE_HREF_PREFIX = "#ref-";
const POPOVER_WIDTH_PX = 360;
const POPOVER_MARGIN_PX = 12;
const POPOVER_ESTIMATED_HEIGHT_PX = 280;

interface CitationGroupToggleProps {
  citations: ReactNode[];
}

function CitationGroupToggle({ citations }: CitationGroupToggleProps) {
  const [isOpen, setIsOpen] = useState(false);
  const label = citations.length > 1 ? "Sources" : "Source";
  return (
    <span className="citation-group">
      <button
        type="button"
        className={`citation-group-toggle${isOpen ? " citation-group-toggle-open" : ""}`}
        aria-expanded={isOpen}
        aria-label={isOpen ? `Hide ${label.toLowerCase()}` : `Show ${label.toLowerCase()}`}
        onClick={() => setIsOpen((current) => !current)}
      >
        {!isOpen ? <span className="citation-group-label">{label}</span> : null}
        <ChevronRight
          className="citation-group-arrow"
          aria-hidden="true"
        />
      </button>
      {isOpen ? (
        <span className="citation-group-list">
          {citations.map((citation, index) => (
            <span key={index} className="citation-group-item">
              {citation}
            </span>
          ))}
        </span>
      ) : null}
    </span>
  );
}

function _isWhitespaceTextNode(node: ReactNode): boolean {
  return typeof node === "string" && node.trim().length === 0;
}

function _isCitationRefNode(node: ReactNode): boolean {
  if (!isValidElement(node)) {
    return false;
  }
  const props = node.props as Record<string, unknown> | undefined;
  const marker = props?.["data-reference-link"];
  if (marker === "true" || marker === true) {
    return true;
  }
  const href = typeof props?.href === "string" ? props.href : "";
  return href.startsWith(REFERENCE_HREF_PREFIX);
}

function _isCitationSeparatorNode(node: ReactNode): boolean {
  if (_isWhitespaceTextNode(node)) {
    return true;
  }
  if (typeof node !== "string") {
    return false;
  }
  return /^[\s,;|/._-]*$/.test(node);
}

function _collapseCitationRuns(children: ReactNode): ReactNode[] {
  const nodes = Children.toArray(children);
  const collapsed: ReactNode[] = [];
  let index = 0;

  while (index < nodes.length) {
    const current = nodes[index];
    if (!_isCitationRefNode(current)) {
      collapsed.push(current);
      index += 1;
      continue;
    }

    const run: ReactNode[] = [current];
    let cursor = index + 1;
    while (cursor < nodes.length) {
      const next = nodes[cursor];
      if (
        _isCitationSeparatorNode(next) &&
        cursor + 1 < nodes.length &&
        _isCitationRefNode(nodes[cursor + 1])
      ) {
        run.push(nodes[cursor + 1]);
        cursor += 2;
        continue;
      }
      if (_isCitationRefNode(next)) {
        run.push(next);
        cursor += 1;
        continue;
      }
      break;
    }

    if (run.length > 1) {
      collapsed.push(
        <CitationGroupToggle
          key={`citation-group-${index}`}
          citations={run}
        />,
      );
    } else {
      collapsed.push(current);
    }
    index = cursor;
  }

  return collapsed;
}

function _toPlainText(node: ReactNode): string {
  if (typeof node === "string" || typeof node === "number") {
    return String(node);
  }
  if (Array.isArray(node)) {
    return node.map(_toPlainText).join("");
  }
  if (node && typeof node === "object" && "props" in node) {
    const childNode = (node as { props?: { children?: ReactNode } }).props?.children;
    return _toPlainText(childNode);
  }
  return "";
}

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

function _buildReferenceCacheKey(
  runId: string | null,
  refId: string | null,
): string {
  return `${runId ?? "__no_run__"}::${refId ?? "__no_ref__"}`;
}

function _referenceItemToPointer(
  item: RunReferenceListItem,
  runId: string,
): CitationPointer {
  return {
    cityName: formatCityLabel(item.city_name ?? "") || "Source",
    sourceRunId: runId,
    sourceRefId: item.ref_id,
  };
}

export function MarkdownWithReferences({
  content,
  runId,
  className,
  chatCitations,
  prefetchRunReferences = true,
}: MarkdownWithReferencesProps) {
  const popoverRef = useRef<HTMLDivElement | null>(null);
  const activePopoverRef = useRef<ReferencePopoverState | null>(null);
  const activeRequestKeyRef = useRef<string | null>(null);
  const [referenceCache, setReferenceCache] = useState<
    Record<string, RunReferenceListItem>
  >({});
  const [runReferencePointers, setRunReferencePointers] = useState<
    Record<string, CitationPointer>
  >({});
  const [activePopover, setActivePopover] = useState<ReferencePopoverState | null>(
    null,
  );
  const [loadingRequestKey, setLoadingRequestKey] = useState<string | null>(null);
  const [referenceError, setReferenceError] = useState<string | null>(null);

  const markdownContent = useMemo(() => _toReferenceMarkdown(content), [content]);

  const chatCitationPointers = useMemo(() => {
    const mapping: Record<string, CitationPointer> = {};
    (chatCitations ?? []).forEach((citation) => {
      const refId = citation.ref_id?.trim();
      if (!refId) {
        return;
      }
      mapping[refId] = {
        cityName: formatCityLabel(citation.city_name ?? "") || "Source",
        sourceRunId: citation.source_run_id?.trim() || runId,
        sourceRefId: citation.source_ref_id?.trim() || refId,
      };
    });
    return mapping;
  }, [chatCitations, runId]);

  const citationPointers = useMemo(
    () => ({ ...runReferencePointers, ...chatCitationPointers }),
    [chatCitationPointers, runReferencePointers],
  );

  const activeReferenceCacheKey =
    activePopover !== null
      ? _buildReferenceCacheKey(activePopover.sourceRunId, activePopover.sourceRefId)
      : null;
  const activeReference =
    activeReferenceCacheKey !== null
      ? referenceCache[activeReferenceCacheKey]
      : undefined;

  useEffect(() => {
    activePopoverRef.current = activePopover;
  }, [activePopover]);

  useEffect(() => {
    activeRequestKeyRef.current = null;
    setActivePopover(null);
    setLoadingRequestKey(null);
    setReferenceError(null);
    setReferenceCache({});
    setRunReferencePointers({});
  }, [runId, chatCitations]);

  useEffect(() => {
    if (!prefetchRunReferences || !runId) {
      return;
    }
    let cancelled = false;
    fetchRunReferences(runId)
      .then((payload) => {
        if (cancelled) {
          return;
        }
        const pointers: Record<string, CitationPointer> = {};
        payload.references.forEach((item) => {
          pointers[item.ref_id] = _referenceItemToPointer(item, runId);
        });
        setRunReferencePointers(pointers);
      })
      .catch(() => {
        if (!cancelled) {
          setRunReferencePointers({});
        }
      });
    return () => {
      cancelled = true;
    };
  }, [prefetchRunReferences, runId]);

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

    const pointer = citationPointers[refId];
    const sourceRunId = pointer?.sourceRunId ?? runId;
    const sourceRefId = pointer?.sourceRefId ?? refId;

    if (activePopover?.refId === refId) {
      setActivePopover(null);
      setReferenceError(null);
      return;
    }

    const targetRect = event.currentTarget.getBoundingClientRect();
    setActivePopover({
      refId,
      sourceRunId,
      sourceRefId,
      top: _resolvePopoverTop(targetRect.top, targetRect.bottom),
      left: _resolvePopoverLeft(targetRect.left),
    });
    setReferenceError(null);
    const requestKey = _buildReferenceCacheKey(sourceRunId, sourceRefId);

    if (!sourceRunId || !sourceRefId) {
      setReferenceError("Reference source is missing, so this quote cannot be loaded.");
      return;
    }
    if (referenceCache[requestKey]) {
      return;
    }

    activeRequestKeyRef.current = requestKey;
    setLoadingRequestKey(requestKey);
    try {
      const payload = await fetchRunReferences(sourceRunId, {
        refId: sourceRefId,
        includeQuote: true,
      });
      const reference = payload.references[0];
      if (!reference) {
        throw new Error("Reference was not found.");
      }
      setReferenceCache((current) => ({ ...current, [requestKey]: reference }));
    } catch (error) {
      const activePopoverState = activePopoverRef.current;
      const activePopoverKey =
        activePopoverState !== null
          ? _buildReferenceCacheKey(
              activePopoverState.sourceRunId,
              activePopoverState.sourceRefId,
            )
          : null;
      if (
        activeRequestKeyRef.current === requestKey &&
        activePopoverKey === requestKey
      ) {
        setReferenceError(
          error instanceof Error ? error.message : "Failed to load reference.",
        );
      }
    } finally {
      if (activeRequestKeyRef.current === requestKey) {
        activeRequestKeyRef.current = null;
      }
      setLoadingRequestKey((current) =>
        current === requestKey ? null : current,
      );
    }
  }

  function renderReferenceLabel(refId: string): string {
    const pointer = citationPointers[refId];
    if (pointer?.cityName) {
      return pointer.cityName;
    }
    return "Source";
  }

  return (
    <>
      <div className={className}>
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            h1: ({ children }) => {
              const isQuestionHeading =
                _toPlainText(children).trim().toLowerCase() === "question";
              if (isQuestionHeading) {
                return <h1 className="document-question-heading">{children}</h1>;
              }
              return <h1>{children}</h1>;
            },
            p: ({ children }) => <p>{_collapseCitationRuns(children)}</p>,
            li: ({ children }) => <li>{_collapseCitationRuns(children)}</li>,
            table: ({ children }) => (
              <div className="markdown-table-wrap">
                <table className="markdown-table">{children}</table>
              </div>
            ),
            th: ({ children }) => <th className="markdown-table-head">{children}</th>,
            td: ({ children }) => (
              <td className="markdown-table-cell">{_collapseCitationRuns(children)}</td>
            ),
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
                    title={refId}
                  >
                    <span className="citation-ref-city">{renderReferenceLabel(refId)}</span>
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

              {loadingRequestKey === activeReferenceCacheKey && !activeReference ? (
                <p className="citation-popover-muted">Loading quote...</p>
              ) : null}

              {referenceError ? (
                <p className="citation-popover-error">{referenceError}</p>
              ) : null}

              {!referenceError && activeReference ? (
                <div className="citation-popover-body">
                  <p>{activeReference.quote?.trim() || "(empty quote)"}</p>
                </div>
              ) : null}
            </div>,
            document.body,
          )
        : null}
    </>
  );
}
