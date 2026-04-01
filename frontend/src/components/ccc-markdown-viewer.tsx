"use client";

import {
  memo,
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  useTransition,
} from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { Button } from "@/components/ui/button";

export interface CccSourceSection {
  id: string;
  content: string;
  headingText: string | null;
  headingDepth: number | null;
  charCount: number;
}

interface CccMarkdownViewerProps {
  content: string;
}

const SECTION_MAX_CHARS = 20_000;
const INITIAL_BATCH_CHAR_BUDGET = 100_000;
const INITIAL_BATCH_MIN_SECTIONS = 8;
const LOAD_MORE_CHAR_BUDGET = 60_000;
const LOAD_MORE_MIN_SECTIONS = 5;
const OBSERVER_ROOT_MARGIN = "480px 0px";
const MARKDOWN_COMPONENTS = {
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
const MARKDOWN_PLUGINS = [remarkGfm];
const parsedSectionCache = new Map<string, CccSourceSection[]>();

const CccMarkdownSection = memo(function CccMarkdownSection({
  content,
}: {
  content: string;
}) {
  return (
    <ReactMarkdown remarkPlugins={MARKDOWN_PLUGINS} components={MARKDOWN_COMPONENTS}>
      {content}
    </ReactMarkdown>
  );
});

function trimBoundaryNewlines(value: string): string {
  return value.replace(/^\n+|\n+$/g, "");
}

function normalizeSectionContent(value: string): string {
  return trimBoundaryNewlines(value).trimEnd();
}

function buildSection(
  id: string,
  content: string,
  headingText: string | null,
  headingDepth: number | null,
): CccSourceSection {
  return {
    id,
    content,
    headingText,
    headingDepth,
    charCount: content.length,
  };
}

function splitRawChunkByLines(chunk: string, maxChars: number): string[] {
  if (chunk.length <= maxChars) {
    return [chunk];
  }

  const lines = chunk.split("\n");
  const parts: string[] = [];
  let currentLines: string[] = [];
  let currentLength = 0;

  const flush = (): void => {
    if (currentLines.length === 0) {
      return;
    }
    parts.push(currentLines.join("\n"));
    currentLines = [];
    currentLength = 0;
  };

  for (const line of lines) {
    if (line.length > maxChars) {
      flush();
      for (let start = 0; start < line.length; start += maxChars) {
        parts.push(line.slice(start, start + maxChars));
      }
      continue;
    }

    const nextLength = currentLength === 0 ? line.length : currentLength + line.length + 1;
    if (currentLines.length > 0 && nextLength > maxChars) {
      flush();
    }

    currentLines.push(line);
    currentLength = currentLength === 0 ? line.length : currentLength + line.length + 1;
  }

  flush();
  return parts;
}

function splitOversizedBlock(block: string, maxChars: number): string[] {
  if (block.length <= maxChars) {
    return [block];
  }
  return splitRawChunkByLines(block, maxChars);
}

function splitIntoBlocks(content: string): string[] {
  const normalizedContent = normalizeSectionContent(content);
  if (!normalizedContent) {
    return [];
  }

  return normalizedContent
    .split(/\n{2,}/)
    .map((block) => normalizeSectionContent(block))
    .filter((block) => block.length > 0)
    .flatMap((block) => splitOversizedBlock(block, SECTION_MAX_CHARS));
}

function splitLargeSection(section: CccSourceSection): CccSourceSection[] {
  if (section.charCount <= SECTION_MAX_CHARS) {
    return [section];
  }

  const blocks = splitIntoBlocks(section.content);
  if (blocks.length === 0) {
    return [section];
  }

  const chunks: string[] = [];
  let currentBlocks: string[] = [];
  let currentLength = 0;

  const flush = (): void => {
    if (currentBlocks.length === 0) {
      return;
    }
    chunks.push(currentBlocks.join("\n\n"));
    currentBlocks = [];
    currentLength = 0;
  };

  blocks.forEach((block) => {
    const nextLength = currentLength === 0 ? block.length : currentLength + block.length + 2;
    if (currentBlocks.length > 0 && nextLength > SECTION_MAX_CHARS) {
      flush();
    }
    currentBlocks.push(block);
    currentLength = currentLength === 0 ? block.length : currentLength + block.length + 2;
  });

  flush();

  return chunks.map((chunk, index) =>
    buildSection(
      `${section.id}-part-${index + 1}`,
      chunk,
      section.headingText,
      section.headingDepth,
    ),
  );
}

function parseHeadingSection(id: string, content: string): CccSourceSection[] {
  const normalizedContent = normalizeSectionContent(content);
  if (!normalizedContent) {
    return [];
  }

  const [firstLine = ""] = normalizedContent.split("\n", 1);
  const headingMatch = firstLine.match(/^(#{1,6})\s+(.+)$/);
  const headingDepth = headingMatch ? headingMatch[1].length : null;
  const headingText = headingMatch ? headingMatch[2].trim() : null;

  return splitLargeSection(
    buildSection(id, normalizedContent, headingText, headingDepth),
  );
}

function buildCccSections(content: string): CccSourceSection[] {
  const cached = parsedSectionCache.get(content);
  if (cached) {
    return cached;
  }

  const sections: CccSourceSection[] = [];
  const headingMatches = Array.from(content.matchAll(/^(#{1,6})\s+(.+)$/gm));

  if (headingMatches.length === 0) {
    const parsedSections = parseHeadingSection("section-1", content);
    parsedSectionCache.set(content, parsedSections);
    return parsedSections;
  }

  const firstHeadingIndex = headingMatches[0].index ?? 0;
  const preamble = normalizeSectionContent(content.slice(0, firstHeadingIndex));
  if (preamble) {
    sections.push(...parseHeadingSection("preamble", preamble));
  }

  headingMatches.forEach((match, index) => {
    const start = match.index ?? 0;
    const end =
      headingMatches[index + 1]?.index ?? content.length;
    sections.push(...parseHeadingSection(`section-${index + 1}`, content.slice(start, end)));
  });

  parsedSectionCache.set(content, sections);
  return sections;
}

function resolveNextSectionCount(
  sections: CccSourceSection[],
  currentCount: number,
  minSections: number,
  charBudget: number,
): number {
  if (currentCount >= sections.length) {
    return sections.length;
  }

  let nextCount = currentCount;
  let consumedChars = 0;

  while (nextCount < sections.length) {
    const nextSection = sections[nextCount];
    const shouldAppend =
      nextCount - currentCount < minSections ||
      consumedChars + nextSection.charCount <= charBudget;
    if (!shouldAppend) {
      break;
    }
    consumedChars += nextSection.charCount;
    nextCount += 1;
  }

  if (nextCount === currentCount) {
    return Math.min(sections.length, currentCount + 1);
  }
  return nextCount;
}

export function CccMarkdownViewer({ content }: CccMarkdownViewerProps) {
  const sentinelRef = useRef<HTMLDivElement | null>(null);
  const sections = useMemo(() => buildCccSections(content), [content]);
  const initialVisibleSectionCount = useMemo(
    () =>
      resolveNextSectionCount(
        sections,
        0,
        INITIAL_BATCH_MIN_SECTIONS,
        INITIAL_BATCH_CHAR_BUDGET,
      ),
    [sections],
  );
  const [visibleSectionCount, setVisibleSectionCount] = useState(initialVisibleSectionCount);
  const [supportsIntersectionObserver, setSupportsIntersectionObserver] = useState(false);
  const [canAutoLoadSections, setCanAutoLoadSections] = useState(false);
  const [isAppendingSections, startAppendingTransition] = useTransition();

  useEffect(() => {
    setVisibleSectionCount(initialVisibleSectionCount);
  }, [initialVisibleSectionCount]);

  useEffect(() => {
    setSupportsIntersectionObserver(
      typeof window !== "undefined" && "IntersectionObserver" in window,
    );
  }, []);

  const hasMoreSections = visibleSectionCount < sections.length;
  const visibleSections = sections.slice(0, visibleSectionCount);

  const loadMoreSections = useCallback(() => {
    if (!hasMoreSections) {
      return;
    }

    startAppendingTransition(() => {
      setVisibleSectionCount((currentCount) =>
        resolveNextSectionCount(
          sections,
          currentCount,
          LOAD_MORE_MIN_SECTIONS,
          LOAD_MORE_CHAR_BUDGET,
        ),
      );
    });
  }, [hasMoreSections, sections]);

  useEffect(() => {
    if (!hasMoreSections || !supportsIntersectionObserver) {
      setCanAutoLoadSections(false);
      return;
    }

    const sentinel = sentinelRef.current;
    const root = sentinel?.closest("[data-radix-scroll-area-viewport]");
    if (!(sentinel instanceof HTMLDivElement) || !(root instanceof HTMLElement)) {
      setCanAutoLoadSections(false);
      return;
    }

    setCanAutoLoadSections(true);
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          loadMoreSections();
        }
      },
      {
        root,
        rootMargin: OBSERVER_ROOT_MARGIN,
      },
    );
    observer.observe(sentinel);

    return () => {
      observer.disconnect();
    };
  }, [hasMoreSections, loadMoreSections, supportsIntersectionObserver]);

  return (
    <div className="space-y-4">
      {visibleSections.map((section) => (
        <CccMarkdownSection key={section.id} content={section.content} />
      ))}

      {hasMoreSections ? (
        <div className="flex flex-col items-center gap-3 py-2">
          <div ref={sentinelRef} aria-hidden="true" className="h-px w-full" />
          {!canAutoLoadSections ? (
            <Button type="button" variant="outline" size="sm" onClick={loadMoreSections}>
              Load more
            </Button>
          ) : null}
          {isAppendingSections ? (
            <p className="text-xs text-slate-500">Loading more sections...</p>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
