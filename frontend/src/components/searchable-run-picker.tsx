"use client";

import { useEffect, useRef, useState } from "react";
import { Check, ChevronDown, Loader2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { RunSummary } from "@/lib/api";
import { filterImmediateRunMatches } from "@/lib/run-picker-search";
import { cn } from "@/lib/utils";

interface SearchableRunPickerProps {
  id: string;
  runs: RunSummary[];
  selectedRunId: string;
  searchQuery: string;
  onSearchQueryChange: (value: string) => void;
  onSelectRun: (runId: string) => void;
  formatRunLabel: (run: RunSummary) => string;
  disabled?: boolean;
  isLoading?: boolean;
  popupClassName?: string;
  placeholder?: string;
  searchPlaceholder?: string;
}

export function SearchableRunPicker({
  id,
  runs,
  selectedRunId,
  searchQuery,
  onSearchQueryChange,
  onSelectRun,
  formatRunLabel,
  disabled = false,
  isLoading = false,
  popupClassName,
  placeholder = "Select a run",
  searchPlaceholder = "Filter by question, date, run ID, or city",
}: SearchableRunPickerProps) {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const searchInputRef = useRef<HTMLInputElement | null>(null);

  const selectedRun = runs.find((run) => run.run_id === selectedRunId) ?? null;
  const visibleRuns = filterImmediateRunMatches(runs, searchQuery);
  const triggerLabel =
    selectedRun !== null
      ? formatRunLabel(selectedRun)
      : isLoading && runs.length === 0
        ? "Loading runs..."
        : placeholder;

  useEffect(() => {
    if (!isOpen) {
      return;
    }
    searchInputRef.current?.focus();
    searchInputRef.current?.select();
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen) {
      return;
    }

    const handlePointerDown = (event: MouseEvent): void => {
      const target = event.target;
      if (!(target instanceof Node) || containerRef.current?.contains(target)) {
        return;
      }
      setIsOpen(false);
    };

    const handleKeyDown = (event: KeyboardEvent): void => {
      if (event.key === "Escape") {
        setIsOpen(false);
      }
    };

    document.addEventListener("mousedown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [isOpen]);

  const emptyMessage = searchQuery.trim() ? "No matching runs found." : "No runs found.";

  return (
    <div ref={containerRef} className="relative min-w-0 flex-1">
      <Button
        id={id}
        type="button"
        variant="outline"
        className="h-11 w-full justify-between px-3 text-left font-normal"
        aria-expanded={isOpen}
        aria-haspopup="listbox"
        disabled={disabled}
        onClick={() => setIsOpen((current) => !current)}
      >
        <span className="truncate">{triggerLabel}</span>
        <ChevronDown
          className={cn(
            "h-4 w-4 shrink-0 text-slate-500 transition-transform",
            isOpen ? "rotate-180" : null,
          )}
        />
      </Button>

      {isOpen ? (
        <div
          className={cn(
            "absolute left-0 top-full z-20 mt-2 w-full rounded-md border border-slate-200 bg-white p-2 shadow-lg",
            popupClassName,
          )}
        >
          <div className="relative border-b border-slate-100 pb-2">
            <Input
              ref={searchInputRef}
              type="search"
              name={`${id}-search`}
              value={searchQuery}
              onChange={(event) => onSearchQueryChange(event.target.value)}
              placeholder={searchPlaceholder}
              aria-label="Search runs"
              role="searchbox"
              autoComplete="off"
              autoCorrect="off"
              autoCapitalize="none"
              spellCheck={false}
              enterKeyHint="search"
              data-form-type="other"
              data-lpignore="true"
              data-1p-ignore="true"
              className="pr-9"
              disabled={disabled}
            />
            {isLoading ? (
              <Loader2 className="pointer-events-none absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 animate-spin text-slate-400" />
            ) : null}
          </div>

          <div
            role="listbox"
            aria-busy={isLoading}
            aria-label="Available runs"
            className="mt-2 max-h-64 overflow-y-auto overscroll-contain pr-1"
          >
            {visibleRuns.length > 0 ? (
              <div className="grid gap-1">
                {visibleRuns.map((run) => {
                  const isSelected = run.run_id === selectedRunId;
                  return (
                    <Button
                      key={run.run_id}
                      type="button"
                      variant="ghost"
                      role="option"
                      aria-selected={isSelected}
                      className={cn(
                        "h-auto w-full justify-start px-3 py-2 text-left whitespace-normal",
                        isSelected ? "bg-slate-100" : null,
                      )}
                      onClick={() => {
                        onSelectRun(run.run_id);
                        setIsOpen(false);
                      }}
                    >
                      <span className="flex-1 truncate" title={formatRunLabel(run)}>
                        {formatRunLabel(run)}
                      </span>
                      {isSelected ? <Check className="h-4 w-4 shrink-0 text-slate-600" /> : null}
                    </Button>
                  );
                })}
              </div>
            ) : isLoading ? (
              <div className="flex items-center gap-2 px-3 py-2 text-sm text-slate-500">
                <Loader2 className="h-4 w-4 animate-spin" />
                Loading runs...
              </div>
            ) : (
              <p className="px-3 py-2 text-sm text-slate-500">{emptyMessage}</p>
            )}
          </div>
        </div>
      ) : null}
    </div>
  );
}
