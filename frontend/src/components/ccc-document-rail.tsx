"use client";

import { FileText, Loader2 } from "lucide-react";

import { MarkdownWithReferences } from "@/components/markdown-with-references";
import { SearchableCityPicker } from "@/components/searchable-city-picker";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { formatCityLabel } from "@/lib/utils";

interface CccDocumentRailProps {
  cities: string[];
  selectedCity: string | null;
  onSelectCity: (city: string) => void;
  content: string | null;
  sourcePaths: string[];
  isLoadingCities?: boolean;
  citiesError?: string | null;
  isLoadingContent?: boolean;
  contentError?: string | null;
}

export function CccDocumentRail({
  cities,
  selectedCity,
  onSelectCity,
  content,
  sourcePaths,
  isLoadingCities = false,
  citiesError = null,
  isLoadingContent = false,
  contentError = null,
}: CccDocumentRailProps) {
  const selectedCities = selectedCity ? [selectedCity] : [];
  const selectedCityLabel = selectedCity ? formatCityLabel(selectedCity) : "Select a city";

  return (
    <div className="space-y-4">
      <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
        <div className="flex items-start justify-between gap-3">
          <div className="inline-flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">
            <FileText className="h-3.5 w-3.5" />
            CCC Source
          </div>
          <Badge variant="outline">{selectedCityLabel}</Badge>
        </div>

        <div className="mt-4">
          <SearchableCityPicker
            cities={cities}
            selectedCities={selectedCities}
            onSelectCity={onSelectCity}
            isLoading={isLoadingCities}
            errorMessage={citiesError}
            loadingMessage="Loading CCC cities..."
            emptyMessage="No CCC cities found."
            searchPlaceholder="Search CCC city..."
            scrollAreaClassName="h-40"
          />
        </div>
      </div>

      <div className="rounded-xl border border-slate-200 bg-white">
        <div className="border-b border-slate-200 px-4 py-3">
          <div className="flex items-center justify-between gap-2">
            <p className="text-sm font-medium text-slate-900">Rendered CCC</p>
            {sourcePaths.length > 0 ? (
              <Badge variant="secondary">
                {sourcePaths.length} source{sourcePaths.length === 1 ? "" : "s"}
              </Badge>
            ) : null}
          </div>
          {sourcePaths.length > 0 ? (
            <p className="mt-2 text-xs text-slate-500">{sourcePaths.join(", ")}</p>
          ) : null}
        </div>

        <ScrollArea className="h-[42vh]">
          {isLoadingContent ? (
            <div className="flex items-center gap-2 p-4 text-sm text-slate-600">
              <Loader2 className="h-4 w-4 animate-spin" />
              Loading CCC markdown...
            </div>
          ) : contentError ? (
            <div className="p-4 text-sm text-red-600">{contentError}</div>
          ) : content ? (
            <article className="document-markdown document-markdown-rail p-4">
              <MarkdownWithReferences
                content={content}
                runId={null}
                prefetchRunReferences={false}
                hideImages
              />
            </article>
          ) : (
            <div className="p-4 text-sm text-slate-500">
              Select a city to open its CCC source.
            </div>
          )}
        </ScrollArea>
      </div>
    </div>
  );
}
