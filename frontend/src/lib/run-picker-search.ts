import type { RunSummary } from "@/lib/api";

function normalizeRunSearchValue(value: string | null | undefined): string {
  return value?.trim().toLowerCase() ?? "";
}

function compactRunSearchValue(value: string | null | undefined): string {
  return normalizeRunSearchValue(value).replace(/[^a-z0-9]+/g, "");
}

function hasNumericSearchFragment(value: string): boolean {
  return /\d/.test(value);
}

export function matchesImmediateRunSearch(run: RunSummary, searchQuery: string): boolean {
  const normalizedQuery = normalizeRunSearchValue(searchQuery);
  if (!normalizedQuery || !hasNumericSearchFragment(normalizedQuery)) {
    return true;
  }

  const compactQuery = compactRunSearchValue(normalizedQuery);
  if (!compactQuery) {
    return true;
  }

  return [run.run_id, run.picker_timestamp, run.question].some((value) =>
    compactRunSearchValue(value).includes(compactQuery),
  );
}

export function filterImmediateRunMatches(
  runs: RunSummary[],
  searchQuery: string,
): RunSummary[] {
  return runs.filter((run) => matchesImmediateRunSearch(run, searchQuery));
}
