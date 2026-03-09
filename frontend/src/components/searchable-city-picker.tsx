"use client";

import { useMemo, useState } from "react";
import { Loader2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn, formatCityLabel } from "@/lib/utils";

interface SearchableCityPickerProps {
  cities: string[];
  selectedCities: string[];
  onSelectCity: (city: string) => void;
  className?: string;
  disabled?: boolean;
  errorMessage?: string | null;
  isLoading?: boolean;
  emptyMessage?: string;
  loadingMessage?: string;
  scrollAreaClassName?: string;
  searchPlaceholder?: string;
}

export function SearchableCityPicker({
  cities,
  selectedCities,
  onSelectCity,
  className,
  disabled = false,
  errorMessage = null,
  isLoading = false,
  emptyMessage = "No cities found.",
  loadingMessage = "Loading cities...",
  scrollAreaClassName,
  searchPlaceholder = "Filter cities...",
}: SearchableCityPickerProps) {
  const [filter, setFilter] = useState("");

  const filteredCities = useMemo(() => {
    const needle = filter.trim().toLowerCase();
    if (!needle) {
      return cities;
    }
    return cities.filter((city) => city.toLowerCase().includes(needle));
  }, [cities, filter]);

  return (
    <div className={cn("flex min-h-0 flex-col gap-3", className)}>
      <Input
        placeholder={searchPlaceholder}
        value={filter}
        onChange={(event) => setFilter(event.target.value)}
        disabled={disabled || isLoading}
      />
      <div className="min-h-0 flex-1 overflow-hidden rounded-md border border-slate-200 bg-white">
        <div
          className={cn(
            "min-h-0 h-64 overflow-y-auto overscroll-contain p-3 [scrollbar-gutter:stable]",
            scrollAreaClassName,
          )}
        >
          <div className="grid gap-2">
            {isLoading ? (
              <div className="flex items-center gap-2 text-sm text-slate-600">
                <Loader2 className="h-4 w-4 animate-spin" />
                {loadingMessage}
              </div>
            ) : errorMessage ? (
              <p className="text-sm text-red-600">{errorMessage}</p>
            ) : filteredCities.length > 0 ? (
              filteredCities.map((city) => {
                const isSelected = selectedCities.includes(city);
                return (
                  <Button
                    key={city}
                    type="button"
                    variant={isSelected ? "default" : "outline"}
                    className="justify-start"
                    disabled={disabled}
                    onClick={() => onSelectCity(city)}
                  >
                    {formatCityLabel(city)}
                  </Button>
                );
              })
            ) : (
              <p className="text-sm text-slate-500">{emptyMessage}</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
