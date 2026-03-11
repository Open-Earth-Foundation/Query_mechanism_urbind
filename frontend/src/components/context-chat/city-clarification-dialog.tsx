import { Loader2 } from "lucide-react";

import { SearchableCityPicker } from "@/components/searchable-city-picker";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { formatCityLabel } from "@/lib/utils";

interface CityClarificationDialogProps {
  open: boolean;
  isSending: boolean;
  question: string | null;
  cities: string[];
  selectedCity: string | null;
  onSelectCity: (city: string | null) => void;
  onClose: () => void;
  onConfirm: () => void;
  errorMessage: string | null;
  isLoading: boolean;
}

export function CityClarificationDialog({
  open,
  isSending,
  question,
  cities,
  selectedCity,
  onSelectCity,
  onClose,
  onConfirm,
  errorMessage,
  isLoading,
}: CityClarificationDialogProps) {
  return (
    <Dialog
      open={open}
      onOpenChange={(nextOpen) => {
        if (isSending) {
          return;
        }
        if (!nextOpen) {
          onClose();
          return;
        }
      }}
    >
      <DialogContent className="flex max-h-[85vh] max-w-xl flex-col overflow-hidden p-0">
        <DialogHeader className="border-b border-slate-200 px-6 pb-4 pt-6 pr-14">
          <DialogTitle>Choose One City</DialogTitle>
          <DialogDescription>Pick one city for additional search.</DialogDescription>
        </DialogHeader>
        <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-hidden px-6 py-5">
          {question ? (
            <div className="rounded-md border border-slate-200 bg-slate-50 p-3 text-sm text-slate-700">
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                Original question
              </p>
              <p className="mt-1 whitespace-pre-wrap">{question}</p>
            </div>
          ) : null}
          <SearchableCityPicker
            cities={cities}
            selectedCities={selectedCity ? [selectedCity] : []}
            onSelectCity={onSelectCity}
            className="min-h-0 flex-1"
            disabled={isSending}
            errorMessage={errorMessage}
            isLoading={isLoading}
            emptyMessage="No cities match the current filter."
            loadingMessage="Loading cities..."
          />
        </div>
        <DialogFooter className="border-t border-slate-200 bg-slate-50 px-6 py-4 sm:justify-between sm:space-x-0">
          <p className="text-xs text-slate-600">
            {selectedCity
              ? `Selected city: ${formatCityLabel(selectedCity)}`
              : "Select one city to continue with a focused follow-up search."}
          </p>
          <div className="flex items-center justify-end gap-2">
            <Button type="button" variant="outline" disabled={isSending} onClick={onClose}>
              Cancel
            </Button>
            <Button type="button" disabled={!selectedCity || isSending} onClick={onConfirm}>
              {isSending ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
              Continue with {selectedCity ? formatCityLabel(selectedCity) : "City"}
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
