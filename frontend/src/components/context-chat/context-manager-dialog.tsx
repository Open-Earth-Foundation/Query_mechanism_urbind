import { Loader2 } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ChatContextSummary } from "@/lib/api";

interface ContextManagerDialogProps {
  open: boolean;
  runId: string;
  managerTokenCap: number;
  selectedContextTokens: number;
  selectionExceedsDirectCap: boolean;
  isLoadingCatalog: boolean;
  contextCatalog: ChatContextSummary[];
  managerSelection: string[];
  onToggleSelection: (runId: string) => void;
  onOpenChange: (open: boolean) => void;
  onSave: () => void;
  isSavingContexts: boolean;
  pendingJob: { job_id: string } | null;
  catalogError: string | null;
}

export function ContextManagerDialog({
  open,
  runId,
  managerTokenCap,
  selectedContextTokens,
  selectionExceedsDirectCap,
  isLoadingCatalog,
  contextCatalog,
  managerSelection,
  onToggleSelection,
  onOpenChange,
  onSave,
  isSavingContexts,
  pendingJob,
  catalogError,
}: ContextManagerDialogProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[85vh] max-w-3xl overflow-hidden p-0">
        <DialogHeader className="border-b border-slate-200 p-5">
          <DialogTitle>Context Manager</DialogTitle>
          <DialogDescription>
            Select or combine multiple run contexts. Direct prompt cap before overflow:{" "}
            {managerTokenCap.toLocaleString()} tokens.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-3 p-5">
          <div className="rounded-md border border-slate-200 bg-slate-50 p-3 text-xs text-slate-700">
            Selected context tokens: {selectedContextTokens.toLocaleString()} /{" "}
            {managerTokenCap.toLocaleString()}
            {selectionExceedsDirectCap ? (
              <p className="mt-2 text-amber-700">
                This context is very large. AI may take longer to answer.
              </p>
            ) : null}
          </div>

          <ScrollArea className="h-[45vh] rounded-md border border-slate-200 p-3">
            {isLoadingCatalog ? (
              <div className="flex items-center gap-2 text-sm text-slate-600">
                <Loader2 className="h-4 w-4 animate-spin" />
                Loading available contexts...
              </div>
            ) : contextCatalog.length === 0 ? (
              <p className="text-sm text-slate-600">No completed run contexts available.</p>
            ) : (
              <div className="space-y-2">
                {contextCatalog.map((context) => {
                  const selected = managerSelection.includes(context.run_id);
                  const isPinnedBaseRun = context.run_id === runId;
                  return (
                    <button
                      key={context.run_id}
                      type="button"
                      onClick={() => onToggleSelection(context.run_id)}
                      disabled={isPinnedBaseRun}
                      className={
                        selected
                          ? "w-full rounded-md border border-teal-300 bg-teal-50 p-3 text-left"
                          : "w-full rounded-md border border-slate-200 bg-white p-3 text-left hover:border-slate-300 disabled:cursor-not-allowed disabled:opacity-45"
                      }
                    >
                      <div className="mb-1 flex items-center justify-between gap-2">
                        <p className="text-sm font-semibold text-slate-900">{context.run_id}</p>
                        <Badge variant="outline">
                          {context.prompt_context_tokens.toLocaleString()} tokens
                        </Badge>
                      </div>
                      <p className="line-clamp-2 text-xs text-slate-600">{context.question}</p>
                      {isPinnedBaseRun ? (
                        <p className="mt-1 text-xs text-teal-700">Pinned parent run</p>
                      ) : null}
                    </button>
                  );
                })}
              </div>
            )}
          </ScrollArea>

          <div className="flex items-center justify-between gap-3">
            {catalogError ? (
              <p className="text-xs text-red-600">{catalogError}</p>
            ) : (
              <p className="text-xs text-slate-500">
                The parent run stays pinned; select additional runs as needed. Selections above the
                direct prompt cap are allowed and will use overflow mode.
              </p>
            )}
            <Button
              type="button"
              size="sm"
              onClick={onSave}
              disabled={!!pendingJob || isSavingContexts || isLoadingCatalog || managerSelection.length === 0}
            >
              {isSavingContexts ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
              Save Selection
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
