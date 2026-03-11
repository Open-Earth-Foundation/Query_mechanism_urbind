"use client";

import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { FrontendMode } from "@/lib/frontend-mode";

interface DevModeToggleProps {
  mode: FrontendMode;
  onModeChange: (mode: FrontendMode) => void;
}

export function DevModeToggle({ mode, onModeChange }: DevModeToggleProps) {
  const isDevMode = mode === "dev";

  return (
    <div className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white/90 px-4 py-2 shadow-sm">
      <div className="flex items-center gap-2">
        <Badge variant={isDevMode ? "outline" : "secondary"}>Standard</Badge>
        <Switch
          checked={isDevMode}
          onCheckedChange={(checked) => onModeChange(checked ? "dev" : "standard")}
          aria-label="Toggle frontend dev mode"
        />
        <Badge variant={isDevMode ? "secondary" : "outline"}>Dev</Badge>
      </div>
    </div>
  );
}
