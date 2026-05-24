"use client";

import { FormEvent, useState } from "react";
import { useRouter } from "next/navigation";
import { Loader2, LockKeyhole } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

interface LoginFormProps {
  nextPath: string;
}

export function LoginForm({ nextPath }: LoginFormProps) {
  const router = useRouter();
  const [password, setPassword] = useState("");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    if (isSubmitting) {
      return;
    }

    setIsSubmitting(true);
    setErrorMessage(null);

    try {
      const response = await fetch("/api/auth/login", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          next: nextPath,
          password,
        }),
      });

      const payload = (await response.json()) as {
        detail?: string;
        redirectTo?: string;
      };

      if (!response.ok) {
        throw new Error(payload.detail || `Login failed (${response.status}).`);
      }

      router.replace(payload.redirectTo || "/");
      router.refresh();
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Login failed.");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <Card className="w-full max-w-md border-slate-300">
      <CardHeader className="space-y-3">
        <div className="inline-flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.18em] text-amber-700">
          <LockKeyhole className="h-3.5 w-3.5" />
          Shared Access
        </div>
        <div>
          <CardTitle>Enter the shared password</CardTitle>
          <CardDescription>
            This gate protects the shared workspace and its run history.
          </CardDescription>
        </div>
      </CardHeader>
      <CardContent>
        <form className="space-y-4" onSubmit={(event) => void handleSubmit(event)}>
          <input
            type="text"
            name="username"
            autoComplete="username"
            value="shared-access"
            readOnly
            tabIndex={-1}
            aria-hidden="true"
            className="sr-only"
          />
          <div className="space-y-2">
            <Label htmlFor="password">Password</Label>
            <Input
              id="password"
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              autoComplete="current-password"
              autoFocus
            />
          </div>
          {errorMessage ? <p className="text-sm text-red-600">{errorMessage}</p> : null}
          <Button type="submit" className="w-full" disabled={isSubmitting || password.length === 0}>
            {isSubmitting ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
            Continue
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}
