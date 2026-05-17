"use client";

import { useAuth } from "@clerk/nextjs";
import { useEffect } from "react";

import { registerSessionTokenProvider } from "@/lib/api";

export function ClerkSessionBridge() {
  const { getToken, isLoaded, isSignedIn } = useAuth();

  useEffect(() => {
    if (!isLoaded || !isSignedIn) {
      registerSessionTokenProvider(null);
      return;
    }

    registerSessionTokenProvider(async () => {
      return await getToken();
    });

    return () => {
      registerSessionTokenProvider(null);
    };
  }, [getToken, isLoaded, isSignedIn]);

  return null;
}
