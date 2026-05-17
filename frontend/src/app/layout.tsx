import type { Metadata } from "next";
import { ClerkProvider } from "@clerk/nextjs";
import { Space_Grotesk, Source_Serif_4 } from "next/font/google";

import { ClerkSessionBridge } from "@/components/clerk-session-bridge";
import "./globals.css";

const spaceGrotesk = Space_Grotesk({
  variable: "--font-space-grotesk",
  subsets: ["latin"],
});

const sourceSerif = Source_Serif_4({
  variable: "--font-source-serif",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Query Mechanism Document Builder",
  description: "Document-first run builder with optional context chat.",
};

function readOptionalEnv(value: string | undefined, fallback: string): string {
  const cleaned = value?.trim();
  return cleaned && cleaned.length > 0 ? cleaned : fallback;
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const signInUrl = readOptionalEnv(
    process.env.NEXT_PUBLIC_CLERK_SIGN_IN_URL,
    "/sign-in",
  );
  const signUpUrl = readOptionalEnv(
    process.env.NEXT_PUBLIC_CLERK_SIGN_UP_URL,
    "/sign-up",
  );
  const signInFallbackRedirectUrl = readOptionalEnv(
    process.env.NEXT_PUBLIC_CLERK_SIGN_IN_FALLBACK_REDIRECT_URL,
    "/",
  );
  const signUpFallbackRedirectUrl = readOptionalEnv(
    process.env.NEXT_PUBLIC_CLERK_SIGN_UP_FALLBACK_REDIRECT_URL,
    "/",
  );

  return (
    <html lang="en">
      <body className={`${spaceGrotesk.variable} ${sourceSerif.variable} antialiased`}>
        <ClerkProvider
          signInUrl={signInUrl}
          signUpUrl={signUpUrl}
          signInFallbackRedirectUrl={signInFallbackRedirectUrl}
          signUpFallbackRedirectUrl={signUpFallbackRedirectUrl}
        >
          <ClerkSessionBridge />
          {children}
        </ClerkProvider>
      </body>
    </html>
  );
}
