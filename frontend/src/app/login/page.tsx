import { LoginForm } from "@/components/login-form";

interface LoginPageProps {
  searchParams: Promise<{
    next?: string | string[];
  }>;
}

function normalizeNextPath(value: string | string[] | undefined): string {
  const resolved = Array.isArray(value) ? value[0] : value;
  const cleaned = resolved?.trim();
  if (!cleaned || !cleaned.startsWith("/") || cleaned.startsWith("//")) {
    return "/";
  }
  return cleaned;
}

export default async function LoginPage({ searchParams }: LoginPageProps) {
  const params = await searchParams;

  return (
    <main className="flex min-h-screen items-center justify-center bg-slate-100 px-4 py-12">
      <LoginForm nextPath={normalizeNextPath(params.next)} />
    </main>
  );
}
