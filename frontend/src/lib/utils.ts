import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

export function formatCityLabel(value: string): string {
  const trimmed = value.trim();
  if (!trimmed) {
    return "";
  }
  return `${trimmed.charAt(0).toLocaleUpperCase()}${trimmed.slice(1)}`;
}

