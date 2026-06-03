/**
 * Button — minimal Tailwind primitive.
 *
 * Why hand-rolled rather than shadcn/ui per DECISION-mvp.next.1:
 *   shadcn requires radix + cva + tailwind-merge. For our 3-component
 *   MVP atom set that's 5 npm deps and a CSS-class-juggling layer of
 *   abstraction we don't need yet. When the surface grows past
 *   primary/secondary, swap in shadcn — it's a one-day refactor.
 *
 * Variant set kept intentionally tiny:
 *   primary  → main CTA (Send message, etc.)
 *   secondary → less-prominent action (Cancel, Reset)
 *   ghost    → near-invisible icon-style trigger
 */

import type { ButtonHTMLAttributes, ReactNode } from "react";

type Variant = "primary" | "secondary" | "ghost";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  children: ReactNode;
}

const VARIANT_CLASSES: Record<Variant, string> = {
  primary:
    "bg-blue-600 text-white hover:bg-blue-700 active:bg-blue-800 disabled:bg-blue-300",
  secondary:
    "bg-gray-200 text-gray-900 hover:bg-gray-300 active:bg-gray-400 disabled:bg-gray-100 disabled:text-gray-400",
  ghost:
    "bg-transparent text-gray-700 hover:bg-gray-100 active:bg-gray-200 disabled:text-gray-300",
};

const BASE_CLASSES =
  "inline-flex items-center justify-center rounded-md px-4 py-2 text-sm font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1 disabled:cursor-not-allowed";

export function Button({
  variant = "primary",
  className = "",
  children,
  ...rest
}: ButtonProps) {
  return (
    <button
      className={`${BASE_CLASSES} ${VARIANT_CLASSES[variant]} ${className}`}
      {...rest}
    >
      {children}
    </button>
  );
}
