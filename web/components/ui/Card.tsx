/**
 * Card — content container primitive.
 *
 * Used to wrap chat messages, answer blocks, citation lists. Just a
 * Tailwind div with consistent padding + border. No header/footer
 * sub-components yet — caller composes those inline. Add slot
 * sub-components when N=2 callers need the same composition.
 */

import type { HTMLAttributes, ReactNode } from "react";

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode;
}

const BASE_CLASSES =
  "rounded-lg border border-gray-200 bg-white p-4 shadow-sm";

export function Card({ className = "", children, ...rest }: CardProps) {
  return (
    <div className={`${BASE_CLASSES} ${className}`} {...rest}>
      {children}
    </div>
  );
}
