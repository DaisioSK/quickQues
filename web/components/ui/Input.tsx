/**
 * Input + TextArea — minimal Tailwind primitives.
 *
 * Two components in one file because they share base classes; splitting
 * would invent ceremony without adding clarity. If TextArea sprouts
 * resize handles, autosize, etc., factor it out then.
 */

import type { InputHTMLAttributes, TextareaHTMLAttributes } from "react";

const BASE_CLASSES =
  "block w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-sm text-gray-900 placeholder:text-gray-400 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 disabled:bg-gray-50 disabled:text-gray-500";

export function Input({
  className = "",
  ...rest
}: InputHTMLAttributes<HTMLInputElement>) {
  return <input className={`${BASE_CLASSES} ${className}`} {...rest} />;
}

export function TextArea({
  className = "",
  rows = 3,
  ...rest
}: TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return (
    <textarea
      className={`${BASE_CLASSES} resize-y ${className}`}
      rows={rows}
      {...rest}
    />
  );
}
