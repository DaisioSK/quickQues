/**
 * Smoke tests for the Button atom.
 *
 * Why we test the atoms at all: they're the shared surface ssChat and
 * ssPdfViewer both depend on. A regression here would cascade into
 * both downstream sub-sprints; cheap to lock the contract now.
 */

import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { Button } from "./Button";

describe("Button", () => {
  it("renders children", () => {
    render(<Button>Submit</Button>);
    expect(screen.getByRole("button", { name: "Submit" })).toBeInTheDocument();
  });

  it("calls onClick when clicked", () => {
    const onClick = vi.fn();
    render(<Button onClick={onClick}>Submit</Button>);
    fireEvent.click(screen.getByRole("button"));
    expect(onClick).toHaveBeenCalledOnce();
  });

  it("respects the disabled prop", () => {
    const onClick = vi.fn();
    render(
      <Button disabled onClick={onClick}>
        Submit
      </Button>,
    );
    const btn = screen.getByRole("button");
    expect(btn).toBeDisabled();
    // Disabled buttons don't fire onClick — DOM semantics, not us.
    fireEvent.click(btn);
    expect(onClick).not.toHaveBeenCalled();
  });

  it("applies variant classes", () => {
    render(<Button variant="secondary">x</Button>);
    const btn = screen.getByRole("button");
    // We only assert the variant-specific class to avoid coupling the
    // test to the full base-class string (which churns with Tailwind tweaks).
    expect(btn.className).toMatch(/bg-gray-200/);
  });
});
