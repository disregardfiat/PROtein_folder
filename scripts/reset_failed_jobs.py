#!/usr/bin/env python3
"""
Reset all failed CASP jobs so they can be retried.

Moves outputs/{base}.failed.txt back to pending/{base}.txt and sets attempts=0.
Run from repo root (or set CASP_OUTPUT_DIR). Usage:

  python scripts/reset_failed_jobs.py
  CASP_OUTPUT_DIR=/path/to/casp_results python scripts/reset_failed_jobs.py
"""

from __future__ import annotations

import os
import sys

# Repo root on path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_script_dir)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Use same dirs as casp_server
_output_base = os.environ.get("CASP_OUTPUT_DIR") or os.path.join(_repo_root, "casp_results")
OUTPUTS_DIR = os.path.join(_output_base, "outputs")
PENDING_DIR = os.path.join(_output_base, "pending")


def _update_attempts_in_file(txt_path: str, attempts: int) -> None:
    with open(txt_path) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("attempts="):
            lines[i] = f"attempts={attempts}\n"
            break
    else:
        lines.insert(0, f"attempts={attempts}\n")
    with open(txt_path, "w") as f:
        f.writelines(lines)


def reset_failed_jobs() -> int:
    """Move each outputs/*.failed.txt to pending/{base}.txt and set attempts=0. Returns count reset."""
    if not os.path.isdir(OUTPUTS_DIR):
        return 0
    count = 0
    for name in os.listdir(OUTPUTS_DIR):
        if not name.endswith(".failed.txt"):
            continue
        base = name[: -len(".failed.txt")]
        src = os.path.join(OUTPUTS_DIR, name)
        dst = os.path.join(PENDING_DIR, f"{base}.txt")
        try:
            os.makedirs(PENDING_DIR, exist_ok=True)
            with open(src) as f:
                content = f.read()
            with open(dst, "w") as f:
                f.write(content)
            os.remove(src)
            _update_attempts_in_file(dst, 0)
            count += 1
            print(f"Reset {base} -> attempts=0")
        except Exception as e:
            print(f"Error resetting {base}: {e}", file=sys.stderr)
    return count


if __name__ == "__main__":
    n = reset_failed_jobs()
    print(f"Reset {n} failed job(s).")
