#!/usr/bin/env python3
"""One-command entry point and private crash/resume worker launcher."""

import os
import sys
from pathlib import Path

from tdes import run_crash_phase, run_demo, run_resume_phase


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker-crash":
        run_crash_phase(Path(sys.argv[2]))
        # This is intentionally abrupt: no exception handler or Python cleanup.
        os._exit(86)
    elif len(sys.argv) == 3 and sys.argv[1] == "--worker-resume":
        run_resume_phase(Path(sys.argv[2]))
    elif len(sys.argv) == 1:
        result = run_demo()
        print(f"Demo complete: {result['artifacts']}/evidence.md")
    else:
        raise SystemExit("usage: python S6/run_demo.py")
