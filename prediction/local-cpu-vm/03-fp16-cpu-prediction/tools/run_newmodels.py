#!/usr/bin/env python3
"""tools/run_l256.py for the generalisation-test models (2026-11-22). Unchanged scheduler; only the run-time ordering
key (run395.cost) learns the new model names, mapped to the closest model of the main table."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run395  # noqa: E402

ALIAS = {"convnext-tiny-224": "resnet-50", "efficientnet-b0": "mobilenet-v2-1.0-224", "regnet-y-040": "resnet-50",
         "qwen2.5-0.5b-instruct": "tinyllama-1.1b-chat", "smollm2-360m-instruct": "gpt2-medium",
         "pythia-410m-deduped": "gpt2-medium"}
_orig = run395.cost


def cost(c):
    return _orig(dict(c, short=ALIAS.get(c["short"], c["short"])))


run395.cost = cost
import run_l256  # noqa: E402

run_l256.cost = cost
if __name__ == "__main__":
    sys.exit(run_l256.main())
