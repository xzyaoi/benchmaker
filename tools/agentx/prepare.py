"""Download the AgentX (SemiAnalysis / InferenceX) WEKA trace corpus.

Source:
    https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126       (full)
    https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k  (<=256k contexts)

The corpus holds 393 opt-in, sanitized Claude Code sessions in WEKA trace
format: one JSON object per session, each request recording its
session-scoped chained ``hash_ids`` (64-token blocks), recorded timestamps,
input/output token counts and subagent topology — but no prompt text. The
``benchmaker agentx`` recipe (and
:class:`benchmaker.workloads.agentx.AgentXWorkload`) synthesizes deterministic
prompts from those block hashes, so the replay preserves the recording's exact
prefix-cache structure.

The recipe can stream the corpus straight from HuggingFace by itself; this
script exists to pin a local copy (reproducibility, air-gapped clusters) and
to carve out a small subset for fast iteration:

    python tools/agentx/prepare.py
    python tools/agentx/prepare.py --variant 256k
    python tools/agentx/prepare.py --max-traces 20 --min-input-tokens 0

Usage:
    python tools/agentx/prepare.py                       # .local/agentx-traces-weka-062126.jsonl
    python tools/agentx/prepare.py --variant 256k        # the 256k pre-dropped variant
    python tools/agentx/prepare.py --out my.jsonl --max-traces 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

VARIANTS = {
    "full": "semianalysisai/cc-traces-weka-062126",
    "256k": "semianalysisai/cc-traces-weka-062126-256k",
}
DEFAULT_OUT = ".local/agentx-traces-weka-062126.jsonl"


def download(repo_id: str, filename: str, out: str) -> str:
    from huggingface_hub import hf_hub_download

    print(f"[download] {repo_id}/{filename} -> {out}")
    cached = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    if os.path.abspath(cached) != os.path.abspath(out):
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        # Copy (not symlink) so the local file is stable across HF cache gc.
        with open(cached, "rb") as src, open(out, "wb") as dst:
            while True:
                chunk = src.read(1 << 22)
                if not chunk:
                    break
                dst.write(chunk)
    print(f"[done] {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
    return out


def subset(src: str, dst: str, *, max_traces: Optional[int] = None,
           min_input_tokens: Optional[int] = None,
           max_input_tokens: Optional[int] = None,
           min_turns: Optional[int] = None,
           want_subagents: Optional[bool] = None) -> dict:
    """Filter the corpus into a smaller JSONL; returns summary stats."""
    kept = 0
    total_requests = 0
    n_in = 0
    with open(src, "r") as f, open(dst, "w") as out:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            reqs = [r for r in row.get("requests") or [] if isinstance(r, dict)]
            has_sub = any(r.get("type") == "subagent" for r in reqs)
            if want_subagents is True and not has_sub:
                continue
            if want_subagents is False and has_sub:
                continue
            if min_turns is not None and len(reqs) < min_turns:
                continue
            ins = [r.get("in") for r in reqs if isinstance(r.get("in"), (int, float))]
            if min_input_tokens is not None and ins and max(ins) < min_input_tokens:
                continue
            if max_input_tokens is not None and ins and min(ins) > max_input_tokens:
                continue
            out.write(json.dumps(row, separators=(",", ":")) + "\n")
            kept += 1
            total_requests += len(reqs)
            n_in += sum(int(i) for i in ins)
            if max_traces is not None and kept >= max_traces:
                break
    stats = {"traces": kept, "requests": total_requests, "input_tokens": n_in}
    print(f"[subset] {dst}: {stats}")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="full",
                    help="Corpus variant to download (default: full).")
    ap.add_argument("--dataset", default=None,
                    help="Explicit HF dataset id (overrides --variant).")
    ap.add_argument("--file", default="traces.jsonl",
                    help="Trace file name inside the dataset repo.")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="Output path (default: %(default)s).")
    ap.add_argument("--skip-subset", action="store_true",
                    help="Only download; do not apply filters.")
    ap.add_argument("--max-traces", type=int, default=None,
                    help="Keep at most this many traces (first-come order).")
    ap.add_argument("--min-input-tokens", type=int, default=None,
                    help="Keep only traces whose peak recorded input reaches this.")
    ap.add_argument("--max-input-tokens", type=int, default=None,
                    help="Keep only traces whose smallest recorded input is below this.")
    ap.add_argument("--min-turns", type=int, default=None,
                    help="Keep only traces with at least this many requests.")
    ap.add_argument("--subagents", choices=("yes", "no", "any"), default="any",
                    help="Keep only traces with (no) subagent entries.")
    args = ap.parse_args()

    repo_id = args.dataset or VARIANTS[args.variant]
    out = args.out or DEFAULT_OUT
    path = download(repo_id, args.file, out)
    if not args.skip_subset and (args.max_traces or args.min_input_tokens
                                 or args.max_input_tokens or args.min_turns
                                 or args.subagents != "any"):
        base, ext = os.path.splitext(out)
        sub_out = f"{base}.subset{ext}"
        subset(
            path, sub_out,
            max_traces=args.max_traces,
            min_input_tokens=args.min_input_tokens,
            max_input_tokens=args.max_input_tokens,
            min_turns=args.min_turns,
            want_subagents=None if args.subagents == "any"
            else (args.subagents == "yes"),
        )
        print(f"\nRun: benchmaker agentx --trace {sub_out} \\\n"
              "    --url http://localhost:8000/v1/chat/completions --model ...")
    else:
        print(f"\nRun: benchmaker agentx --trace {path} \\\n"
              "    --url http://localhost:8000/v1/chat/completions --model ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
