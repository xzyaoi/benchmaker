# AgentX — the SemiAnalysis agentic-coding inference benchmark

[AgentX](https://inferencex.semianalysis.com/agentx) is SemiAnalysis's
agentic-coding benchmark from the InferenceX project. It replays **real
Claude Code sessions** — main-agent turns interleaved with concurrent
subagent streams — against an OpenAI-compatible endpoint, and scores the
server on TTFT, inter-token latency and decode throughput under that load.

`benchmaker agentx` implements the published v1.0 methodology end to end:

* **Block-hash synthetic prompts.** The public corpus records no prompt text —
  each request lists the session-scoped 64-token KV blocks it reuses
  (`hash_ids`, chained as the session grows). Prompts are rendered
  deterministically from `(trace_id, hash_id)`, so requests that shared a
  prefix in the recording share a **byte-exact prefix** in the replay, and
  requests that didn't, don't. This reproduces the recording's prefix-cache
  structure without any private data.
* **Session-tree DAG.** Each session is a tree: main-agent turns plus subagent
  groups that *spawn* after a recorded main turn and *join* before the next
  dependent one. A request dispatches the moment its recorded dependencies
  are done and its recorded end-to-start gap has elapsed — subagents run
  concurrently with the main stream, and a join turn waits for every child
  to drain.
* **Client concurrency = live session trees.** `--concurrency N` keeps `N`
  trees in flight (each tree may keep several subagent requests in flight
  simultaneously). A finished tree recycles to the next trace from turn 0.
* **Seeded start points.** Each lane starts 25–75% through its recorded
  session (`--start-min-ratio` / `--start-max-ratio`), sampling the
  mid-session working set rather than only cold starts. Deterministic given
  `--seed`.
* **Warmup pass, then one profiling window.** Each lane first sends a
  `max_tokens=1` **primer** (materializing the live prefix in the server's KV
  cache), then `--warmup-requests` (default 10) further requests — all
  **excluded from reported metrics**. Only the following window (default one
  hour) is measured.
* **Cache-bust on recycle.** Every play prepends a unique `[rid:…]` marker
  (compensated out of the recorded token budget), so the second play of a
  trace never inherits the first play's KV prefix.
* **Decode-length fidelity.** `max_tokens` is the recorded `output_tokens`
  with `ignore_eos` on, so the server decodes the recorded lengths instead of
  stopping early.

## Requirements

**A long-context server.** The corpus is real coding sessions and they are
big — measured over all 393 traces:

| Server context window | Traces that fit |
| --- | --- |
| 32k  | 0 of 393 |
| 64k  | 7 (2%) |
| 128k | 82 (21%) |
| 256k | 220 (56%) |
| 400k | 294 (75%) |

Median peak context is **226k tokens** (max ~1M). Target a ≥128k server;
≥256k replays the full corpus. Use `--max-context` (or the `-256k` corpus
variant, below) to carve the corpus to your window — traces whose peak
recorded input + output exceeds it are dropped at load time.

Everything else is standard benchmaker: `uv sync` (add `--extra hf` for the
exact-tokenizer mode), an OpenAI-compatible `/v1/chat/completions` endpoint.

## 1. Get the corpus

Two HuggingFace datasets, recorded in the [WEKA trace
format](https://github.com/ai-dynamo/aiperf) (one JSON object per session, no
prompt text):

| Dataset | Contents |
| --- | --- |
| `semianalysisai/cc-traces-weka-062126` | 393 opt-in sanitized Claude Code sessions — 98,827 requests, 1,697 subagent groups, full context lengths |
| `semianalysisai/cc-traces-weka-062126-256k` | same sessions pre-dropped to ≤256k peak context |

The recipe streams the public dataset from HuggingFace automatically on first
use. To pin a local copy (or subset it):

```bash
# full corpus -> .local/agentx-traces-weka-062126.jsonl (~1.8 GB)
python tools/agentx/prepare.py

# the pre-carved 256k variant
python tools/agentx/prepare.py --variant 256k

# small slice for smoke tests
python tools/agentx/prepare.py --max-traces 12
#   -> .local/agentx-traces-weka-062126.subset.jsonl
```

If HuggingFace's cache directory is not writable in your environment, point
it elsewhere first: `export HF_HOME=/tmp/hf-home`.

## 2. Smoke test

```bash
export OPENAI_API_BASE_URL=http://localhost:8000/v1/chat/completions
export OPENAI_COMPATIBLE_MODEL=your-model

python tools/agentx/prepare.py --max-traces 5
benchmaker agentx --trace .local/agentx-traces-weka-062126.subset.jsonl \
    --concurrency 2 --duration 60s
```

You should see `agentx: warmup complete — N requests (N ok)` on stderr, then
a live progress line and finally the metrics table for the profiling window.

## 3. Run a benchmark

```bash
# Stream the full corpus straight from HuggingFace:
benchmaker agentx --concurrency 8

# Or pin a local corpus carved to the server's context window
# (a 128k window keeps the 82 traces that fit; see the table above):
benchmaker agentx --trace .local/agentx-traces-weka-062126.jsonl \
    --max-context 131072 --concurrency 8 --duration 1h
```

Defaults worth knowing: 8 lanes, a 1-hour profiling window, a 1800 s
per-request timeout (deep prefills are slow), and an admission RPS high
enough that the trees themselves throttle dispatch (closed-tree scheduling;
`--rate` caps it further if you want). Warmup + profiling are separate
passes — the reported metrics start at zero once the warmup pass ends.

**Choosing concurrency.** `--concurrency` is the AgentX client count, **not**
a request cap — the number of in-flight requests is whatever the recorded
trees produce at that client count (subagents multiply it). Sweep it the way
the methodology does: several client counts at fixed duration, comparing
interactivity (TTFT/ITL) at matched output throughput.

## Flags

Corpus selection:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--trace FILE` | – | Local WEKA-format JSONL (overrides `--dataset`). |
| `--dataset ID` | `semianalysisai/cc-traces-weka-062126` | HuggingFace dataset id; use `...-256k` for ~256k servers. |
| `--file NAME` | `traces.jsonl` | Trace file name inside the dataset repo. |
| `--max-context N` | – | Drop traces whose peak input + output exceeds N tokens (set to the server's context capacity). |
| `--max-traces N` | – | Cap the corpus (smoke tests only — changes the workload). |

Replay semantics:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--concurrency N` | 8 | Live session trees (AgentX client concurrency). |
| `--seed N` | 0 | Start points, recycle order, cache-bust markers. |
| `--start-min/max-ratio F` | 0.25 / 0.75 | Window sampling each lane's initial mid-session start point. |
| `--system-idle-gap-cap S` | 10 s | When *every* tree waits on a recorded gap longer than this, all pending timers shift earlier — real sessions span hours to days, so true idle time must be bounded. `0` disables. |
| `--no-cache-bust` | off | Disable per-play markers (reproduces wrap-around prefix reuse). |
| `--no-ignore-eos` | ignore_eos on | Let the server stop at EOS instead of decoding recorded lengths. |
| `--tokenizer ID` | chars mode | Exact prompt sizing with a HF tokenizer (needs `pip install -e .[tokenizer]`); otherwise 4 chars/token. |
| `--default-output-tokens N` | 256 | `max_tokens` for requests with no recorded output. |

Phases and run:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--warmup-requests N` | 10 | Warmup requests per lane after the primer (unmeasured). |
| `--duration` | 1h | Profiling-window length (`30s`, `2m`, …). |
| `--rate` | high | Load spec; the DAG's closed-tree scheduling throttles naturally. |
| `--timeout S` | 1800 | Per-request timeout. |
| `--out-dir DIR` | – | Write the run bundle under `<dir>/<run-id>/`. |

Prompt sizing reconciles with the recorded token counts the same way AIPerf's
WEKA loader does: blocks render at 64 tokens each; overhead beyond the block
structure becomes a per-request filler tail. On the real corpus this is a
no-op — every request's recorded input is exactly `hash_ids × 64` — so
shared prefixes are byte-exact everywhere. (Exact sizing only matters with
`--tokenizer`, where trim/pad reconciles the rendered text to the recorded
count per request.)

## 4. Read the results

`--out-dir results/x` writes the usual bundle (`meta.json`, `samples.jsonl`,
`summary.json`). `samples.jsonl` rows carry the agentx metadata in `meta`:

| Field | Meaning |
| --- | --- |
| `agentx_trace` / `agentx_play` | Source session and which replay of it. |
| `agentx_lane` | Which of the `--concurrency` trees. |
| `agentx_kind` / `agentx_agent_id` | `main` or `subagent` (+ the subagent stream id). |
| `agentx_req_index` / `agentx_main_index` | Index within its chain / absolute main-turn index. |
| `agentx_phase` | `profile` only — warmup rows are excluded from the bundle by design. |
| `agentx_recorded_input/output_tokens` | The recording's token accounting (compare with the server's `prompt_tokens` / `completion_tokens`). |
| `agentx_hash_blocks` | Recorded 64-token KV blocks reused by this request. |

The metrics table covers the profiling window only. Expect the subagent
streams to dominate dispatch: in the real corpus 1,697 subagent groups
interleave with 393 main chains, so most in-flight requests at any moment
are parallel subagent turns.

## YAML

The DAG advances through a completion post-hook; `benchmaker run` installs a
workload's declared hook automatically, so no extra wiring is needed:

```yaml
workload_type:
  type: openai-chat
  url: http://localhost:8000/v1/chat/completions
  model: your-model
  passthrough_meta: true
workload:
  type: agentx
  path: .local/agentx-traces-weka-062126.jsonl
  concurrency: 8
  warmup_requests: 10
  max_context: 131072   # carve to the server's window (128k here)
load: 10000          # high constant RPS; closed-tree scheduling throttles
duration: 1h
timeout_s: 1800
```

Note that `benchmaker run` measures one continuous window — use
`benchmaker agentx` (the recipe above) when you want the primer + warmup
pass separated from the measured profiling window, as the methodology
specifies.

## Python API

```python
from benchmaker.workloads.agentx import AgentXWorkload

workload = AgentXWorkload(
    ".local/agentx-traces-weka-062126.jsonl",
    concurrency=8,
    warmup_requests=10,
    max_context=131072,
)
# Pair with OpenAIChatWorkloadType(passthrough_meta=True) and install
# workload.completion_hook() as a runner post-hook — the DAG cannot advance
# without it. See docs/workloads.md for the full pattern.
```

## Notes & gotchas

* **Failures are not retried.** A failed request is recorded as completed
  (the DAG advances); a failed *warmup* request on a root chain aborts the
  run before the profiling window starts.
* **The idle guard is essential, not cosmetic.** Recorded sessions have a
  median wall-clock span of ~169 minutes (up to days). Without the
  `--system-idle-gap-cap` shift, replay would spend most of its time asleep.
* **Round order within a session is never shuffled** — only the corpus order
  across traces (`--shuffle/--no-shuffle`) and each lane's start point.
* **The corpus is sanitized**: no prompt text, code, tool inputs or paths.
  Everything sent to the server is synthesized from block ids.
