# Changelog

## Unreleased

- Added the `agentx` recipe and workload (`benchmaker agentx`, YAML `type:
  agentx`): the SemiAnalysis AgentX v1.0 agentic-coding benchmark
  (https://inferencex.semianalysis.com/agentx). Replays the public WEKA-format
  Claude Code trace corpus (`semianalysisai/cc-traces-weka-062126[-256k]`, or
  a local JSONL) with block-hash synthetic prompts that preserve the
  recording's byte-exact prefix-cache structure, session-tree DAG dispatch
  (main turns + concurrent subagent streams with recorded spawn/join/gap
  timing and a system-idle cap), seeded mid-session start points, a per-lane
  primer + warmup pass excluded from metrics, per-play cache-bust markers,
  recorded decode lengths (`ignore_eos`), and a default one-hour profiling
  window. Includes `tools/agentx/prepare.py` for corpus download/subsetting
  and `docs/agentx.md`.
- **Breaking:** renamed the `trajectory-replay` workload to `agentic`, since it
  represents multi-turn *agentic* workloads. The CLI recipe is now
  `benchmaker agentic` (was `trajectory-replay`); the YAML workload `type:
  trajectory` is now `type: agentic`; and `TrajectoryReplayWorkload` is now
  `AgenticWorkload` (in `benchmaker.workloads.agentic`). The underlying
  recorded-data concept (a "trajectory" of agent turns) and its flags
  (`--max-trajectories`, `--max-turns-per-trajectory`, `expand_trajectory`)
  are unchanged. The separate SWE-bench `trajectory` module and
  `swebench-replay` recipe are unaffected.
- Fixed `OpenAIChatWorkloadType` silently ignoring `delta.reasoning_content`
  for thinking models (GLM-4.x, DeepSeek-R1, Qwen3-thinking, gpt-5 reasoning,
  …), which made `ttft_s`, `itl_ms_*`, and (when `usage` was absent)
  `tokens_out` wrong by the entire reasoning phase. Reasoning tokens are now
  counted the same as content tokens for TTFT/ITL, the
  `usage.completion_tokens_details.reasoning_tokens` breakdown is surfaced
  (`reasoning_tokens` / `content_tokens`), and a new `ttft_token`
  (`"any"` | `"content"`, default `"any"`) knob selects whether the headline
  `ttft_s` measures the first token of any kind or the first visible content
  token. The first-content time is always available separately as
  `content_ttft_s` (#14).

## 0.1.4 — 2026-06-24

- Fixed silent loss of streamed `usage` (real `prompt_tokens`, `cached_tokens`)
  and truncated eval text when an SSE event was split across HTTP chunk
  boundaries — common on the `mix` path under high concurrency with large
  prompts, which zeroed prefix-cache hit rates. SSE parsing now reassembles
  lines across chunks (shared by the `openai-chat`, `sglang`, and eval paths),
  and the `openai-chat` workload warns once when `include_usage` was requested
  but no usage block was parsed (#13).
- Added `DeepRAGWorkload` and a HotpotQA distractor JSONL preparation tool for
  prefill-heavy retrieval benchmarks.
- Added concurrent named workload lanes with independent load models, durable
  `meta.lane` tags, and per-lane metric summaries.
- Added a phase-swinging DeepRAG + ShareGPT configuration example.
