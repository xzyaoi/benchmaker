"""``agentx`` recipe — the SemiAnalysis AgentX v1.0 agentic-coding benchmark.

Replays the public WEKA-format coding-agent trace corpus
(``semianalysisai/cc-traces-weka-062126[-256k]``, also loadable from a local
JSONL) against an OpenAI-compatible endpoint, following the methodology at
https://inferencex.semianalysis.com/agentx:

* prompts are synthesized deterministically from the recorded session-scoped
  64-token block hashes, so real prefix-cache sharing is preserved;
* main-agent turns and subagent trees replay as a session DAG with recorded
  end-to-start inter-turn gaps and a system-idle bound;
* ``--concurrency`` is the number of live session trees (the AgentX "client"
  count — a tree may legitimately keep several subagent requests in flight);
* each lane is seeded to start 25–75% through its recorded session, dispatches
  a ``max_tokens=1`` primer plus ``--warmup-requests`` further requests as an
  **unmeasured warmup pass**, and only the following profiling window
  (default one hour) contributes reported metrics;
* every play of a trace carries a unique cache-bust marker so recycled plays
  never accumulate a shared prefix.

Run the profiler-facing summary at the end reports TTFT, ITL and decode
throughput over the profiling window.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any, Optional

import click

from benchmaker.core.load import ConstantRPS, parse_duration, parse_rate_spec
from benchmaker.core.runner import BenchConfig, BenchRunner
from benchmaker.recipes import register
from benchmaker.recipes._cli_shared import parse_headers, write_bundle_if_requested
from benchmaker.recipes.base import (
    DEFAULT_DURATION,
    DEFAULT_RATE,
    DEFAULT_TIMEOUT_S,
    BuildResult,
    Recipe,
    SharedOpts,
)
from benchmaker.workloads.datasets import Workload

DEFAULT_DATASET = "semianalysisai/cc-traces-weka-062126"
DEFAULT_FILE = "traces.jsonl"

# The AgentX dispatcher is paced by the session DAG (dependencies + recorded
# gaps), not by an arrival rate.  Admission is pull-driven: this open-loop
# ceiling just guarantees a driver is always available to dispatch the next
# eligible request the moment it becomes eligible.
_ADMISSION_RPS = 10000.0


class _KeepOpenWorkload(Workload):
    """Delegating wrapper whose ``aclose`` is a no-op.

    BenchRunner closes its workload when a pass ends; the warmup pass must
    leave the underlying AgentXWorkload (its mid-DAG lane state in particular)
    alive for the measured pass that follows.
    """

    def __init__(self, inner: Workload):
        self._inner = inner
        self.name = inner.name

    async def next_item(self) -> Any:
        return await self._inner.next_item()

    def completion_hook(self):
        return self._inner.completion_hook()

    async def aclose(self) -> None:
        return None


def _resolve_trace_source(trace: Optional[str], dataset: str, file: str) -> tuple[str, str]:
    """Return (local_path, description) for the corpus."""
    if trace is not None:
        return trace, f"local:{trace}"
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:  # pragma: no cover
        raise click.ClickException(
            "--dataset needs `huggingface_hub` (installed with benchmaker)."
        ) from e
    path = hf_hub_download(repo_id=dataset, filename=file, repo_type="dataset")
    return path, f"hf:{dataset}/{file}"


class AgentXRecipe(Recipe):
    name = "agentx"
    help = ("AgentX v1.0 agentic-coding benchmark: replay the SemiAnalysis WEKA "
            "trace corpus (block-hash synthetic prompts, subagent DAGs, seeded "
            "warmup, one profiling window).")

    def options(self) -> list:
        return [
            # --- endpoint ---
            click.option("--url", default=None,
                         help="Endpoint URL (e.g. http://host:8000/v1/chat/completions). "
                              "Falls back to $OPENAI_API_BASE_URL/$OPENAI_BASE_URL."),
            click.option("--model", default=None,
                         help="Model name sent in the request body (the model under test; "
                              "trace-recorded model names are NOT sent). Falls back to "
                              "$OPENAI_COMPATIBLE_MODEL/$OPENAI_MODEL."),
            click.option("--api-key", "api_key", default=None,
                         help="API key. Falls back to $OPENAI_API_KEY."),
            click.option("--header", "-H", "header", multiple=True,
                         help="Extra header 'Name: value'. Repeatable."),
            click.option("--temperature", type=float, default=0.0),
            click.option("--extra", "extras", multiple=True,
                         help="Extra sampling param 'key=value' (JSON or string)."),

            # --- dataset ---
            click.option("--trace", "trace", type=click.Path(exists=True, dir_okay=False),
                         default=None,
                         help="Local WEKA-format JSONL corpus (overrides --dataset)."),
            click.option("--dataset", "dataset", default=DEFAULT_DATASET,
                         show_default=True,
                         help="HuggingFace dataset id of the WEKA corpus. Use "
                              f"'semianalysisai/cc-traces-weka-062126-256k' when the server's "
                              "context window is ~256k."),
            click.option("--file", "file", default=DEFAULT_FILE, show_default=True,
                         help="Trace file name inside the dataset repo."),
            click.option("--max-context", "max_context", type=int, default=None,
                         help="Drop traces whose peak recorded input + output exceeds "
                              "this many tokens (set to the server's context capacity)."),
            click.option("--max-output-tokens", "max_output_tokens", type=int, default=None,
                         help="Ceiling on per-request max_tokens derived from the trace."),
            click.option("--max-traces", "max_traces", type=int, default=None,
                         help="Cap the number of corpus traces replayed (smoke tests only; "
                              "reducing the corpus changes the workload)."),
            click.option("--shuffle/--no-shuffle", default=True,
                         help="Shuffle the corpus trace order (seeded). Rounds within "
                              "each session always stay ordered."),

            # --- AgentX replay controls ---
            click.option("--concurrency", "concurrency", type=int, default=8,
                         show_default=True,
                         help="Number of live session trees (AgentX client concurrency). "
                              "Each tree may keep several subagent requests in flight."),
            click.option("--seed", type=int, default=0, show_default=True,
                         help="RNG seed: start points, recycle order and cache-bust "
                              "markers are deterministic given the corpus + seed."),
            click.option("--start-min-ratio", "start_min_ratio", type=float, default=0.25,
                         show_default=True,
                         help="Lower bound of the window sampling each lane's initial "
                              "start point within the recorded session."),
            click.option("--start-max-ratio", "start_max_ratio", type=float, default=0.75,
                         show_default=True,
                         help="Upper bound of the start-point window."),
            click.option("--warmup-requests", "warmup_requests", type=int, default=10,
                         show_default=True,
                         help="Warmup requests per lane after the primer (unmeasured)."),
            click.option("--cache-bust/--no-cache-bust", "cache_bust", default=True,
                         show_default=True,
                         help="Prepend a unique per-play [rid:...] marker so recycled "
                              "plays never share a KV prefix."),
            click.option("--system-idle-gap-cap", "idle_gap_cap", type=float, default=10.0,
                         show_default=True,
                         help="Bound on true system-idle time: when every tree waits on a "
                              "recorded gap longer than this, all pending timers shift "
                              "earlier. 0 disables the guard."),
            click.option("--no-ignore-eos", "ignore_eos", flag_value=False, default=True,
                         help="Let the server stop early at EOS instead of decoding the "
                              "recorded output length (the AgentX methodology decodes "
                              "fully; ignore_eos stays on by default)."),
            click.option("--default-output-tokens", "default_output_tokens", type=int,
                         default=256, show_default=True,
                         help="max_tokens for requests whose recorded output is missing."),

            # --- prompt sizing ---
            click.option("--chars-per-token", "chars_per_token", type=float, default=4.0,
                         show_default=True,
                         help="Char-mode token-size approximation. Ignored when "
                              "--tokenizer is set."),
            click.option("--tokenizer", "tokenizer", default=None,
                         help="HuggingFace tokenizer id for exact token-count prompt "
                              "sizing (needs `pip install -e .[tokenizer]`)."),
        ]

    def build(self, shared: SharedOpts, *, url, model, api_key, header, temperature,
              extras, trace, dataset, file, max_context, max_output_tokens,
              max_traces, concurrency, seed, start_min_ratio, start_max_ratio,
              warmup_requests, cache_bust, idle_gap_cap, ignore_eos,
              default_output_tokens, chars_per_token, tokenizer, shuffle) -> BuildResult:
        from benchmaker.workloads.agentx import AgentXWorkload
        from benchmaker.workloads.llm import OpenAIChatWorkloadType

        wt_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "timeout_s": shared.timeout_s,
            "headers": parse_headers(header),
            "passthrough_meta": True,
        }
        for item in extras:
            if "=" not in item:
                raise click.BadParameter(f"--extra must be 'key=value', got {item!r}")
            k, v = item.split("=", 1)
            try:
                parsed: Any = json.loads(v)
            except json.JSONDecodeError:
                parsed = v
            wt_kwargs[k.strip()] = parsed

        wt = OpenAIChatWorkloadType.from_env(
            url=url, model=model, api_key=api_key, dotenv_path=shared.dotenv,
            **wt_kwargs,
        )

        path, source_desc = _resolve_trace_source(trace, dataset, file)
        workload = AgentXWorkload(
            path=path,
            concurrency=concurrency,
            seed=seed,
            start_min_ratio=start_min_ratio,
            start_max_ratio=start_max_ratio,
            warmup_requests=warmup_requests,
            cache_bust=cache_bust,
            idle_gap_cap_s=idle_gap_cap,
            max_context=max_context,
            max_output_tokens=max_output_tokens,
            default_output_tokens=default_output_tokens,
            ignore_eos=ignore_eos,
            tokenizer=tokenizer,
            chars_per_token=chars_per_token,
            max_traces=max_traces,
            shuffle=shuffle,
        )

        source_config = {
            "recipe": "agentx",
            "dataset_source": source_desc,
            "workload_type": {
                "type": "openai-chat", "url": wt._url, "model": wt._model,
                **{k: v for k, v in wt_kwargs.items() if k != "headers"},
            },
            "workload": {
                "type": "agentx",
                "path": path,
                "concurrency": concurrency,
                "seed": seed,
                "start_min_ratio": start_min_ratio,
                "start_max_ratio": start_max_ratio,
                "warmup_requests": warmup_requests,
                "cache_bust": cache_bust,
                "idle_gap_cap_s": idle_gap_cap,
                "ignore_eos": ignore_eos,
                "default_output_tokens": default_output_tokens,
                "max_context": max_context,
                "max_output_tokens": max_output_tokens,
                "max_traces": max_traces,
                "chars_per_token": chars_per_token,
                "tokenizer": tokenizer,
                "shuffle": shuffle,
            },
        }
        return BuildResult(
            workload_type=wt,
            workload=workload,
            source_config=source_config,
            # AgentX defaults: DAG-paced admission (see _ADMISSION_RPS), a
            # one-hour profiling window, and a generous per-request timeout
            # for long prefills.
            default_rate=str(int(_ADMISSION_RPS)),
            default_duration="1h",
            default_timeout_s=1800.0,
        )

    def run(self, shared: SharedOpts, **params: Any) -> Optional[int]:
        """Warmup pass (unmeasured) followed by the measured profiling window."""
        import json as _json

        result = self.build(shared, **params)
        wl = result.workload
        hook = wl.completion_hook()

        # Shared-flag default resolution (mirrors recipes.base.run).
        rate = shared.rate
        if result.default_rate is not None and shared.rate == DEFAULT_RATE:
            rate = result.default_rate
        duration = shared.duration
        if result.default_duration is not None and shared.duration == DEFAULT_DURATION:
            duration = result.default_duration
        timeout_s = shared.timeout_s
        if result.default_timeout_s is not None and shared.timeout_s == DEFAULT_TIMEOUT_S:
            timeout_s = result.default_timeout_s

        dur_s = parse_duration(duration) if isinstance(duration, str) else duration
        load = parse_rate_spec(rate, duration_s=dur_s, max_requests=shared.max_requests)

        async def _main():
            # Phase 1 — warmup: primer + warmup_requests per lane, priming the
            # server's KV cache with realistic deep-prefix contexts. Ends via
            # StopAsyncIteration when every lane is through its warmup; the
            # metrics from this pass are discarded.
            warm_cfg = BenchConfig(
                workload_type=result.workload_type,
                workload=_KeepOpenWorkload(wl),
                load=ConstantRPS(_ADMISSION_RPS),
                post_hooks=[hook] if hook else [],
                timeout_s=timeout_s,
                connection_limit=shared.connection_limit,
                progress_every_s=0.0 if shared.quiet else 5.0,
            )
            warm_res = await BenchRunner(warm_cfg).run()
            n_warm = len(warm_res.samples)
            n_warm_ok = sum(1 for s in warm_res.samples if s.ok)
            print(f"agentx: warmup complete — {n_warm} requests "
                  f"({n_warm_ok} ok); excluded from reported metrics",
                  file=sys.stderr)

            # Phase 2 — profiling window: lanes resume their DAGs from exactly
            # where warmup left off; only this window is reported.
            wl.prepare_profile_phase()
            cfg = BenchConfig(
                workload_type=result.workload_type,
                workload=wl,
                load=load,
                post_hooks=[hook] if hook else [],
                timeout_s=timeout_s,
                connection_limit=shared.connection_limit,
                progress_every_s=0.0 if shared.quiet else 1.0,
            )
            runner = BenchRunner(cfg)
            await runner.run()
            return runner

        try:
            runner = asyncio.run(_main())
        except RuntimeError as e:
            # Root warmup failure (AgentX aborts rather than measure a
            # degraded tree pool).
            print(f"error: {e}", file=sys.stderr)
            return 1

        runner.metrics.render(sys.stdout)

        source_config = {
            **result.source_config,
            "load": rate,
            "duration": duration,
            "profile_window_s": dur_s,
            "max_requests": shared.max_requests,
            "timeout_s": timeout_s,
            "connection_limit": shared.connection_limit,
            "warmup": {"included_in_metrics": False},
        }
        write_bundle_if_requested(
            runner, source_config,
            shared.out_dir, shared.run_id, shared.labels, shared.notes,
        )
        return None


register(AgentXRecipe())
