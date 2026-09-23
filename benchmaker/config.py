"""YAML config loading: build a BenchConfig from a dict.

Config shape:

    workload_type:               # how to talk
        type: http | openai | ...
        ...kwargs...
    workload:                    # what to send  (optional; defaults to one None item)
        type: static | jsonl | callable
        ...kwargs...
    load: <rate spec>            # when to fire
    duration: 30s
    pre_hooks: [module:fn, ...]
    post_hooks: [module:fn, ...]
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Optional

from benchmaker.env import interpolate, load_dotenv
from benchmaker.core.load import parse_duration, parse_rate_spec
from benchmaker.core.monitors import FunctionMonitor, Monitor, PrometheusMonitor
from benchmaker.core.runner import BenchConfig, BenchLane
from benchmaker.workloads.base import WorkloadType
from benchmaker.workloads.datasets import (
    CallableWorkload,
    JsonlWorkload,
    StaticWorkload,
    Workload,
)
from benchmaker.workloads.hf import HFDatasetWorkload
from benchmaker.workloads.rag import DeepRAGWorkload
from benchmaker.workloads.tracelab import TraceLabWorkload
from benchmaker.workloads.http import HttpWorkloadType
from benchmaker.workloads.llm import OpenAIChatWorkloadType
from benchmaker.workloads.sandbox import SandboxWorkloadType
from benchmaker.workloads.agent import Agent, AgentWorkloadType
from benchmaker.workloads.eval import (
    EvalWorkloadType,
    Scorer,
    contains,
    correctness_hook,
    exact_match,
    json_valid,
    judge_llm,
    multiple_choice,
    openai_chat_judge,
    regex_match,
)
from benchmaker.core.trace import (
    ReplayWorkloadType,
    TracePacedLoad,
    TraceRecorder,
    TraceWorkload,
    load_trace,
)


def resolve_callable(ref: str) -> Callable:
    if ":" in ref:
        modname, attr = ref.split(":", 1)
    else:
        modname, _, attr = ref.rpartition(".")
        if not modname:
            raise ValueError(f"Cannot resolve callable {ref!r}")
    mod = importlib.import_module(modname)
    obj: Any = mod
    for part in attr.split("."):
        obj = getattr(obj, part)
    if not callable(obj):
        raise TypeError(f"{ref!r} is not callable")
    return obj


def build_workload_type(spec: dict) -> WorkloadType:
    """Build a WorkloadType from a dict."""
    if "factory" in spec:
        fn = resolve_callable(spec["factory"])
        kwargs = {k: v for k, v in spec.items() if k != "factory"}
        obj = fn(**kwargs)
        if not isinstance(obj, WorkloadType):
            raise TypeError(f"factory must return a WorkloadType, got {type(obj)}")
        return obj

    t = (spec.get("type") or "http").lower()
    kwargs = {k: v for k, v in spec.items() if k != "type"}
    if t == "http":
        return HttpWorkloadType(**kwargs)
    if t in ("openai", "openai-chat", "llm-chat", "llm"):
        return OpenAIChatWorkloadType(**kwargs)
    if t in ("sglang", "sglang-generate"):
        from benchmaker.workloads.sglang import SGLangGenerateWorkloadType
        return SGLangGenerateWorkloadType(**kwargs)
    if t in ("sandbox", "flash-sandbox"):
        return SandboxWorkloadType(**kwargs)
    if t == "agent":
        return _build_agent_workload_type(kwargs)
    raise ValueError(f"Unknown workload_type {t!r}")


def _build_agent_workload_type(spec: dict) -> AgentWorkloadType:
    """Build an `AgentWorkloadType` from YAML.

    Accepts ``agent: 'module:ClassOrCallable'`` (resolved via `resolve_callable`)
    plus optional ``agent_kwargs`` forwarded to the constructor. The remaining
    keys are passed to ``AgentWorkloadType`` (``reference_key``,
    ``extra_meta_keys``, ``name``).
    """
    ref = spec.pop("agent", None) or spec.pop("class", None)
    if not ref:
        raise ValueError(
            "agent workload-type requires 'agent: <module:ClassOrCallable>'"
        )
    obj: Any = resolve_callable(ref) if isinstance(ref, str) else ref
    agent_kwargs = spec.pop("agent_kwargs", None) or {}
    return AgentWorkloadType(obj, agent_kwargs=agent_kwargs, **spec)


def build_workload(spec: Any) -> Workload:
    """Build a Workload (dataset) from a dict, list, or string."""
    if spec is None:
        return StaticWorkload()  # one None item, cycled

    # Convenience: a bare list becomes a StaticWorkload.
    if isinstance(spec, list):
        return StaticWorkload(items=spec)

    # Convenience: a bare string ending in .jsonl becomes a JsonlWorkload.
    if isinstance(spec, str):
        if spec.endswith(".jsonl"):
            return JsonlWorkload(path=spec)
        # Otherwise treat as a single static string item.
        return StaticWorkload(items=[spec])

    if not isinstance(spec, dict):
        raise TypeError(f"workload spec must be dict|list|str|None, got {type(spec)}")

    if "factory" in spec:
        fn = resolve_callable(spec["factory"])
        kwargs = {k: v for k, v in spec.items() if k != "factory"}
        obj = fn(**kwargs)
        if not isinstance(obj, Workload):
            raise TypeError(f"factory must return a Workload, got {type(obj)}")
        return obj

    t = (spec.get("type") or "static").lower()
    kwargs = {k: v for k, v in spec.items() if k != "type"}
    if t == "static":
        return StaticWorkload(**kwargs)
    if t == "jsonl":
        return JsonlWorkload(**kwargs)
    if t == "callable":
        fn = resolve_callable(kwargs.pop("fn"))
        return CallableWorkload(fn=fn, **kwargs)
    if t in ("hf", "huggingface"):
        return HFDatasetWorkload(**kwargs)
    if t in ("deeprag", "deep-rag", "rag"):
        return DeepRAGWorkload(**kwargs)
    if t in ("tracelab", "trace-lab", "trace"):
        return TraceLabWorkload(**kwargs)
    if t == "agentic":
        from benchmaker.workloads.agentic import AgenticWorkload
        return AgenticWorkload(**kwargs)
    if t in ("agentx", "agent-x"):
        from benchmaker.workloads.agentx import AgentXWorkload
        return AgentXWorkload(**kwargs)
    raise ValueError(f"Unknown workload type {t!r}")


def build_monitor(spec: dict) -> Monitor:
    """Build a Monitor from a dict.

    Forms:
        type: prometheus     url=..., metric_names=[...], interval_s=...
        type: function       fn='module:func', interval_s=...
        factory: 'module:fn' kwargs...
    """
    if "factory" in spec:
        fn = resolve_callable(spec["factory"])
        kwargs = {k: v for k, v in spec.items() if k != "factory"}
        obj = fn(**kwargs)
        if not isinstance(obj, Monitor):
            raise TypeError(f"monitor factory must return a Monitor, got {type(obj)}")
        return obj

    t = (spec.get("type") or "function").lower()
    kwargs = {k: v for k, v in spec.items() if k != "type"}
    if t == "prometheus":
        if "metric_names" in kwargs and kwargs["metric_names"] is not None:
            kwargs["metric_names"] = set(kwargs["metric_names"])
        return PrometheusMonitor(**kwargs)
    if t == "function":
        fn = resolve_callable(kwargs.pop("fn"))
        return FunctionMonitor(fn=fn, **kwargs)
    raise ValueError(f"Unknown monitor type {t!r}")


_SCORER_BUILDERS: dict[str, Any] = {
    "exact_match":     lambda kw: exact_match(**kw),
    "exact":           lambda kw: exact_match(**kw),
    "contains":        lambda kw: contains(**kw),
    "regex_match":     lambda kw: regex_match(**kw),
    "regex":           lambda kw: regex_match(**kw),
    "json_valid":      lambda kw: json_valid(**kw),
    "multiple_choice": lambda kw: multiple_choice(**kw),
}


def build_scorer(spec: Any) -> tuple[Scorer, Optional[Callable[[], Any]]]:
    """Build a (scorer, optional_aclose) pair from a YAML spec.

    Spec forms:
        type: exact_match | contains | regex | json_valid | multiple_choice | judge_llm
        # ...type-specific kwargs

    OR:
        factory: 'module:fn'
        # ...kwargs forwarded to the factory

    A bare string ("exact_match") is shorthand for `{type: <that>}`.

    The aclose callable, when not None, owns transient resources held by the
    scorer (e.g. a judge LLM's httpx2 client). The YAML build path wires it
    into the wrapped workload-type's `aclose` chain.
    """
    if spec is None:
        raise ValueError("correctness.scorer must be set")
    if isinstance(spec, str):
        spec = {"type": spec}
    if not isinstance(spec, dict):
        raise TypeError(f"scorer spec must be dict|str, got {type(spec).__name__}")

    if "factory" in spec:
        fn = resolve_callable(spec["factory"])
        kwargs = {k: v for k, v in spec.items() if k != "factory"}
        obj = fn(**kwargs)
        # Factories may return either (scorer, aclose) or just scorer.
        if isinstance(obj, tuple) and len(obj) == 2 and callable(obj[0]):
            return obj  # type: ignore[return-value]
        if callable(obj):
            return obj, None
        raise TypeError(
            f"scorer factory {spec['factory']!r} must return a callable "
            f"(or (callable, aclose) tuple), got {type(obj).__name__}"
        )

    t = (spec.get("type") or "").lower()
    if t in _SCORER_BUILDERS:
        kwargs = {k: v for k, v in spec.items() if k != "type"}
        return _SCORER_BUILDERS[t](kwargs), None

    if t in ("judge_llm", "judge"):
        return _build_judge_scorer(spec)

    raise ValueError(f"unknown scorer type {t!r}")


def _build_judge_scorer(spec: dict) -> tuple[Scorer, Optional[Callable[[], Any]]]:
    """Construct a judge_llm scorer.

    Either `send_factory: 'module:fn'` (a factory returning a `send` callable or
    `(send, aclose)`) OR `openai_chat: {url, model, api_key, ...}` shortcut.
    """
    template = spec.get("template")
    max_concurrency = int(spec.get("max_concurrency", 4))
    parse = None
    if "parse_factory" in spec:
        parse = resolve_callable(spec["parse_factory"])

    aclose: Optional[Callable[[], Any]] = None

    if "send_factory" in spec:
        fn = resolve_callable(spec["send_factory"])
        kwargs = spec.get("send_kwargs", {}) or {}
        result = fn(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            send, aclose = result
        else:
            send = result
    elif "openai_chat" in spec:
        oc = dict(spec["openai_chat"])
        send, aclose = openai_chat_judge(**oc)
    else:
        raise ValueError(
            "judge_llm scorer requires either 'send_factory' or 'openai_chat'"
        )

    kwargs: dict[str, Any] = {"max_concurrency": max_concurrency}
    if template is not None:
        kwargs["template"] = template
    if parse is not None:
        kwargs["parse"] = parse
    return judge_llm(send, **kwargs), aclose


def apply_correctness(workload_type: WorkloadType, spec: dict
                      ) -> tuple[WorkloadType, list]:
    """Install correctness grading on a workload-type.

    If the workload-type sets ``handles_reference = True`` (it already peels
    ``reference`` out of items into ``Request.meta``), we install just the
    post-hook. Otherwise we wrap the workload-type in ``EvalWorkloadType``
    so it gets the reference plumbing for free.

    Returns ``(workload_type_to_use, [hook])``. If a scorer owns transient
    resources (e.g. a judge session), its ``aclose`` is chained onto the
    workload-type's ``aclose`` so the runner cleans it up.
    """
    reference_key = spec.get("reference_key", "reference")
    extra_meta_keys = tuple(spec.get("extra_meta_keys") or ())
    gate_key = spec.get("gate_key", "correct")
    if gate_key in ("", "null", "none"):
        gate_key = None
    prefix = spec.get("prefix", "")
    require_reference = bool(spec.get("require_reference", True))
    max_prediction_chars: Any = spec.get("max_prediction_chars", 2048)
    if isinstance(max_prediction_chars, str) and max_prediction_chars.lower() in (
        "", "null", "none", "all", "full"
    ):
        max_prediction_chars = None

    scorer_spec = spec.get("scorer")
    scorer, scorer_aclose = build_scorer(scorer_spec)

    if getattr(workload_type, "handles_reference", False):
        wrapped = workload_type
    else:
        wrapped = EvalWorkloadType(
            workload_type,
            reference_key=reference_key,
            extra_meta_keys=extra_meta_keys,
        )

    if scorer_aclose is not None:
        original_aclose = wrapped.aclose

        async def _chained() -> None:
            try:
                await original_aclose()
            finally:
                res = scorer_aclose()
                if hasattr(res, "__await__"):
                    await res

        wrapped.aclose = _chained  # type: ignore[method-assign]

    hook = correctness_hook(
        scorer,
        reference_key=reference_key,
        gate_key=gate_key,
        prefix=prefix,
        require_reference=require_reference,
        max_prediction_chars=max_prediction_chars,
    )
    return wrapped, [hook]


def build_config(cfg: dict, dotenv_path: Optional[str] = ".env",
                 interpolate_env: bool = True) -> BenchConfig:
    """Build a BenchConfig from a (typically YAML-loaded) dict.

    Args:
        cfg: the config dict.
        dotenv_path: if non-None and the file exists, KEY=VALUE pairs are
            loaded into `os.environ` (existing vars are NOT overwritten).
        interpolate_env: if True, `${VAR}` and `${VAR:-default}` are
            substituted in all string values throughout the config.
    """
    if dotenv_path:
        load_dotenv(dotenv_path)
    if interpolate_env:
        cfg = interpolate(cfg)

    replay_spec = cfg.get("replay")
    mix_spec = cfg.get("mix")
    if replay_spec is not None and mix_spec is not None:
        raise ValueError("'replay' and 'mix' are mutually exclusive")
    if replay_spec is not None:
        workload_type, workload, load_model = _build_replay(replay_spec)
        lanes: list[BenchLane] = []
    else:
        wt_spec = cfg.get("workload_type")
        if not wt_spec:
            # Back-compat shim: if old 'workload' key looks like a workload-type config
            # (has 'type: http' or 'type: openai'), accept it.
            legacy = cfg.get("workload")
            if isinstance(legacy, dict) and legacy.get("type", "http").lower() in (
                "http", "openai", "openai-chat", "llm", "llm-chat"
            ):
                wt_spec = legacy
                cfg = {**cfg, "workload": None}
            else:
                raise ValueError("config must define 'workload_type' or 'replay'")

        workload_type = build_workload_type(wt_spec)
        duration = cfg.get("duration") or cfg.get("duration_s")
        if duration is not None and isinstance(duration, str):
            duration = parse_duration(duration)
        if mix_spec is not None:
            if cfg.get("load") is not None:
                raise ValueError("a mixed config cannot also define top-level 'load'")
            workload = StaticWorkload()
            load_model = None
            lanes = _build_lanes(
                mix_spec,
                duration_s=duration,
                max_requests=cfg.get("max_requests"),
            )
        else:
            workload = build_workload(cfg.get("workload"))
            load_spec = cfg.get("load")
            if load_spec is None:
                raise ValueError("config must define 'load' or 'mix.lanes'")
            load_model = parse_rate_spec(load_spec, duration_s=duration,
                                         max_requests=cfg.get("max_requests"))
            lanes = []

    pre_hooks = [resolve_callable(h) for h in (cfg.get("pre_hooks") or [])]
    post_hooks = [resolve_callable(h) for h in (cfg.get("post_hooks") or [])]
    monitors = [build_monitor(m) for m in (cfg.get("monitors") or [])]

    # Correctness block. In normal mode it wraps the workload-type (to split
    # references out of items); in replay mode the recorded Request already
    # carries the reference under `meta`, so we just install the post-hook.
    correctness_spec = cfg.get("correctness")
    if correctness_spec:
        if not isinstance(correctness_spec, dict):
            raise TypeError("'correctness' must be a dict")
        workload_type, extra_post = apply_correctness(workload_type, correctness_spec)
        post_hooks = list(post_hooks) + list(extra_post)

    # A workload that schedules on per-request completion (e.g. interleaved
    # agentic replay) declares the post-hook it needs; install it so a YAML
    # config can't silently stall waiting for a signal it never wired up.
    workloads = [lane.workload for lane in lanes] if lanes else [workload]
    for lane_workload in workloads:
        wl_hook = lane_workload.completion_hook()
        if wl_hook is not None and wl_hook not in post_hooks:
            post_hooks = list(post_hooks) + [wl_hook]

    recorder = _build_recorder(cfg.get("record"))

    return BenchConfig(
        workload_type=workload_type,
        workload=workload,
        load=load_model,
        lanes=lanes,
        pre_hooks=pre_hooks,
        post_hooks=post_hooks,
        monitors=monitors,
        recorder=recorder,
        connection_limit=int(cfg.get("connection_limit", 1000)),
        timeout_s=float(cfg.get("timeout_s", 60.0)),
        max_in_flight=int(cfg.get("max_in_flight", 10000)),
        progress_every_s=float(cfg.get("progress_every_s", 1.0)),
        stop_on_exhausted=bool(cfg.get("stop_on_exhausted", True)),
    )


def _build_lanes(spec: Any, *, duration_s: Optional[float],
                 max_requests: Optional[int]) -> list[BenchLane]:
    """Build independent workload/load pairs from a ``mix:`` YAML block."""
    if not isinstance(spec, dict):
        raise TypeError("'mix' must be a mapping with a 'lanes' list")
    lane_specs = spec.get("lanes")
    if not isinstance(lane_specs, list) or not lane_specs:
        raise ValueError("'mix.lanes' must be a non-empty list")

    lanes: list[BenchLane] = []
    for index, lane_spec in enumerate(lane_specs):
        if not isinstance(lane_spec, dict):
            raise TypeError(f"mix.lanes[{index}] must be a mapping")
        name = lane_spec.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"mix.lanes[{index}].name must be a non-empty string")
        if "workload" not in lane_spec:
            raise ValueError(f"mix.lanes[{index}] must define a workload")
        rate = lane_spec.get("rate", lane_spec.get("load"))
        if rate is None:
            raise ValueError(f"mix.lanes[{index}] must define rate (or load)")

        lane_duration = lane_spec.get("duration", duration_s)
        if isinstance(lane_duration, str):
            lane_duration = parse_duration(lane_duration)
        lane_max_requests = lane_spec.get("max_requests", max_requests)
        lanes.append(BenchLane(
            name=name,
            workload=build_workload(lane_spec["workload"]),
            load=parse_rate_spec(
                rate,
                duration_s=lane_duration,
                max_requests=lane_max_requests,
            ),
        ))
    return lanes


def _build_recorder(spec: Any) -> Optional[TraceRecorder]:
    if spec is None:
        return None
    path = spec if isinstance(spec, str) else spec.get("path")
    if not path:
        raise ValueError("'record' must specify a 'path' (or be a bare string path)")
    return TraceRecorder(path)


def _build_replay(spec: Any) -> tuple[WorkloadType, Workload, Any]:
    """Resolve a `replay:` block into (workload_type, workload, load_model)."""
    if isinstance(spec, str):
        spec = {"path": spec}
    if not isinstance(spec, dict):
        raise TypeError(f"'replay' must be dict|str, got {type(spec).__name__}")
    path = spec.get("path")
    if not path:
        raise ValueError("'replay' must specify a 'path'")
    speed = float(spec.get("speed", 1.0))
    streaming = bool(spec.get("streaming", False))
    trace = load_trace(path)
    return (
        ReplayWorkloadType(streaming=streaming),
        TraceWorkload(trace),
        TracePacedLoad(trace, speed=speed),
    )

