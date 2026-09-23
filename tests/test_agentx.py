"""AgentX (SemiAnalysis InferenceX) workload + recipe tests."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from benchmaker.workloads.agentx import (
    AgentXWorkload,
    cache_bust_marker,
    parse_trace,
    _synth_text,
)


# ---------------------------------------------------------------------------
# Fixtures: WEKA-format trace builders
# ---------------------------------------------------------------------------

def _main(i, t, *, in_=None, out=32, blocks=None, api_time=0.0,
          model="opus") -> dict:
    if in_ is None:
        in_ = 64 * (i + 1)
    return {
        "t": t, "type": "n", "model": model, "in": in_, "out": out,
        "hash_ids": blocks if blocks is not None else list(range(i * 10, i * 10 + max(1, in_ // 64))),
        "api_time": api_time,
    }


def _subagent(t, agent_id, inner, *, model="haiku") -> dict:
    return {
        "t": t, "type": "subagent", "agent_id": agent_id, "requests": inner,
    }


def _inner(t, in_, out, blocks, *, api_time=0.0, model="haiku") -> dict:
    return {"t": t, "type": "n", "model": model, "in": in_, "out": out,
            "hash_ids": blocks, "api_time": api_time}


def _trace(trace_id: str, requests: list[dict]) -> str:
    """Write a one-trace corpus; all-zero timestamps keep gaps out of the way."""
    return json.dumps({"id": trace_id, "block_size": 64, "hash_id_scope": "local",
                       "requests": requests})


def _corpus(tmp_path: Path, traces: list[str]) -> str:
    p = tmp_path / "traces.jsonl"
    p.write_text("\n".join(traces) + "\n")
    return str(p)


# Growing-prefix main chain: turn i extends turn i-1's blocks.
def _chain_trace(trace_id: str, n_turns: int, *, t0: float = 0.0,
                 gap: float = 0.0) -> str:
    reqs = []
    for i in range(n_turns):
        reqs.append(_main(i, t0 + i * gap, blocks=list(range(i + 1))))
    return _trace(trace_id, reqs)


def _dag_trace(trace_id: str) -> str:
    # Two subagent streams spawn together after main turn 0 and join before
    # main turn 1. Timestamps must be strictly increasing across the join
    # boundary for the parser to anchor the join.
    return _trace(trace_id, [
        _main(0, 0.0, in_=128, blocks=[0, 1]),
        _subagent(0.01, "a1", [_inner(0.01, 64, 8, [5])]),
        _subagent(0.01, "a2", [_inner(0.01, 64, 8, [6])]),
        _main(1, 0.02, in_=192, blocks=[0, 1, 2]),
    ])


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_parse_trace_anchors_subagent_spawn_and_join():
    row = json.loads(_trace("t1", [
        _main(0, 0.0),
        _subagent(2.0, "a1", [_inner(2.0, 64, 10, [10]), _inner(3.0, 128, 10, [10, 11])]),
        _main(1, 6.0),
        _main(2, 9.0),
    ]))
    tr = parse_trace(row)
    assert len(tr.main.requests) == 3
    assert len(tr.groups) == 1
    g = tr.groups[0]
    assert g.spawn_after == 0
    assert g.join_at == 1          # joins before the NEXT dependent main turn
    assert len(g.children) == 1
    assert g.children[0].session_id == "t1::sa:a1"
    assert [r.t for r in g.children[0].requests] == [2.0, 3.0]


def test_parse_trace_drops_subagent_without_parent_and_keeps_background():
    row = json.loads(_trace("t2", [
        _subagent(0.0, "a0", [_inner(0.0, 64, 10, [9])]),
        _main(0, 1.0),
        _subagent(2.0, "a1", [_inner(2.0, 64, 10, [8])]),   # trailing: no join
    ]))
    tr = parse_trace(row)
    assert len(tr.main.requests) == 1
    assert len(tr.groups) == 1
    assert tr.groups[0].join_at is None


def test_parse_trace_detects_relative_inner_timestamps():
    row = json.loads(_trace("t3", [
        _main(0, 10.0),
        _subagent(10.0, "a1", [_inner(0.0, 64, 10, [7]), _inner(0.5, 64, 10, [8])]),
        _main(1, 20.0),
    ]))
    tr = parse_trace(row)
    child = tr.groups[0].children[0]
    # Relative inner times were canonicalized onto the root-absolute basis.
    assert [r.t for r in child.requests] == [10.0, 10.5]


def test_parse_trace_adjacent_subagents_form_separate_groups_same_anchor():
    row = json.loads(_trace("t4", [
        _main(0, 0.0),
        _subagent(1.0, "a", [_inner(1.0, 64, 10, [1])]),
        _subagent(1.5, "b", [_inner(1.5, 64, 10, [2])]),
        _main(1, 5.0),
    ]))
    tr = parse_trace(row)
    assert len(tr.groups) == 2
    assert {g.spawn_after for g in tr.groups} == {0}
    assert {g.join_at for g in tr.groups} == {1}


# ---------------------------------------------------------------------------
# Synthetic payloads
# ---------------------------------------------------------------------------

def test_block_text_deterministic_and_keyed():
    assert _synth_text("t1", 5, 256) == _synth_text("t1", 5, 256)
    assert _synth_text("t1", 5, 256) != _synth_text("t1", 6, 256)
    # hash ids are trace-scoped: same id under a different trace differs.
    assert _synth_text("t1", 5, 256) != _synth_text("t2", 5, 256)


def test_marker_differs_per_play_and_is_stable_within_play():
    assert cache_bust_marker(1, "t", 0) != cache_bust_marker(1, "t", 1)
    assert cache_bust_marker(1, "t", 0) == cache_bust_marker(1, "t", 0)
    assert cache_bust_marker(1, "t", 0) != cache_bust_marker(2, "t", 0)


@pytest.mark.asyncio
async def test_prompt_sized_to_recorded_input_tokens(tmp_path):
    path = _corpus(tmp_path, [_chain_trace("t1", 2)])
    wl = AgentXWorkload(path=path, concurrency=1, seed=0, warmup_requests=0,
                        chars_per_token=4.0)
    item = await wl.next_item()
    content = item["messages"][0]["content"]
    # in = 64 tokens -> 256 chars at 4 chars/token; the per-play marker is
    # compensated inside the same budget.
    assert len(content) == 64 * 4
    assert item["meta"]["agentx_hash_blocks"] == 1
    assert item["meta"]["agentx_recorded_input_tokens"] == 64


@pytest.mark.asyncio
async def test_growing_blocks_share_byte_exact_prefixes(tmp_path):
    # Turn 1's hash list extends turn 0's, so (within one play) turn 1's
    # prompt must start with turn 0's prompt.
    path = _corpus(tmp_path, [_chain_trace("t1", 2)])
    wl = AgentXWorkload(path=path, concurrency=1, seed=0, warmup_requests=0,
                        start_min_ratio=0.0, start_max_ratio=0.0)
    a = await wl.next_item()
    wl.note_complete(a["meta"], True)
    # warmup_requests=0 -> the primer alone ends the warmup pass; the second
    # dispatch comes from the measured pass but the same play, so the prefix
    # relationship still holds.
    wl.prepare_profile_phase()
    b = await wl.next_item()
    ca = a["messages"][0]["content"]
    cb = b["messages"][0]["content"]
    assert cb.startswith(ca)


# ---------------------------------------------------------------------------
# DAG replay: spawn / join / recycle
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_subagent_children_dispatch_before_joining_main_turn(tmp_path):
    path = _corpus(tmp_path, [_dag_trace("d1"), _chain_trace("d2", 2)])
    wl = AgentXWorkload(path=path, concurrency=1, seed=0, warmup_requests=0,
                        start_min_ratio=0.0, start_max_ratio=0.0, shuffle=False)
    wl.warmup_only = False

    root = await wl.next_item()
    assert root["meta"]["agentx_chain"].endswith("::main")
    assert root["meta"]["agentx_kind"] == "main"
    # Primer: first dispatch of the initial play decodes one token.
    assert root["max_tokens"] == 1

    # Main turn 1 is gated: children must run first.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(wl.next_item(), timeout=0.15)

    wl.note_complete(root["meta"], True)

    c1 = await wl.next_item()
    assert c1["meta"]["agentx_kind"] == "subagent"
    assert c1["meta"]["agentx_agent_id"] == "a1"
    c2 = await wl.next_item()
    assert c2["meta"]["agentx_kind"] == "subagent"
    # Both children of the group are in flight concurrently.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(wl.next_item(), timeout=0.15)

    wl.note_complete(c1["meta"], True)
    wl.note_complete(c2["meta"], True)

    join = await wl.next_item()
    assert join["meta"]["agentx_kind"] == "main"
    assert join["meta"]["agentx_req_index"] == 1
    assert join["max_tokens"] == 32  # recorded out, primer only on turn 0
    wl.note_complete(join["meta"], True)

    # Tree drained -> the lane recycles to the next trace at turn 0.
    nxt = await wl.next_item()
    assert nxt["meta"]["agentx_trace"] == "d2"
    assert nxt["meta"]["agentx_play"] == 1
    assert nxt["meta"]["agentx_phase"] == "profile"


def _slow_child_trace(trace_id: str) -> str:
    # One subagent stream with two sequential turns; the join turn must wait
    # for the whole stream, not just its first request.
    return _trace(trace_id, [
        _main(0, 0.0, blocks=[0]),
        _subagent(0.01, "a1", [_inner(0.01, 64, 8, [5]), _inner(0.02, 64, 8, [6])]),
        _main(1, 0.03, blocks=[0, 1]),
    ])


@pytest.mark.asyncio
async def test_join_waits_for_slow_children(tmp_path):
    path = _corpus(tmp_path, [_slow_child_trace("s1"), _chain_trace("s2", 2)])
    wl = AgentXWorkload(path=path, concurrency=1, seed=0, warmup_requests=0,
                        start_min_ratio=0.0, start_max_ratio=0.0, shuffle=False)
    wl.warmup_only = False

    root = await wl.next_item()
    wl.note_complete(root["meta"], True)
    c1 = await wl.next_item()
    wl.note_complete(c1["meta"], True)
    c2 = await wl.next_item()
    # Second child still pending -> join turn stays gated.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(wl.next_item(), timeout=0.15)
    wl.note_complete(c2["meta"], True)
    join = await wl.next_item()
    assert join["meta"]["agentx_kind"] == "main"


@pytest.mark.asyncio
async def test_recycled_play_gets_fresh_marker_and_no_primer(tmp_path):
    path = _corpus(tmp_path, [_chain_trace("r1", 2)])
    wl = AgentXWorkload(path=path, concurrency=1, seed=0, warmup_requests=0,
                        start_min_ratio=0.0, start_max_ratio=0.0,
                        cache_bust=True)
    wl.warmup_only = False

    a = await wl.next_item()
    wl.note_complete(a["meta"], True)
    b = await wl.next_item()
    wl.note_complete(b["meta"], True)
    again = await wl.next_item()  # play 1 of the same trace

    assert again["meta"]["agentx_play"] == 1
    assert again["max_tokens"] != 1  # no primer on recycled plays
    ca = a["messages"][0]["content"]
    cn = again["messages"][0]["content"]
    assert ca != cn                 # new marker -> no cross-play prefix sharing
    assert again["meta"]["conversation_id"] != a["meta"]["conversation_id"]


@pytest.mark.asyncio
async def test_cache_bust_off_reproduces_identical_prompts(tmp_path):
    path = _corpus(tmp_path, [_chain_trace("r1", 2)])
    wl = AgentXWorkload(path=path, concurrency=1, seed=0, warmup_requests=0,
                        start_min_ratio=0.0, start_max_ratio=0.0,
                        cache_bust=False)
    wl.warmup_only = False
    a = await wl.next_item()
    wl.note_complete(a["meta"], True)
    b = await wl.next_item()
    wl.note_complete(b["meta"], True)
    again = await wl.next_item()
    assert again["messages"][0]["content"] == a["messages"][0]["content"]


@pytest.mark.asyncio
async def test_multiple_lanes_run_distinct_trees(tmp_path):
    path = _corpus(tmp_path, [_chain_trace("m1", 2), _chain_trace("m2", 2)])
    wl = AgentXWorkload(path=path, concurrency=2, seed=0, warmup_requests=0,
                        start_min_ratio=0.0, start_max_ratio=0.0)
    wl.warmup_only = False
    a = await wl.next_item()
    b = await wl.next_item()
    assert {a["meta"]["agentx_trace"], b["meta"]["agentx_trace"]} == {"m1", "m2"}
    assert a["meta"]["agentx_lane"] != b["meta"]["agentx_lane"]


@pytest.mark.asyncio
async def test_seeded_start_point_skips_early_turns(tmp_path):
    # t* sampled in [0.5, 0.75] of a 4-turn session with turns at t=0,10,20,30
    # must start at turn 2 or 3 (t >= 15).
    reqs = [_main(i, i * 10.0, blocks=[i], out=4) for i in range(4)]
    path = _corpus(tmp_path, [_trace("k1", reqs)])
    wl = AgentXWorkload(path=path, concurrency=1, seed=3, warmup_requests=0,
                        start_min_ratio=0.5, start_max_ratio=0.75,
                        max_output_tokens=None)
    wl.warmup_only = False
    first = await wl.next_item()
    assert first["meta"]["agentx_main_index"] >= 2


# ---------------------------------------------------------------------------
# Warmup phase (primer + warmup requests, unmeasured)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_warmup_phase_primer_then_warmups_then_stop(tmp_path):
    path = _corpus(tmp_path, [_chain_trace("w1", 8)])
    wl = AgentXWorkload(path=path, concurrency=1, seed=0, warmup_requests=2,
                        start_min_ratio=0.0, start_max_ratio=0.0)
    # warmup_only defaults True.

    primer = await wl.next_item()
    assert primer["max_tokens"] == 1
    assert primer["meta"]["agentx_phase"] == "warmup"
    assert primer["meta"]["agentx_primer"] is True
    wl.note_complete(primer["meta"], True)

    w1 = await wl.next_item()
    assert w1["max_tokens"] == 32
    assert w1["meta"]["agentx_phase"] == "warmup"
    wl.note_complete(w1["meta"], True)
    w2 = await wl.next_item()
    wl.note_complete(w2["meta"], True)

    # All lanes warmed -> the warmup pass ends.
    with pytest.raises(StopAsyncIteration):
        await wl.next_item()

    # The measured pass resumes the DAG from where warmup left off.
    wl.prepare_profile_phase()
    p1 = await wl.next_item()
    assert p1["meta"]["agentx_phase"] == "profile"
    assert p1["meta"]["agentx_req_index"] == 3


@pytest.mark.asyncio
async def test_warmup_counts_span_recycled_plays(tmp_path):
    # 2-turn trace, 3 warmup requests: the lane must recycle to finish its
    # warmup budget, and the budget carries across the play boundary.
    path = _corpus(tmp_path, [_chain_trace("w2", 2)])
    wl = AgentXWorkload(path=path, concurrency=1, seed=0, warmup_requests=3,
                        start_min_ratio=0.0, start_max_ratio=0.0)

    items = []
    for _ in range(3):
        items.append(await wl.next_item())
        wl.note_complete(items[-1]["meta"], True)
    # primer + turn1 (play 0) + turn0 (play 1): still warmup.
    assert items[2]["meta"]["agentx_play"] == 1
    assert items[2]["meta"]["agentx_phase"] == "warmup"
    wl.note_complete(items[2]["meta"], True)
    third = await wl.next_item()
    wl.note_complete(third["meta"], True)
    with pytest.raises(StopAsyncIteration):
        await wl.next_item()
    wl.prepare_profile_phase()
    p = await wl.next_item()
    assert p["meta"]["agentx_phase"] == "profile"


@pytest.mark.asyncio
async def test_root_warmup_failure_aborts(tmp_path):
    path = _corpus(tmp_path, [_chain_trace("f1", 4)])
    wl = AgentXWorkload(path=path, concurrency=1, seed=0, warmup_requests=1,
                        start_min_ratio=0.0, start_max_ratio=0.0)
    primer = await wl.next_item()
    wl.note_complete(primer["meta"], ok=False)
    with pytest.raises(RuntimeError, match="warmup"):
        await wl.next_item()


# ---------------------------------------------------------------------------
# Timing controls
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_recorded_gap_delays_next_turn(tmp_path):
    # Turn 1 recorded 0.3s after turn 0 (api_time 0): after completing turn 0,
    # turn 1 must not dispatch before the gap elapses.
    reqs = [_main(0, 0.0, blocks=[0]), _main(1, 0.3, blocks=[0, 1])]
    path = _corpus(tmp_path, [_trace("g1", reqs), _chain_trace("g2", 1)])
    wl = AgentXWorkload(path=path, concurrency=1, seed=0, warmup_requests=0,
                        start_min_ratio=0.0, start_max_ratio=0.0, shuffle=False)
    wl.warmup_only = False
    a = await wl.next_item()
    wl.note_complete(a["meta"], True)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(wl.next_item(), timeout=0.1)
    b = await asyncio.wait_for(wl.next_item(), timeout=2.0)
    assert b["meta"]["agentx_req_index"] == 1


@pytest.mark.asyncio
async def test_idle_guard_shifts_long_gaps(tmp_path):
    # 300s recorded gap with a 1s idle cap -> the next turn dispatches after
    # ~1s, not ~300s.
    reqs = [_main(0, 0.0, blocks=[0]), _main(1, 300.0, blocks=[0, 1])]
    path = _corpus(tmp_path, [_trace("i1", reqs), _chain_trace("i2", 1)])
    wl = AgentXWorkload(path=path, concurrency=1, seed=0, warmup_requests=0,
                        start_min_ratio=0.0, start_max_ratio=0.0,
                        idle_gap_cap_s=1.0, shuffle=False)
    wl.warmup_only = False
    a = await wl.next_item()
    wl.note_complete(a["meta"], True)
    b = await asyncio.wait_for(wl.next_item(), timeout=5.0)
    assert b["meta"]["agentx_req_index"] == 1


@pytest.mark.asyncio
async def test_subagent_spawn_offset_respected(tmp_path):
    # Child recorded 0.2s after the spawn anchor: it must not dispatch at the
    # moment the anchor completes.
    reqs = [
        _main(0, 0.0, blocks=[0]),
        _subagent(0.2, "a", [_inner(0.2, 64, 8, [5])]),
        _main(1, 0.4, blocks=[0, 1]),
    ]
    path = _corpus(tmp_path, [_trace("o1", reqs), _chain_trace("o2", 1)])
    wl = AgentXWorkload(path=path, concurrency=1, seed=0, warmup_requests=0,
                        start_min_ratio=0.0, start_max_ratio=0.0, shuffle=False)
    wl.warmup_only = False
    root = await wl.next_item()
    wl.note_complete(root["meta"], True)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(wl.next_item(), timeout=0.05)
    child = await asyncio.wait_for(wl.next_item(), timeout=2.0)
    assert child["meta"]["agentx_kind"] == "subagent"


# ---------------------------------------------------------------------------
# Corpus handling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_max_context_filters_traces(tmp_path):
    big = _trace("big", [_main(0, 0.0, in_=4096, blocks=list(range(64)))])
    small = _trace("small", [_main(0, 0.0, in_=128, blocks=[0, 1])])
    path = _corpus(tmp_path, [big, small])
    wl = AgentXWorkload(path=path, concurrency=1, seed=0, warmup_requests=0,
                        max_context=1024, start_min_ratio=0.0,
                        start_max_ratio=0.0)
    wl.warmup_only = False
    item = await wl.next_item()
    assert item["meta"]["agentx_trace"] == "small"


@pytest.mark.asyncio
async def test_gzip_corpus_supported(tmp_path):
    import gzip
    p = tmp_path / "traces.jsonl.gz"
    p.write_text("")
    with gzip.open(p, "wt") as f:
        f.write(_chain_trace("z1", 2) + "\n")
    wl = AgentXWorkload(path=str(p), concurrency=1, seed=0, warmup_requests=0,
                        start_min_ratio=0.0, start_max_ratio=0.0)
    wl.warmup_only = False
    item = await wl.next_item()
    assert item["meta"]["agentx_trace"] == "z1"


@pytest.mark.asyncio
async def test_max_output_tokens_caps_and_ignore_eos_flag(tmp_path):
    reqs = [_main(0, 0.0, in_=64, out=999, blocks=[0])]
    path = _corpus(tmp_path, [_trace("c1", reqs)])
    wl = AgentXWorkload(path=path, concurrency=1, seed=0, warmup_requests=0,
                        max_output_tokens=128, start_min_ratio=0.0,
                        start_max_ratio=0.0)
    wl.warmup_only = False
    primer = await wl.next_item()
    assert primer["max_tokens"] == 1  # primer decodes one token
    wl.note_complete(primer["meta"], True)
    wl.prepare_profile_phase()
    item = await wl.next_item()  # play 1: no primer, capped output
    assert item["meta"]["agentx_play"] == 1
    assert item["max_tokens"] == 128
    assert item["ignore_eos"] is True


def test_missing_or_bad_params(tmp_path):
    with pytest.raises(ValueError):
        AgentXWorkload(path=_corpus(tmp_path, [_chain_trace("x", 1)]),
                       concurrency=0)
    with pytest.raises(ValueError):
        AgentXWorkload(path=_corpus(tmp_path, [_chain_trace("x", 1)]),
                       start_min_ratio=0.8, start_max_ratio=0.2)
    with pytest.raises(ValueError):  # no usable traces after filtering
        AgentXWorkload(path=_corpus(tmp_path, [_chain_trace("x", 1)]),
                       max_context=1)


# ---------------------------------------------------------------------------
# Recipe wiring
# ---------------------------------------------------------------------------

def _shared(**kw):
    from benchmaker.recipes.base import SharedOpts
    defaults = dict(rate="10", duration="10s", max_requests=None, timeout_s=600.0,
                    connection_limit=1000, dotenv="", quiet=True, out_dir=None,
                    run_id=None, labels=(), notes="")
    defaults.update(kw)
    return SharedOpts(**defaults)


def test_recipe_build_records_source_config(tmp_path):
    from benchmaker.recipes.agentx import AgentXRecipe

    path = _corpus(tmp_path, [_chain_trace("r", 2)])
    recipe = AgentXRecipe()
    res = recipe.build(_shared(dotenv=""), url="http://x:8000/v1/chat/completions",
                       model="m", api_key=None, header=(), temperature=0.0,
                       extras=(), trace=path, dataset="unused", file="traces.jsonl",
                       max_context=None, max_output_tokens=None, max_traces=None,
                       concurrency=2, seed=7, start_min_ratio=0.25,
                       start_max_ratio=0.75, warmup_requests=10, cache_bust=True,
                       idle_gap_cap=10.0, ignore_eos=True,
                       default_output_tokens=256, chars_per_token=4.0,
                       tokenizer=None, shuffle=False)
    assert res.workload_type.name == "openai-chat"
    assert res.workload.name.startswith("agentx")
    assert res.source_config["recipe"] == "agentx"
    assert res.source_config["dataset_source"] == f"local:{path}"
    assert res.source_config["workload"]["concurrency"] == 2
    assert res.source_config["workload"]["seed"] == 7
    # AgentX defaults: pull-driven admission + 1h profiling window.
    assert res.default_duration == "1h"
    assert int(res.default_rate) >= 1000
    # The workload schedules a DAG, so the recipe must install its hook.
    assert callable(res.workload.completion_hook())


def test_recipe_registered():
    from benchmaker.recipes import get
    assert get("agentx").name == "agentx"


def test_yaml_build_workload(tmp_path):
    from benchmaker.config import build_workload
    path = _corpus(tmp_path, [_chain_trace("y", 2)])
    wl = build_workload({"type": "agentx", "path": path, "concurrency": 1,
                         "warmup_requests": 0, "start_min_ratio": 0.0,
                         "start_max_ratio": 0.0})
    assert isinstance(wl, AgentXWorkload)


# ---------------------------------------------------------------------------
# End-to-end CLI run against a stub OpenAI endpoint
# ---------------------------------------------------------------------------

def _free_port() -> int:
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_agentx_e2e_cli_two_phase(tmp_path):
    """Full `benchmaker agentx` run: warmup pass excluded, profiling window
    measured, completion hook advancing the DAG — against a stub server."""
    import threading

    from aiohttp import web
    from click.testing import CliRunner

    from benchmaker.cli import main

    seen: list[dict] = []

    async def _sse(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        seen.append({
            "max_tokens": body.get("max_tokens"),
            "prompt_len": len(body["messages"][0]["content"]),
            "meta": body.get("meta") or {},
        })
        resp = web.StreamResponse(
            status=200, headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        for i in range(3):
            chunk = {"choices": [{"index": 0, "delta": {"content": f"t{i} "},
                                  "finish_reason": None}]}
            await resp.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
        await resp.write(b"data: " + json.dumps({
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 3,
                      "total_tokens": 11},
        }).encode() + b"\n\n")
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    port = _free_port()
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def _serve():
        asyncio.set_event_loop(loop)
        app = web.Application()
        app.router.add_post("/v1/chat/completions", _sse)
        runner = web.AppRunner(app)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, "127.0.0.1", port)
        loop.run_until_complete(site.start())
        ready.set()
        loop.run_forever()

    t = threading.Thread(target=_serve, daemon=True)
    t.start()
    ready.wait(timeout=5)

    path = _corpus(tmp_path, [_chain_trace("e1", 6)])
    out_dir = tmp_path / "bundle"
    try:
        res = CliRunner().invoke(main, [
            "agentx", "--url", f"http://127.0.0.1:{port}/v1/chat/completions",
            "--model", "stub", "--trace", str(path),
            "--concurrency", "1", "--warmup-requests", "1",
            "--start-min-ratio", "0", "--start-max-ratio", "0",
            "--duration", "2s", "--dotenv", "", "--quiet",
            "--out-dir", str(out_dir),
        ])
        assert res.exit_code == 0, res.output
        assert "warmup complete — 2 requests (2 ok)" in res.output
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=5)

    # The wire sees the primer and recorded decode lengths...
    assert len(seen) >= 3                      # 2 warmup + >= 1 profile
    assert seen[0]["max_tokens"] == 1          # primer decodes one token
    assert all(r["max_tokens"] == 32 for r in seen[1:])

    # ...and the bundle's samples carry the measured pass only: warmup rows
    # are excluded by design (meta is recorded runner-side, not on the wire).
    import glob
    sample_files = glob.glob(str(out_dir / "**" / "samples.jsonl"), recursive=True)
    assert sample_files, f"no samples.jsonl under {out_dir}"
    phases: list[str] = []
    with open(sample_files[0]) as f:
        for line in f:
            row = json.loads(line)
            meta = row.get("meta") or {}
            if "agentx_phase" in meta:
                phases.append(meta["agentx_phase"])
    assert len(phases) >= 1
    assert set(phases) == {"profile"}


@pytest.mark.asyncio
async def test_live_at_start_skips_children_finished_before_tstar(tmp_path):
    # Group spawns at t=0 with two streams: one finishes at t=5 (before the
    # t* window), one is still running at t=20. Starting at t*=20 (turn at
    # t=20) must replay only the live stream, and the join turn must not wait
    # on the finished one.
    reqs = [
        _main(0, 0.0, blocks=[0]),
        _subagent(0.5, "done_early", [_inner(0.5, 64, 8, [5]), _inner(5.0, 64, 8, [6])]),
        _subagent(0.5, "still_running", [
            _inner(0.5, 64, 8, [7]), _inner(20.0, 64, 8, [8])]),
        _main(1, 30.0, blocks=[0, 1]),
        _main(2, 40.0, blocks=[0, 1, 2]),
    ]
    path = _corpus(tmp_path, [_trace("lv", reqs)])
    wl = AgentXWorkload(path=path, concurrency=1, seed=0, warmup_requests=0,
                        start_min_ratio=0.5, start_max_ratio=0.5,
                        idle_gap_cap_s=1.0, shuffle=False)
    # t* = 0.5 * 40s = 20.0 exactly: first main turn with t >= 20 is turn 1
    # (t=30); still_running resumes at its t=20 request (dispatches first,
    # matching the recording); done_early (last request t=5) is not live.
    wl.warmup_only = False

    # The live subagent stream's remaining request is dispatched first, as a
    # primer (it is a live stream at t*).
    child = await wl.next_item()
    assert child["meta"]["agentx_kind"] == "subagent"
    assert child["meta"]["agentx_agent_id"] == "still_running"
    assert child["max_tokens"] == 1
    assert child["meta"]["agentx_primer"] is True
    wl.note_complete(child["meta"], True)

    # The finished-before-t* stream contributes nothing; the main start turn
    # is primed next.
    primer = await wl.next_item()
    assert primer["meta"]["agentx_kind"] == "main"
    assert primer["meta"]["agentx_main_index"] == 1
    assert primer["max_tokens"] == 1
    wl.note_complete(primer["meta"], True)
    wl.prepare_profile_phase()

    # Join turn: only the live stream gated it, and it is done. (The recorded
    # 10s gap is shifted earlier by the idle guard.)
    join = await asyncio.wait_for(wl.next_item(), timeout=5.0)
    assert join["meta"]["agentx_main_index"] == 2


# ---------------------------------------------------------------------------
# Exact tokenizer mode (trim / pad reconciliation)
# ---------------------------------------------------------------------------

def _install_fake_tokenizer(wl: AgentXWorkload, chars_per_token: int) -> None:
    """Fake tokenizer: every token is exactly `chars_per_token` chars of 'a'."""

    class _FakeTok:
        def __call__(self, text: str, add_special_tokens: bool = False):
            return {"input_ids": list(range(len(text) // chars_per_token))}

        def decode(self, ids, **kw):
            return "a" * (chars_per_token * len(ids))

    def _count(text: str) -> int:
        return max(0, len(text) // chars_per_token)

    wl._count_tokens = _count
    wl._tok = _FakeTok()


@pytest.mark.asyncio
async def test_exact_tokenizer_trims_prompt_to_recorded_tokens(tmp_path):
    # 3 chars/token fake: the chars-mode render (4 chars/token) overshoots,
    # so _token_exact must trim to the recorded count.
    path = _corpus(tmp_path, [_chain_trace("tk", 2)])
    wl = AgentXWorkload(path=path, concurrency=1, seed=0, warmup_requests=0,
                        start_min_ratio=0.0, start_max_ratio=0.0)
    _install_fake_tokenizer(wl, 3)
    wl.warmup_only = False
    item = await wl.next_item()
    content = item["messages"][0]["content"]
    assert wl._count_tokens(content) == 64  # recorded in=64
    assert content.endswith("a" * 3) or len(content) % 3 == 0


@pytest.mark.asyncio
async def test_exact_tokenizer_pads_prompt_to_recorded_tokens(tmp_path):
    # 8 chars/token fake: the chars-mode render undershoots, so _token_exact
    # must pad deterministically up to the recorded count.
    path = _corpus(tmp_path, [_chain_trace("tp", 2)])
    wl = AgentXWorkload(path=path, concurrency=1, seed=0, warmup_requests=0,
                        start_min_ratio=0.0, start_max_ratio=0.0)
    _install_fake_tokenizer(wl, 8)
    wl.warmup_only = False
    a = await wl.next_item()
    wl.note_complete(a["meta"], True)
    wl.prepare_profile_phase()
    b = await wl.next_item()
    for item, recorded in ((a, 64), (b, 128)):
        content = item["messages"][0]["content"]
        assert wl._count_tokens(content) == recorded
    # NOTE: no prefix assertion here — reconciliation (trim/pad) only fires
    # when the recorded token count deviates from the block structure, and in
    # that case the reconciled tail is request-specific by necessity. The
    # published corpus is block-aligned (in = blocks * 64), so the
    # byte-exact prefix property holds on real data (see the chars-mode test
    # above and the block-aligned counts here).
