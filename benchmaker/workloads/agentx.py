"""AgentX workload — replay the SemiAnalysis AgentX v1.0 agentic-coding corpus.

Implements the AgentX replay recipe (https://inferencex.semianalysis.com/agentx)
on top of the published WEKA-format trace corpus
(``semianalysisai/cc-traces-weka-062126[-256k]``): opt-in Claude Code sessions
sanitized into session-scoped chained hashes of 64-token blocks, with recorded
timing and subagent topology but no prompt/code/tool payloads.

Replay semantics (the AgentX methodology):

* **Synthetic payloads.** Every request's prompt is rebuilt deterministically
  from its ``hash_ids``: each block id maps to a fixed block of
  ``block_size`` synthetic coding tokens, so requests that share block ids
  share byte-exact prompt prefixes and the server's prefix cache is exercised
  exactly as in the recording.  The last block is trimmed / deterministic
  filler appended so the rendered prompt matches the recorded input token
  count.
* **Session-tree DAG.** Top-level requests form the main-agent chain;
  ``type: "subagent"`` entries become child chains that spawn after the
  preceding main turn and join before the next dependent main turn (one-off
  trailing subagents run without a join).  Inter-turn delays are the recorded
  end-to-start gaps, replayed from the moment the previous turn completes.
* **Per-session-tree concurrency.** ``concurrency`` lanes each hold one live
  session tree (root + every subagent it spawns).  A lane recycles only once
  its whole tree drains, then replays the next trace from turn 0.
* **Seeded warmup.** A seed picks each lane's start point uniformly in
  25–75% of the recorded session; the start turn (and one request per
  already-live subagent stream) is dispatched as a ``max_tokens=1`` primer;
  each lane then completes ``warmup_requests`` further requests.  Lanes report
  ``phase="warmup"`` meta during this stage; the recipe runs it as a separate
  unmeasured pass and only the following profiling window is reported.
* **Cache-bust markers.** Every play of a trace prepends a unique
  ``[rid:xxxxxxxx]`` tag to every request of that play, so recycled plays and
  concurrent lanes running the same trace cannot accumulate a shared prefix.
* **Idle guard.** If every live tree is waiting on a recorded gap more than
  ``idle_gap_cap`` seconds away, all pending timers are shifted earlier by the
  same amount (recorded order and spacing preserved), bounding true
  system-idle time.

Simplifications vs. the AIPerf harness: flattened top-level fan-outs are not
re-split via hash LCP detection (nested subagent inner requests replay as one
chain in time order), prompts are a single user message rather than a
system/user split, and speculative-decoding acceptance forcing is left to the
server configuration.  Everything else follows the published recipe.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from benchmaker.workloads.datasets import Workload

logger = logging.getLogger(__name__)

# Deterministic coding-session-ish vocabulary for synthetic blocks. The exact
# words don't matter; what matters is that the same block id always renders the
# same bytes (per-trace scope) and different ids render different bytes.
_WORDS = (
    "def", "return", "self", "import", "from", "class", "await", "async",
    "None", "True", "False", "try", "except", "raise", "with", "as", "if",
    "else", "elif", "for", "in", "while", "break", "continue", "yield",
    "lambda", "assert", "global", "pass", "not", "and", "or", "is", "const",
    "let", "var", "function", "export", "default", "typeof", "interface",
    "git", "commit", "push", "pull", "checkout", "branch", "merge", "rebase",
    "test", "spec", "fixture", "mock", "patch", "refactor", "fix", "bug",
    "TODO", "FIXME", "src", "lib", "bin", "usr", "etc", "var", "opt", "tmp",
    "http", "https", "localhost", "port", "server", "client", "request",
    "response", "json", "yaml", "toml", "xml", "html", "css", "node", "npm",
    "pip", "uv", "venv", "docker", "image", "container", "volume", "cache",
    "prefix", "token", "tokens", "model", "prompt", "completion", "chat",
    "stream", "chunk", "buffer", "queue", "task", "worker", "thread", "lock",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "foo", "bar", "baz",
    "qux", "quux", "corge", "grault", "garply", "waldo", "fred", "plugh",
    "xyzzy", "thud",
)


def _stable_seed(*parts: Any) -> int:
    """Platform-stable 64-bit seed from arbitrary parts (never Python hash())."""
    h = hashlib.md5("\x00".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "big")


def _synth_text(scope: str, key: Any, n_chars: int) -> str:
    """Deterministic pseudo coding-session text of exactly ``n_chars`` chars."""
    if n_chars <= 0:
        return ""
    rng = random.Random(_stable_seed(scope, key))
    words = []
    n = 0
    while n < n_chars:
        w = _WORDS[rng.randrange(len(_WORDS))]
        words.append(w)
        n += len(w) + 1
    text = " ".join(words)
    if len(text) > n_chars:
        text = text[:n_chars]
    while len(text) < n_chars:  # degenerate vocabulary guard
        text += "x"
    return text


def cache_bust_marker(seed: int, trace_id: str, play: int) -> str:
    """Per-play marker text prepended to every request of that play.

    Same trace + same play -> same marker for the whole tree; different plays
    (and different lanes wrapping the same trace) -> different markers, so no
    two plays ever accumulate a shared KV prefix.
    """
    tag = hashlib.md5(
        f"agentx-rid\x00{seed}\x00{trace_id}\x00{play}".encode()
    ).hexdigest()[:8]
    return f"[rid:{tag}]\n\n"


# ---------------------------------------------------------------------------
# Trace parsing (WEKA format)
# ---------------------------------------------------------------------------

@dataclass
class _Req:
    t: float                       # recorded start time (s, root-absolute)
    in_tokens: Optional[int]
    out_tokens: Optional[int]
    hash_ids: Any                  # sequence of block ids (list/array)
    api_time: Optional[float]
    model: Optional[str]


@dataclass
class _ChainPlan:
    """One sequential request chain (main agent or one subagent stream)."""
    session_id: str
    kind: str                      # "main" | "subagent"
    agent_id: Optional[str]
    requests: list[_Req] = field(default_factory=list)


@dataclass
class _GroupPlan:
    """Subagent group: spawns after main[spawn_after], joins before main[join_at]."""
    children: list[_ChainPlan]
    spawn_after: int               # absolute main index of the spawning turn
    join_at: Optional[int]         # absolute main index that waits (None=background)


@dataclass
class _Trace:
    trace_id: str
    block_size: int
    main: _ChainPlan
    groups: list[_GroupPlan]
    duration_s: float
    total_requests: int
    peak_context: int              # max(in_tokens + out_tokens) over all requests

    def chains(self) -> list[_ChainPlan]:
        out = [self.main]
        for g in self.groups:
            out.extend(g.children)
        return out


def _inner_request(entry: dict) -> _Req:
    return _Req(
        t=float(entry["t"]),
        in_tokens=entry.get("in"),
        out_tokens=entry.get("out"),
        hash_ids=entry.get("hash_ids") or [],
        api_time=entry.get("api_time"),
        model=entry.get("model"),
    )


def parse_trace(row: dict) -> _Trace:
    """Parse one WEKA trace row into main chain + subagent groups.

    Subagent anchoring follows the AgentX/AIPerf mapping: a subagent entry
    spawns after the preceding main turn and joins before the next dependent
    main turn.  Entries with no preceding main turn are dropped; entries with
    no following main turn become background branches (no join).
    """
    trace_id = str(row.get("id") or row.get("trace_id") or "")
    block_size = int(row.get("block_size") or 64)
    main = _ChainPlan(session_id=trace_id, kind="main", agent_id=None)
    groups: list[_GroupPlan] = []
    pending: list[_GroupPlan] = []   # spawned, awaiting their join anchor
    sub_idx = 0

    for entry in row.get("requests") or []:
        etype = entry.get("type")
        if etype == "subagent":
            if not main.requests:
                continue  # no preceding parent turn -> dropped (AIPerf rule)
            inner = [_inner_request(e) for e in (entry.get("requests") or [])
                     if isinstance(e, dict)]
            if not inner:
                continue
            # Nested timestamp basis: the published corpus stores inner `t`
            # values as root-absolute; producer traces may store them relative
            # to the marker. Detect per trace (same heuristic as AIPerf auto).
            basis = "absolute"
            mt = float(entry.get("t") or 0.0)
            if any(r.t < mt - 1e-6 for r in inner):
                basis = "relative"
            if basis == "relative":
                for r in inner:
                    r.t = mt + r.t
            child = _ChainPlan(
                session_id=f"{trace_id}::sa:{entry.get('agent_id', sub_idx)}",
                kind="subagent",
                agent_id=entry.get("agent_id"),
                requests=sorted(inner, key=lambda r: r.t),
            )
            sub_idx += 1
            g = _GroupPlan(children=[child], spawn_after=len(main.requests) - 1,
                           join_at=None)
            groups.append(g)
            pending.append(g)
        else:
            # "n" (non-streaming), "s" (streaming) — main-agent turns.
            main.requests.append(_inner_request(entry))
            for g in pending:
                g.join_at = len(main.requests) - 1
            pending.clear()

    all_reqs = [r for c in ([main] + [ch for g in groups for ch in g.children])
                for r in c.requests]
    duration = max((r.t for r in all_reqs), default=0.0)
    peak = 0
    for r in all_reqs:
        if r.in_tokens is not None:
            peak = max(peak, int(r.in_tokens) + int(r.out_tokens or 0))
    return _Trace(trace_id=trace_id, block_size=block_size, main=main,
                  groups=groups, duration_s=duration,
                  total_requests=len(all_reqs), peak_context=peak)


# ---------------------------------------------------------------------------
# Runtime lane state (per-play copies of the parsed plans)
# ---------------------------------------------------------------------------

class _Chain:
    """Mutable replay state for one chain during one play."""

    __slots__ = ("plan", "requests", "cursor", "in_flight", "ready_at",
                 "primer_pending", "chain_key")

    def __init__(self, plan: _ChainPlan, requests: list[_Req], chain_key: str,
                 primer: bool):
        self.plan = plan
        self.requests = requests      # sliced for mid-session starts
        self.cursor = 0
        self.in_flight = False
        self.ready_at = 0.0
        self.primer_pending = primer  # first dispatch of this chain is a primer
        self.chain_key = chain_key

    @property
    def has_pending(self) -> bool:
        return self.cursor < len(self.requests)

    @property
    def done(self) -> bool:
        return not self.has_pending and not self.in_flight


class _Group:
    __slots__ = ("plan", "children", "active")

    def __init__(self, plan: _GroupPlan, children: list[_Chain]):
        self.plan = plan
        self.children = children
        self.active = False           # becomes True when the parent turn completes

    @property
    def done(self) -> bool:
        return all(c.done for c in self.children)


class _Lane:
    """One live session tree (root + subagent streams) on one replay slot."""

    __slots__ = ("index", "trace", "play", "marker", "main", "groups",
                 "warm_count", "warm", "warm_needed", "t_star")

    def __init__(self, index: int):
        self.index = index
        self.trace: Optional[_Trace] = None
        self.play = -1
        self.marker = ""
        self.main: Optional[_Chain] = None
        self.groups: list[_Group] = []
        self.warm_count = 0
        self.warm = False
        self.warm_needed = 0
        self.t_star = 0.0

    def chains(self) -> list[_Chain]:
        out = [self.main] if self.main is not None else []
        for g in self.groups:
            out.extend(g.children)
        return out

    @property
    def drained(self) -> bool:
        if self.main is None or not self.main.done:
            return False
        return all(g.active and g.done for g in self.groups)


# ---------------------------------------------------------------------------
# Trace store: byte-offset index + on-demand parse with a small LRU
# ---------------------------------------------------------------------------

class _TraceStore:
    """Random access to a WEKA JSONL corpus without holding it all in memory.

    A one-time pass records each row's file offset plus lightweight scalars
    (id, duration, peak context, request count); rows are parsed on demand and
    kept in a small LRU.  Gzip input is eagerly parsed (no seek support).
    """

    def __init__(self, path: str, *, max_context: Optional[int] = None,
                 max_output_tokens: Optional[int] = None,
                 max_traces: Optional[int] = None,
                 lru_size: int = 16):
        self._path = path
        self._offsets: list[int] = []
        self._meta: list[dict] = []
        self._lru: "OrderedDict[int, _Trace]" = OrderedDict()
        self._lru_size = max(2, lru_size)
        self._index(max_context, max_output_tokens, max_traces)

    # -- indexing -----------------------------------------------------------

    def _index(self, max_context: Optional[int], max_output: Optional[int],
               max_traces: Optional[int]) -> None:
        opener = open
        if self._path.endswith(".gz"):
            import gzip
            opener = gzip.open  # type: ignore[assignment]
        kept_offsets: list[int] = []
        kept_meta: list[dict] = []
        n_seen = 0
        offset = 0
        logger.info("agentx: indexing trace corpus %s ...", self._path)
        with opener(self._path, "rb") as f:
            for line in f:
                line_start = offset
                offset += len(line)
                if not line.strip():
                    continue
                row = json.loads(line)
                meta = self._scan_row(row, max_output)
                n_seen += 1
                if max_context is not None and meta["peak_context"] > max_context:
                    continue
                kept_offsets.append(line_start)
                kept_meta.append(meta)
                if max_traces is not None and len(kept_meta) >= max_traces:
                    break
        if not kept_meta:
            raise ValueError(
                f"agentx: no usable traces in {self._path} "
                f"(seen {n_seen}, max_context={max_context})")
        self._offsets = kept_offsets
        self._meta = kept_meta
        total_reqs = sum(m["total_requests"] for m in kept_meta)
        logger.info(
            "agentx: corpus ready — %d/%d traces, %d requests, "
            "median input %d tokens",
            len(kept_meta), n_seen, total_reqs,
            sorted(m["median_input"] for m in kept_meta)[len(kept_meta) // 2],
        )

    @staticmethod
    def _scan_row(row: dict, max_output: Optional[int]) -> dict:
        """Extract index scalars without retaining hash_ids."""
        ins: list[int] = []
        peak = 0
        total = 0
        dur = 0.0

        def req_stats(r: dict) -> None:
            nonlocal peak, total, dur
            it = r.get("in")
            ot = r.get("out")
            t = r.get("t")
            if isinstance(t, (int, float)):
                dur = max(dur, float(t))
            if isinstance(it, (int, float)) and it > 0:
                ins.append(int(it))
                eff_out = int(ot or 0)
                if max_output is not None:
                    eff_out = min(eff_out, max_output)
                peak = max(peak, int(it) + eff_out)
            total += 1

        def walk(entries: list) -> None:
            for e in entries:
                if not isinstance(e, dict):
                    continue
                if e.get("type") == "subagent":
                    walk(e.get("requests") or [])
                else:
                    req_stats(e)

        walk(row.get("requests") or [])
        ins.sort()
        return {
            "trace_id": str(row.get("id") or row.get("trace_id") or ""),
            "block_size": int(row.get("block_size") or 64),
            "duration_s": dur,
            "total_requests": total,
            "peak_context": peak,
            "median_input": ins[len(ins) // 2] if ins else 0,
        }

    # -- access -------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._meta)

    @property
    def paths_meta(self) -> list[dict]:
        return self._meta

    def get(self, idx: int) -> _Trace:
        tr = self._lru.get(idx)
        if tr is not None:
            self._lru.move_to_end(idx)
            return tr
        tr = self._parse(idx)
        self._lru[idx] = tr
        while len(self._lru) > self._lru_size:
            self._lru.popitem(last=False)
        return tr

    def _parse(self, idx: int) -> _Trace:
        if self._path.endswith(".gz"):
            import gzip
            with gzip.open(self._path, "rb") as f:
                for i, line in enumerate(f):
                    if i == idx:
                        return parse_trace(json.loads(line))
            raise IndexError(idx)
        with open(self._path, "rb") as f:
            f.seek(self._offsets[idx])
            line = f.readline()
        return parse_trace(json.loads(line))


# ---------------------------------------------------------------------------
# The workload
# ---------------------------------------------------------------------------

class AgentXWorkload(Workload):
    """Replay the AgentX agentic-coding corpus (see module docstring).

    Items are OpenAI-chat shaped (``messages`` / ``max_tokens`` /
    ``ignore_eos`` / ``meta``) for use with ``OpenAIChatWorkloadType`` in
    ``passthrough_meta`` mode.  The workload schedules a session-tree DAG per
    lane and needs its :meth:`completion_hook` installed as a post-hook.
    """

    name = "agentx"

    def __init__(
        self,
        *,
        path: str,
        concurrency: int = 8,
        seed: int = 0,
        start_min_ratio: float = 0.25,
        start_max_ratio: float = 0.75,
        warmup_requests: int = 10,
        cache_bust: bool = True,
        idle_gap_cap_s: float = 10.0,
        max_context: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        default_output_tokens: int = 256,
        ignore_eos: bool = True,
        tokenizer: Optional[str] = None,
        chars_per_token: float = 4.0,
        max_traces: Optional[int] = None,
        shuffle: bool = True,
        name: Optional[str] = None,
    ):
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        if not 0.0 <= start_min_ratio <= start_max_ratio <= 1.0:
            raise ValueError("need 0 <= start_min_ratio <= start_max_ratio <= 1")
        if warmup_requests < 0:
            raise ValueError("warmup_requests must be >= 0")
        self.name = name or f"agentx:{path.rsplit('/', 1)[-1]}"
        self._path = path
        self._concurrency = concurrency
        self._seed = seed
        self._start_min = start_min_ratio
        self._start_max = start_max_ratio
        self._warmup_requests = warmup_requests
        self._cache_bust = cache_bust
        self._idle_cap = idle_gap_cap_s
        self._max_output = max_output_tokens
        self._default_out = default_output_tokens
        self._ignore_eos = ignore_eos
        self._chars_per_token = chars_per_token
        self._shuffle = shuffle
        self._count_tokens = self._make_counter(tokenizer)

        self._store = _TraceStore(
            path, max_context=max_context, max_output_tokens=max_output_tokens,
            max_traces=max_traces, lru_size=max(2 * concurrency, 8))
        pool = len(self._store)
        if not cache_bust and concurrency > pool:
            logger.warning(
                "agentx: concurrency %d > corpus size %d with cache-bust off; "
                "capping lanes at the corpus size (wrap would share prefixes)",
                concurrency, pool)
            concurrency = pool
        self._lanes: list[_Lane] = []
        self._n_lanes = concurrency

        # Scheduling state.
        self._pool_rng = random.Random(_stable_seed(seed, "pool"))
        self._order = list(range(pool))
        if self._shuffle:
            self._pool_rng.shuffle(self._order)
        self._cursor = 0
        self._gen: Optional[Iterator[_Trace]] = None  # sequential draw source
        self._lock = asyncio.Lock()
        self._wakeup: Optional[asyncio.Event] = None
        self._abort: Optional[str] = None
        self._rr = 0  # rotating scan cursor for fairness across lanes

        # Phase control: the recipe runs warmup as its own pass, then flips to
        # the measured profiling window.
        self.warmup_only = True

    # -- setup helpers ------------------------------------------------------

    def _next_trace(self) -> _Trace:
        """Draw the next trace from the shuffled pool, wrapping forever."""
        if self._gen is None:
            def _src() -> Iterator[_Trace]:
                while True:
                    idx = self._order[self._cursor % len(self._order)]
                    self._cursor += 1
                    yield self._store.get(idx)
            self._gen = _src()
        return next(self._gen)

    # -- lane lifecycle -----------------------------------------------------

    def _start_lane(self, lane: _Lane, trace: _Trace, play: int, *,
                    initial: bool) -> None:
        """(Re)initialize a lane for one play of ``trace``.

        Initial plays sample a start point t* in the configured ratio window,
        dispatch a max_tokens=1 primer on the start turn (and on each
        already-live subagent stream), and count primer + warmup requests.
        Recycled plays restart at turn 0 with no primer.
        """
        lane.trace = trace
        lane.play = play
        lane.marker = (cache_bust_marker(self._seed, trace.trace_id, play)
                       if self._cache_bust else "")
        # NOTE: warm/warm_count deliberately persist across plays — the
        # primer + warmup budget spans plays until the lane is through it.
        now = time.monotonic()

        if not initial:
            k = 0
            t_star = 0.0
            lane.t_star = 0.0
        else:
            rng = random.Random(_stable_seed(self._seed, "start", lane.index,
                                             trace.trace_id))
            lo = self._start_min * trace.duration_s
            hi = self._start_max * trace.duration_s
            t_star = rng.uniform(lo, hi) if hi > lo else lo
            k = 0
            for i, r in enumerate(trace.main.requests):
                if r.t >= t_star:
                    k = i
                    break
            else:
                k = max(0, len(trace.main.requests) - 1)
            # Always leave enough tree remaining for primer(s) + warmup + at
            # least one profiling request; move the start earlier if needed.
            while (self._suffix_size(trace, k)
                   < self._n_primers_at(trace, k) + self._warmup_requests + 1
                   and k > 0):
                k -= 1
            lane.t_star = t_star

        lane.main = _Chain(
            trace.main, trace.main.requests[k:],
            chain_key=f"{trace.trace_id}::main", primer=initial)
        lane.groups = []
        n_primers = 1 if initial else 0
        for g in trace.groups:
            if not initial:
                children = [_Chain(c, list(c.requests), chain_key=c.session_id,
                                   primer=False) for c in g.children]
                lane.groups.append(_Group(g, children))
                continue
            # Initial play: anchor turns are relative to the sliced main chain.
            spawn_rel = g.spawn_after - k
            join_rel = (g.join_at - k) if g.join_at is not None else None
            if join_rel is not None and join_rel < 0:
                continue  # whole group already in the past
            if spawn_rel < 0:
                # Live at t*: resume each child at its first request >= t*.
                # A stream that already completed before t* is not live and
                # must not be replayed.
                children = []
                for c in g.children:
                    off = len(c.requests)  # default: finished before t*
                    for i, r in enumerate(c.requests):
                        if r.t >= t_star:
                            off = i
                            break
                    if off >= len(c.requests):
                        continue
                    children.append(_Chain(c, c.requests[off:],
                                           chain_key=c.session_id, primer=True))
                    n_primers += 1
                if not children:
                    continue  # whole group finished before t*
                grp = _Group(g, children)
                grp.active = True
                for c in children:
                    c.ready_at = now
                lane.groups.append(grp)
            else:
                children = [_Chain(c, list(c.requests), chain_key=c.session_id,
                                   primer=False) for c in g.children]
                lane.groups.append(_Group(g, children))
        lane.main.ready_at = now
        if initial:
            # Fixed once per lane lifetime: warm_count persists across plays,
            # so warm_needed must not be recomputed on recycle.
            lane.warm_needed = n_primers + self._warmup_requests

    @staticmethod
    def _suffix_size(trace: _Trace, k: int) -> int:
        """Requests remaining in the tree when the main chain starts at k."""
        n = len(trace.main.requests) - k
        for g in trace.groups:
            keep = (g.join_at is None or g.join_at >= k)
            if keep:
                n += sum(len(c.requests) for c in g.children)
        return n

    @staticmethod
    def _n_primers_at(trace: _Trace, k: int) -> int:
        """Primer streams dispatched when starting at main index k (upper bound)."""
        n = 1
        for g in trace.groups:
            if g.join_at is not None and g.join_at < k:
                continue
            if g.spawn_after < k:
                n += len(g.children)
        return n

    # -- prompt synthesis ---------------------------------------------------

    def _render_prompt(self, lane: _Lane, req: _Req, chain: _Chain) -> str:
        trace = lane.trace
        assert trace is not None
        block_size = trace.block_size
        cpt = self._chars_per_token
        target_tokens = req.in_tokens
        if target_tokens is None:
            target_tokens = len(req.hash_ids) * block_size
        target_tokens = max(1, int(target_tokens))

        marker = lane.marker
        marker_chars = len(marker)
        # The marker eats into the recorded token budget so the wire prompt
        # still lands at ~target_tokens (AIPerf compensates the same way).
        budget = int(target_tokens * cpt)
        content_budget = max(1, budget - marker_chars)

        parts = [marker]
        filled = 0
        for i, hid in enumerate(req.hash_ids):
            if filled >= content_budget:
                break
            take = min(block_size * int(cpt), content_budget - filled)
            parts.append(_synth_text(trace.trace_id, hid, take))
            filled += take
        if filled < content_budget:
            # Recorded input exceeds the block structure (chat-template /
            # padding overhead): deterministic filler, unique per request.
            req_key = f"{chain.chain_key}#{chain.cursor}"
            parts.append(_synth_text(trace.trace_id, req_key,
                                     content_budget - filled))
            filled = content_budget
        text = "".join(parts)
        if self._count_tokens is not None:
            text = self._token_exact(text, target_tokens, trace.trace_id,
                                     chain.chain_key, chain.cursor, marker)
        return text

    def _token_exact(self, text: str, target: int, scope: str,
                     chain_key: str, cursor: int, marker: str) -> str:
        """Resize ``text`` to exactly ``target`` tokens (marker kept)."""
        ids = self._count_tokens(text)  # type: ignore[misc]
        if ids == target:
            return text
        if ids > target:
            trimmed = self._decode_ids(self._encode_ids(text)[:target])
            return trimmed if len(trimmed) >= len(marker) else text
        # Pad with deterministic filler until the token count is reached.
        rng = random.Random(_stable_seed(scope, chain_key, cursor, "pad"))
        out = text
        while ids < target:
            chunk = _synth_text(scope, (chain_key, cursor, ids), 512)
            out += chunk
            new_ids = self._encode_ids(out)
            if len(new_ids) > target:
                out = self._decode_ids(new_ids[:target])
                ids = target
                break
            ids = len(new_ids)
        return out

    def _encode_ids(self, text: str) -> list[int]:
        assert self._tok is not None
        return self._tok(text, add_special_tokens=False)["input_ids"]

    def _decode_ids(self, ids: list[int]) -> str:
        assert self._tok is not None
        return self._tok.decode(ids, skip_special_tokens=True)

    def _make_counter(self, tokenizer: Optional[str]):
        if not tokenizer:
            return None
        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "agentx replay with --tokenizer needs `transformers`. Install "
                "with `pip install transformers` or `pip install -e .[tokenizer]`."
            ) from e
        tok = AutoTokenizer.from_pretrained(tokenizer)
        self._tok = tok

        def _count(text: str) -> int:
            return len(self._encode_ids(text))

        return _count

    # -- scheduling ---------------------------------------------------------

    def _admit(self) -> None:
        while len(self._lanes) < self._n_lanes:
            lane = _Lane(index=len(self._lanes))
            trace = self._next_trace()
            self._start_lane(lane, trace, play=0, initial=True)
            self._lanes.append(lane)

    def _reap(self) -> None:
        for lane in self._lanes:
            if not lane.drained:
                continue
            trace = self._next_trace()
            self._start_lane(lane, trace, play=lane.play + 1, initial=False)
            logger.debug("agentx: lane %d recycled to trace %s (play %d)",
                         lane.index, trace.trace_id, lane.play)

    @staticmethod
    def _gap_seconds(prev: _Req, nxt: _Req) -> float:
        """Recorded end-to-start gap between two consecutive chain requests."""
        if prev.api_time is not None:
            d = nxt.t - (prev.t + prev.api_time)
        else:
            d = nxt.t - prev.t
        return d if d > 0.0 else 0.0

    def _eligible(self, now: float):
        """Return (lane, chain, req) for the next dispatchable request."""
        n = len(self._lanes)
        start = self._rr % n if n else 0
        for off in range(n):
            lane = self._lanes[(start + off) % n]
            # Spawned subagent children first (they gate the main chain).
            for g in lane.groups:
                if not g.active:
                    continue
                for c in g.children:
                    if c.has_pending and not c.in_flight and c.ready_at <= now:
                        self._rr = lane.index + 1
                        return lane, c, c.requests[c.cursor]
            m = lane.main
            assert m is not None
            if m.has_pending and not m.in_flight and m.ready_at <= now:
                join_idx = m.cursor  # lane-relative main index to dispatch
                if all(g.done for g in lane.groups
                       if g.plan.join_at is not None
                       and g.plan.join_at - self._k_of(lane) == join_idx):
                    self._rr = lane.index + 1
                    return lane, m, m.requests[m.cursor]
        return None

    def _k_of(self, lane: _Lane) -> int:
        """Absolute main index of the lane's first dispatched turn this play."""
        m = lane.main
        assert lane.trace is not None and m is not None
        return len(lane.trace.main.requests) - len(m.requests)

    def _idle_shift(self, now: float) -> Optional[float]:
        """Bound system-idle time: jump all pending timers when the soonest
        recorded gap exceeds the cap. Returns the seconds skipped."""
        if self._idle_cap <= 0:
            return None
        future = []
        for lane in self._lanes:
            for c in lane.chains():
                if c.has_pending and not c.in_flight and c.ready_at > now:
                    future.append(c.ready_at)
        if not future:
            return None
        wait = min(future) - now
        if wait <= self._idle_cap:
            return None
        shift = wait - self._idle_cap
        for lane in self._lanes:
            for c in lane.chains():
                if c.ready_at > now:
                    c.ready_at -= shift
        return shift

    async def next_item(self) -> Any:
        if self._wakeup is None:
            self._wakeup = asyncio.Event()
        self._admit()
        while True:
            if self._abort is not None:
                raise RuntimeError(self._abort)
            # End of the warmup pass: every lane is through primer + warmup
            # and nothing is in flight. Checked BEFORE reaping so a drained
            # lane does not recycle into a fresh play inside the warmup pass.
            if self.warmup_only and self._all_warm_and_idle():
                raise StopAsyncIteration
            self._reap()
            now = time.monotonic()
            got = self._eligible(now)
            if got is not None:
                lane, chain, req = got
                return self._emit(lane, chain, req, now)
            # Nothing dispatchable right now.
            shift = self._idle_shift(now)
            now = time.monotonic()
            soonest = self._soonest_ready(now)
            timeout = None if soonest is None else max(0.0, soonest - now)
            if shift is not None:
                timeout = min(timeout, self._idle_cap) if timeout is not None \
                    else self._idle_cap
            self._wakeup.clear()
            try:
                await asyncio.wait_for(self._wakeup.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                pass  # a recorded gap elapsed; re-evaluate

    def _soonest_ready(self, now: float) -> Optional[float]:
        waits = []
        for lane in self._lanes:
            for c in lane.chains():
                if c.has_pending and not c.in_flight and c.ready_at > now:
                    waits.append(c.ready_at)
        return min(waits) if waits else None

    def _all_warm_and_idle(self) -> bool:
        if not self._lanes or len(self._lanes) < self._n_lanes:
            return False
        for lane in self._lanes:
            if not lane.warm:
                return False
            for c in lane.chains():
                if c.in_flight:
                    return False
        return True

    def _emit(self, lane: _Lane, chain: _Chain, req: _Req, now: float) -> dict:
        phase = "profile" if lane.warm else "warmup"
        primer = chain.primer_pending and chain.cursor == 0
        chain.primer_pending = False
        prompt = self._render_prompt(lane, req, chain)
        if chain.plan.kind == "main":
            main_index = self._k_of(lane) + chain.cursor
        else:
            main_index = None
        out_tokens = req.out_tokens
        if out_tokens is None:
            out_tokens = self._default_out
        if self._max_output is not None:
            out_tokens = min(int(out_tokens), self._max_output)
        max_tokens = 1 if primer else int(out_tokens)
        meta = {
            "agentx_trace": lane.trace.trace_id,
            "agentx_play": lane.play,
            "agentx_lane": lane.index,
            "agentx_chain": chain.chain_key,
            "agentx_kind": chain.plan.kind,
            "agentx_agent_id": chain.plan.agent_id,
            "agentx_req_index": chain.cursor,
            "agentx_main_index": main_index,
            "agentx_phase": phase,
            "agentx_primer": primer,
            "agentx_hash_blocks": len(req.hash_ids),
            "agentx_recorded_input_tokens": req.in_tokens,
            "agentx_recorded_output_tokens": req.out_tokens,
            "conversation_id": f"{lane.trace.trace_id}#p{lane.play}",
        }
        chain.cursor += 1
        chain.in_flight = True
        return {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "ignore_eos": bool(self._ignore_eos),
            "meta": meta,
        }

    # -- completion signalling ----------------------------------------------

    def note_complete(self, meta: dict, ok: bool) -> None:
        """Advance the lane whose request just completed (from the post-hook)."""
        lane_idx = meta.get("agentx_lane")
        if lane_idx is None or lane_idx >= len(self._lanes):
            return
        lane = self._lanes[lane_idx]
        chain_key = meta.get("agentx_chain")
        req_index = meta.get("agentx_req_index")
        chain = None
        for c in lane.chains():
            if c.chain_key == chain_key:
                chain = c
                break
        if chain is None or not chain.in_flight or chain.cursor - 1 != req_index:
            return
        chain.in_flight = False
        now = time.monotonic()
        trace = lane.trace
        assert trace is not None

        if chain.plan.kind == "main":
            # Spawn groups anchored on the turn that just completed.
            abs_idx = self._k_of(lane) + (chain.cursor - 1)
            for g in lane.groups:
                if not g.active and g.plan.spawn_after == abs_idx:
                    g.active = True
                    anchor = trace.main.requests[abs_idx]
                    # Recorded offset of the child's first request from the
                    # spawn anchor's recorded END (end-to-start, the same
                    # convention as main-turn gaps), clamped at 0.
                    anchor_end = anchor.t + (anchor.api_time or 0.0)
                    for c in g.children:
                        offset = max(0.0, c.requests[0].t - anchor_end)
                        c.ready_at = now + offset
            # Gap to the next main turn.
            if chain.has_pending:
                prev = chain.requests[chain.cursor - 1]
                nxt = chain.requests[chain.cursor]
                chain.ready_at = now + self._gap_seconds(prev, nxt)
        else:
            if chain.has_pending:
                prev = chain.requests[chain.cursor - 1]
                nxt = chain.requests[chain.cursor]
                chain.ready_at = now + self._gap_seconds(prev, nxt)

        # Warmup accounting: every completion while the lane is un-warm counts
        # toward primer(s) + warmup_requests.
        if not lane.warm:
            lane.warm_count += 1
            if lane.warm_count >= lane.warm_needed:
                lane.warm = True
                logger.debug("agentx: lane %d warmed (%d requests)",
                             lane.index, lane.warm_count)

        # Root-conversation warmup failures abort the run (one attempt, no
        # retry) — profiling must not start on a degraded tree pool.
        if (not ok and meta.get("agentx_phase") == "warmup"
                and meta.get("agentx_kind") == "main"):
            self._abort = (
                f"agentx: root warmup request failed for trace "
                f"{meta.get('agentx_trace')} (play {meta.get('agentx_play')}); "
                "aborting before the profiling window")

        if self._wakeup is not None:
            self._wakeup.set()

    def completion_hook(self) -> Optional[Callable]:
        def _hook(request: Any, response: Any, sample: Any) -> Any:
            meta = getattr(request, "meta", None) or {}
            if "agentx_lane" in meta:
                self.note_complete(meta, bool(getattr(sample, "ok", False)))
            return sample
        return _hook

    # -- phase control --------------------------------------------------------

    def prepare_profile_phase(self) -> None:
        """Flip from the warmup pass to the measured profiling window."""
        self.warmup_only = False
        self._wakeup = asyncio.Event()  # fresh, cleared

    async def aclose(self) -> None:
        self._gen = None
