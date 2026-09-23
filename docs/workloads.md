# Workloads & workload-types

Two layered abstractions:

- **WorkloadType** (the protocol) — knows how to talk to a kind of service.
- **Workload** (the dataset) — yields per-request items that the workload-type
  interprets.

Keeping them separate means you can swap the dataset under a workload-type
without changing the protocol, and vice versa. The same `JsonlWorkload` of
prompts feeds `OpenAIChatWorkloadType` and a custom `CompletionsWorkloadType`.

---

## Built-in workload-types

### `HttpWorkloadType`

Generic HTTP. The workload-type holds the static parts (URL, method, default
headers); items from the workload fill in the per-request payload.

```python
HttpWorkloadType(
    url="https://api.example.com/v1/predict",
    method="POST",
    headers={"User-Agent": "benchmaker/0.3"},
    timeout_s=30,
)
```

**Item interpretation:**

| Item type        | Behavior                                                   |
| ---------------- | ---------------------------------------------------------- |
| `None`           | Fire the base request unchanged                            |
| `bytes` / `str`  | Use as raw body                                            |
| `dict` (plain)   | Use as JSON body                                           |
| `dict` w/ Request keys | Apply as Request overrides — see below              |

The "Request keys" that trigger override mode are: `body`, `json`, `params`,
`headers`, `url`, `method`, `meta`, `timeout_s`. When the dict contains *any*
of these, the dict is treated as a partial `Request` rather than a JSON body.

```python
# These two yield identical requests:
StaticWorkload(items=[{"a": 1}])                          # JSON body = {"a": 1}
StaticWorkload(items=[{"json": {"a": 1}}])                # explicit
StaticWorkload(items=[{"json": {"a": 1}, "headers": {"X-Tag": "exp"}}])  # override
```

### `OpenAIChatWorkloadType`

OpenAI-compatible `/v1/chat/completions` with SSE streaming. Works with vLLM,
SGLang, TGI, OpenAI itself, and any other server that speaks the same wire
format.

```python
OpenAIChatWorkloadType(
    url="http://localhost:8000/v1/chat/completions",
    model="meta-llama/Llama-3.1-8B-Instruct",
    max_tokens=128,
    temperature=0.0,
    api_key=os.environ.get("OPENAI_API_KEY"),
)
```

**Sampling params:** `max_tokens` and `temperature` are explicit kwargs;
anything else is forwarded into the request body via `**sampling`. This is
how vLLM/SGLang extensions get plumbed through without needing the library
to know about them:

```python
OpenAIChatWorkloadType(
    url=..., model=...,
    max_tokens=256,
    min_tokens=64,            # vLLM/SGLang
    ignore_eos=True,          # vLLM/SGLang
    top_p=0.9,
    stop=["\n\n"],
    repetition_penalty=1.1,   # forwarded as-is
    guided_json={"type": "object", ...},  # forwarded as-is
)
```

`extra_body=` is still accepted as an explicit dict; on key conflict,
`**sampling` wins. The same pass-through works in YAML (`min_tokens: 64` under
`workload_type:`) and on the `benchmaker llm` CLI (`--min-tokens 64`,
`--extra repetition_penalty=1.1`).

**Item interpretation:**

| Item type        | Behavior                                                       |
| ---------------- | -------------------------------------------------------------- |
| `str`            | Wrapped as `[{"role": "user", "content": item}]`               |
| `list[dict]`     | Used as `messages` directly                                    |
| `dict`           | Merged into the body (may override `model`, `max_tokens`, …)   |

A dict item with a `prompt` key (and no `messages`) gets the `prompt` promoted
to a user message:

```python
StaticWorkload(items=[
    {"prompt": "hi", "max_tokens": 16},
    {"prompt": "explain RDMA", "max_tokens": 256, "temperature": 0.7},
])
```

**Extra metrics captured per request** (in `Sample.extra`, then aggregated):

- `ttft_s`        — time to first token (see `ttft_token`)
- `content_ttft_s` — time to first *visible* (content) token, surfaced only
                     when reasoning preceded it (i.e. it differs from `ttft_s`)
- `itl_ms_mean / itl_ms_p50 / itl_ms_p99` — inter-token latency across the
                     whole generation, counting reasoning tokens the same as
                     content tokens
- `tokens_out`    — completion tokens (from `usage` if the server includes it, else counted from chunks)
- `reasoning_tokens` — from `usage.completion_tokens_details` when present
- `content_tokens`   — `completion_tokens - reasoning_tokens` when both known
- `prompt_tokens` — when present in `usage`
- `tokens_per_s`  — `tokens_out / (latency - ttft)`

`ttft_token` (`"any"` | `"content"`, default `"any"`) selects which token the
headline `ttft_s` is measured to: the first token of any kind (reasoning *or*
content — the engine-cost signal a serving benchmark wants) or the first
visible content token (the latency a user perceives). For thinking models
(GLM-4.x, DeepSeek-R1, …) the reasoning chain streams in `reasoning_content`
before `content`; counting those tokens keeps `ttft_s`, `itl_ms_*`, and
`tokens_out` honest instead of silently ignoring the whole reasoning phase.

A 200-status response with zero tokens is marked `ok=False` (error: `"no
tokens received"`).

**`prompt_tokens`** from the OpenAI `usage` block is captured when present;
**`cached_tokens`** is also captured when the server returns it (vLLM's prefix
cache hit count).

### `DeepRAGWorkload`

`DeepRAGWorkload` reads prepared multi-passage QA JSONL and builds a short
answer request with a deliberately long retrieved context. It emits OpenAI
chat `messages`, a `reference`, `rag_depth`, and `prompt_tokens_hint`.

```python
from benchmaker import DeepRAGWorkload

workload = DeepRAGWorkload(
    path=".local/hotpotqa_distractor.jsonl",
    depth=10,
    context_tokens_target=12000,
    max_tokens=64,
)
```

Prepare the default HotpotQA distractor corpus with
`python tools/rag/prepare.py`. Pair the workload with
`OpenAIChatWorkloadType(passthrough_meta=True)` so metadata is recorded rather
than sent to the endpoint, and optionally wrap it in `EvalWorkloadType` for
short-answer scoring. See [DeepRAG and mixed lanes](deeprag-mix.md) for a full
phase-swing configuration.

### `SGLangGenerateWorkloadType`

SGLang's native `/generate` endpoint. Unlike the OpenAI path, the body is
`{"text": ..., "sampling_params": {...}, "stream": true}` and streamed events
carry cumulative `text` plus a `meta_info` object. Prefer the OpenAI path
where possible; use this for raw-text parity checks.

```python
SGLangGenerateWorkloadType(
    url="http://host:30000/generate",
    max_tokens=128,
    temperature=0.0,
    extra_body=None,
    headers=None,
    timeout_s=600.0,
    passthrough_meta=False,
    **sampling,                 # top_p, top_k, min_p, stop, ...
)
```

**`from_env()` classmethod** — reads `SGLANG_API_BASE_URL` or `SGLANG_BASE_URL`
from the environment (loaded from `.env`), appending `/generate`:

```python
SGLangGenerateWorkloadType.from_env(url=None, dotenv_path=".env")
```

**Item interpretation:**

| Item type | Behavior                                                       |
| --------- | -------------------------------------------------------------- |
| `str`     | `{"text": item}`                                               |
| `dict`    | `text`/`input_ids` pulled out; remaining sampling keys merged into `sampling_params`. With `passthrough_meta`, non-sampling keys are recorded into `Request.meta` instead of sent. |

**Sampling keys** (recognized in dict items and forwarded into `sampling_params`):
`temperature`, `max_new_tokens`, `top_p`, `top_k`, `min_p`, `stop`,
`stop_token_ids`, `frequency_penalty`, `presence_penalty`,
`repetition_penalty`, `ignore_eos`, `skip_special_tokens`, `n`, `seed`,
`min_new_tokens`, `regex`, `json_schema`, `ebnf`.

**Extra metrics captured per request** (in `Sample.extra`):

- `ttft_s`                  — time to first token
- `itl_ms_mean / itl_ms_p50 / itl_ms_p99` — inter-token latency
- `tokens_out`              — completion tokens
- `prompt_tokens`           — from `meta_info`
- `cached_tokens`           — from `meta_info`
- `tokens_per_s`            — `tokens_out / (latency - ttft)`
- `finish_reason`           — in `Sample.meta`

YAML: `workload_type: {type: sglang-generate, url: ...}`.

### `SandboxWorkloadType`

Drives the [Flash Sandbox](https://github.com/swiss-ai/flash-sandbox) HTTP API.
Operation modes map to benchmark tickets so they compose with all the existing
load models / monitors.

```python
SandboxWorkloadType(
    base_url="http://localhost:8080",
    operation="exec",                 # "exec" (default), "create", "lifecycle", or "file"
    spec={                            # body for the lazy POST /sandboxes
        "type": "kubernetes",
        "image": "alpine:3.20",
        "command": ["sh", "-c", "sleep 3600"],
        "memory_mb": 256,
        "cpu_cores": 0.5,
    },
    ttl_seconds=600,                  # orchestrator-side reap safety net
    endpoint_prefix="/sandboxes",     # or "/native/sandboxes" for a single node
    sandbox_id=None,                  # if set, skips create/delete entirely
    persistent=False,                 # True → use /pshell instead of /exec
    default_command=None,             # used when items are None
    file_path="/tmp/benchmaker.bin",  # file mode write/read path
    file_content=b"benchmaker",       # default bytes for file mode
    file_verify_with_exec=False,      # optional extra exec verifier (cat)
    file_verify_command=None,         # override verifier command
    cleanup_on_close=True,
)
```

**`exec` mode** (default). On the first request, a sandbox is created (under
an internal lock, so concurrent workers don't race). Every subsequent request
fires `POST /sandboxes/{sid}/exec` against that one sandbox. `aclose()` (run
in the `finally` block of `BenchRunner.run`) deletes it.

**`create` mode**. Each request is a `POST /sandboxes`. Use this to benchmark
sandbox-startup latency. Set `ttl_seconds` so the orchestrator reaps the
sandboxes — the create mode never deletes anything itself.

**`lifecycle` mode**. Each ticket runs the full
`POST /sandboxes` → `POST /sandboxes/{id}/exec` → `DELETE /sandboxes/{id}`
sequence on its own throwaway sandbox. The three legs are timed individually
(`create_s` / `exec_s` / `delete_s` in `Sample.extra`) and the sample's
`latency_s` covers all three. The delete is best-effort: if it fails the
sample still counts as a success (just with `delete_error` in `meta`) so that
exec failures and teardown flakes don't get confused. `sandbox_id=` is
forbidden in lifecycle mode (the workload-type always allocates its own).
Items are interpreted the same way as in `exec` mode.

**`file` mode**. Each ticket writes bytes with `PUT /sandboxes/{id}/files`
(`path` query param), reads them back with `GET .../files`, and compares bytes
exactly (binary-safe). Optional exec verification can run an extra command
(default `cat <path>`) and compare that output too.

**Item interpretation (`exec` mode):**

| Item type        | Behavior                                                       |
| ---------------- | -------------------------------------------------------------- |
| `None`           | Run the configured `default_command`                           |
| `str`            | `["sh", "-c", item]` (a shell expression)                      |
| `list[str]`      | Used as argv directly                                          |
| `dict`           | `{"command": …, "env": {...}, "input": "...", "persistent": bool}` |

A dict item with `persistent: True` switches that single request to `/pshell`
(persistent shell, preserves `cd`/`export`/`source`). Set `persistent=True`
on the workload-type to use `/pshell` for *every* request.

**Item interpretation (`create` mode):**

| Item type        | Behavior                                                       |
| ---------------- | -------------------------------------------------------------- |
| `None`           | Use the configured `spec` verbatim                             |
| `dict`           | Merged into the spec (e.g. to vary `image` or `memory_mb`)    |

**Item interpretation (`file` mode):**

| Item type        | Behavior                                                       |
| ---------------- | -------------------------------------------------------------- |
| `None`           | Write configured `file_content` to `file_path`                |
| `bytes` / `str`  | Write item bytes (UTF-8 for `str`) to `file_path`             |
| `dict`           | `{"path", "content" or "content_base64", "verify_exec", "verify_command"}` |

**Extra metrics captured per request** (in `Sample.extra`):

- `exit_code`           — exit code; a non-zero value flips `ok=False`
- `server_duration_s`   — server-reported execution duration
- `stdout_bytes`        — length of stdout in the response body
- `stderr_bytes`        — length of stderr in the response body
- `server_created`      — `1.0` per successful create (`create` mode only)
- `create_s` / `exec_s` / `delete_s` / `lifecycle_s` (`lifecycle` mode only)
- `file_write_s` / `file_read_s` / `file_write_bytes` / `file_read_bytes` (`file` mode only)
- `file_mismatch_count` / `file_mismatch_bytes` (`file` mode only)
- `file_exec_s` / `file_exec_mismatch_count` / `file_exec_mismatch_bytes` (`file` + exec verify)

In `create` and `lifecycle` modes, the response's `id` field is stored in
`Sample.meta["sandbox_id"]`. In `lifecycle` mode, a non-blocking delete failure
surfaces as `Sample.meta["delete_error"]`.

YAML:

```yaml
workload_type:
    type: sandbox
    base_url: http://localhost:8080
    spec:
        type: kubernetes
        image: alpine:3.20
        memory_mb: 256
        cpu_cores: 0.5
    ttl_seconds: 600

workload:
    type: static
    items:
        - "echo hello"
        - "uname -a"

load: poisson:20
duration: 30s
```

### `AgentWorkloadType`

Run a user-provided Python `Agent` per ticket. Unlike LLM workload-types, no
request shape is dictated — the agent owns its own client(s) and one "request"
equals one full agent run (which may be many internal HTTP calls).

**The Agent ABC:**

```python
class Agent(ABC):
    @abstractmethod
    async def run(self, ctx: AgentContext) -> AgentResult: ...
    async def aclose(self) -> None: ...        # default no-op
```

**`AgentContext`** — per-task state handed to `Agent.run`:

| Field          | Type                | Meaning                                      |
| -------------- | ------------------- | -------------------------------------------- |
| `item`         | `Any`               | The workload item (typically a dict with `prompt`) |
| `workload_name`| `str`               | Workload-type name                           |
| `fire`         | `FireRequest | None`| Runner's request-firing callable (optional)  |
| `start_mono`   | `float`             | Monotonic start time                         |

**`AgentResult`** — what `Agent.run` returns:

| Field          | Type                | Meaning                                      |
| -------------- | ------------------- | -------------------------------------------- |
| `output`       | `str`               | Text the grader sees                         |
| `ok`           | `bool`              | Whether the agent considers this a success   |
| `error`        | `str | None`        | Error message                                |
| `request_ok`   | `bool`              | Whether the agent ran without infra failure (default `True`) |
| `metrics`      | `dict[str, float]`  | Lands in `Sample.extra`                      |
| `meta`         | `dict[str, Any]`    | Lands in `Sample.meta`                       |
| `bytes_sent`   | `int`               | Bytes sent by the agent                      |
| `bytes_recv`   | `int`               | Bytes received by the agent                  |

When `request_ok=False` the sample is bucketed as "fail" (transport error);
when `request_ok=True` but `ok=False` it's bucketed as "wrong" (delivered
but graded wrong).

**Constructor:**

```python
AgentWorkloadType(
    agent,                          # Agent instance, Agent subclass, or callable
    agent_kwargs=None,              # forwarded to class constructor when agent is a class
    reference_key="reference",      # item key carrying the gold reference
    extra_meta_keys=(),             # additional item keys to copy into Sample.meta
    name="agent",
)
```

`handles_reference = True` — the workload-type splits `reference` out of items
itself, so `correctness_hook` is installed directly without `EvalWorkloadType`
wrapping.

**CallableAgent adapter** — a plain callable `(AgentContext) -> AgentResult|dict|str`
is automatically wrapped in `CallableAgent`:

```python
AgentWorkloadType(agent=my_function, agent_kwargs={"model": "gpt-4o"})
```

YAML:

```yaml
workload_type:
  type: agent
  agent: 'mypkg.myagent:MyAgent'
  agent_kwargs:
    model: 'gpt-4o-mini'
  reference_key: reference
  extra_meta_keys: [task_id]
```

### `AgenticWorkload`

Expands multi-turn agent trajectories into one chat request per assistant
turn — each request sends the growing message prefix up to (but excluding)
that turn. Built for prefix-cache / HiCache parity benchmarking against an
OpenAI-compatible chat endpoint.

```python
AgenticWorkload(
    dataset=None,                   # HuggingFace dataset id (needs `datasets`)
    path=None,                      # or local JSONL file
    split="tool",                   # dataset split
    messages_field="messages",      # field name for messages in each row
    id_field="instance_id",         # field name for instance id
    model_field="model",            # field name for model label
    max_tokens=1024,                # per-request generation cap
    max_turns_per_trajectory=None,  # cap assistant turns per trajectory
    max_trajectories=None,          # cap number of trajectories replayed
    tokenizer=None,                 # HF tokenizer id; enables expected_prefix_tokens
)
```

Provide exactly one of `dataset` or `path`.

**How it works:** Each row's `messages` field is parsed and sanitized
(`sanitize_message` keeps only OpenAI-valid keys: `role`, `content`, `name`,
`tool_calls`, `tool_call_id`). For every assistant turn, a workload item is
emitted containing the prefix messages *before* that turn. The dataset
exhausts naturally (raises `StopAsyncIteration`), ending the run.

**`expected_prefix_tokens`** — when a `tokenizer` is configured, each item's
`meta` carries the theoretical upper bound of cacheable prefix (the previous
turn's prompt token count). Compare to the server's `cached_tokens` to compute
prefix-cache hit efficiency.

**`parse_messages`** handles both a JSON-encoded string (the format SWE-smith
ships) and a plain list.

## Built-in workloads (datasets)

### `StaticWorkload`

Cycle a fixed list. Good for trivial benchmarks and quick experiments.

```python
StaticWorkload(items=[...], shuffle=True, seed=0, max_items=10_000)
```

- `shuffle` permutes the items once at construction
- `max_items` raises `StopAsyncIteration` after that many items (cuts the run short)

### `JsonlWorkload`

Stream items from a JSONL file. Items are decoded JSON objects.

```python
JsonlWorkload(path="data/sharegpt.jsonl",
              field="prompt",   # optional: yield only this field per line
              loop=True,        # restart at EOF
              max_items=None)
```

The file is read line-by-line under an `asyncio.Lock`, so it's safe across
many concurrent workers.

### `CallableWorkload`

Wrap any sync or async callable that returns the next item.

```python
import itertools
seq = itertools.count()
CallableWorkload(fn=lambda: {"seq": next(seq)})

async def async_gen():
    return await fetch_next_from_queue()
CallableWorkload(fn=async_gen)
```

Raise `StopAsyncIteration` from the callable to halt the run.

### `IterableWorkload`

Wrap any iterable or async iterator.

```python
IterableWorkload([{"q": q} for q in queries], loop=False)
```

### `HFDatasetWorkload`

Stream rows from a [HuggingFace dataset](https://huggingface.co/datasets) and
shape them into `{prompt, reference, ...}` items ready for an LLM
workload-type. Requires `pip install -e .[hf]`.

```python
HFDatasetWorkload(
    path="gsm8k", name="main", split="test",
    prompt_field="question",            # row[<this>] -> item["prompt"]
    reference_field="answer",           # row[<this>] -> item["reference"]
    reference_transform="gsm8k_answer", # postprocess the raw value
    extra_fields=("idx",),              # carry through unchanged
    max_items=None,
    shuffle=False, seed=None,
    streaming=False, loop=False,
)
```

**Presets** for common eval datasets (`gsm8k`, `mmlu`, `humaneval`) fill in
sensible defaults — override any of them with explicit kwargs:

```python
HFDatasetWorkload(preset="gsm8k", split="train", max_items=200, shuffle=True)
```

| Preset      | path / name        | prompt field           | reference field + transform               |
| ----------- | ------------------ | ---------------------- | ----------------------------------------- |
| `gsm8k`     | `gsm8k:main`       | `question`             | `answer` → `gsm8k_answer` (strip after `####`) |
| `mmlu`      | `cais/mmlu:all`    | template (Q + 4 choices) | `answer` → `mmlu_letter` (int → A/B/C/D) |
| `humaneval` | `openai_humaneval` | `prompt`               | `canonical_solution` (no transform)        |

**Composing the prompt from multiple columns** (MMLU-style):

```python
HFDatasetWorkload(
    path="cais/mmlu", name="all", split="test",
    prompt_field=None,
    prompt_template=(
        "{question}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer with one letter."
    ),
    prompt_template_fields={
        "question": "question",
        "a": ("choices", 0), "b": ("choices", 1),
        "c": ("choices", 2), "d": ("choices", 3),
    },
    reference_field="answer",
    reference_transform="mmlu_letter",
)
```

`prompt_template_fields` maps a template variable to either a row field name
(`"question"`) or a tuple `(field, idx, ...)` for indexing into list/dict
fields.

**Reference transforms** registered out of the box (string names):

| Name           | Behavior                                                   |
| -------------- | ---------------------------------------------------------- |
| `identity`     | unchanged                                                  |
| `strip`        | `s.strip()`                                                |
| `lower_strip`  | `s.strip().lower()`                                        |
| `first_line`   | first non-empty line                                       |
| `gsm8k_answer` | extract text after `####`, drop commas; fall back to first integer |
| `mmlu_letter`  | int / digit-string → A/B/C/D; otherwise first letter uppercased |

Pass a callable directly for custom postprocessing:

```python
HFDatasetWorkload(
    path="my/ds",
    reference_transform=lambda value: value["target"]["text"],
)
```

**Streaming and loop** are mutually exclusive with shuffle: HF's
`streaming=True` returns an iterable that can't be reindexed, and `loop=True`
is honoured only in non-streaming mode. `max_items` always applies and is the
safest way to cap a run.

YAML — `type: hf` (or `huggingface`) passes everything through to the
constructor; use `preset:` for a one-liner:

```yaml
workload:
  type: hf
  preset: gsm8k
  split: test
  max_items: 200
  shuffle: true
  seed: 0
```

### `TraceLabWorkload`

Replays the [TraceLab](https://github.com/uw-syfi/TraceLab) coding-agent trace
(real Claude Code / Codex LLM rounds) as token-faithful chat requests. The
dataset is sanitized — it carries each round's token accounting (`prefix_tokens`
+ `newly_append_tokens` = `input_tokens_total`, `output_tokens`) and session
structure, but no prompt text — so the workload **synthesizes** prompts sized
to the recorded tokens. See [TraceLab benchmark](tracelab.md) for the full
walkthrough.

```python
from benchmaker import TraceLabWorkload

workload = TraceLabWorkload(
    ".local/syfi_coding_trace.jsonl",
    prefix_cache=True,        # byte-exact growing session prefixes
    match_output_tokens=True, # force the recorded decode length (vLLM/SGLang)
    max_tokens_cap=1024,
    provider="claude",
    max_sessions=500,
    chars_per_token=4.0,      # char-mode sizing (use tokenizer= for exact)
)
```

Pair it with `OpenAIChatWorkloadType(passthrough_meta=True)` so the trace
metadata is recorded into each sample's `meta` (not sent to the server). The
recorded `prompt_tokens_hint` (target input) and `prefix_tokens_hint` (target
cacheable prefix) are promoted into `Sample.extra`, where they sit beside the
server-reported `prompt_tokens` / `cached_tokens` for direct target-vs-realized
comparison. Get the dataset with `python tools/tracelab/prepare.py`.

YAML: `type: tracelab` (accepts every constructor kwarg).

### `AgentXWorkload`

Replays the [SemiAnalysis AgentX](https://inferencex.semianalysis.com/agentx)
v1.0 agentic-coding corpus (WEKA-format Claude Code sessions with subagent
topology, no prompt text): block-hash synthetic prompts that preserve the
recording's byte-exact prefix-cache structure, session-tree DAG dispatch
(concurrent subagent streams, recorded spawn/join/gap timing), a per-lane
primer + warmup pass excluded from metrics, and per-play cache-bust markers.
See [AgentX benchmark](agentx.md) for the full walkthrough.

```python
from benchmaker import AgentXWorkload

workload = AgentXWorkload(
    ".local/agentx-traces-weka-062126.jsonl",
    concurrency=8,             # live session trees
    warmup_requests=10,        # unmeasured warmup per lane (after primer)
    max_context=131072,        # carve corpus to the server's context window
    chars_per_token=4.0,       # char-mode sizing (use tokenizer= for exact)
)
```

Pair it with `OpenAIChatWorkloadType(passthrough_meta=True)` and install the
workload's `completion_hook()` on the runner — the DAG advances through the
post-hook. Get the dataset with `python tools/agentx/prepare.py`.

YAML: `type: agentx` (accepts every constructor kwarg).

---

## Custom workload

```python
import random
from benchmaker import Workload

class WeightedSampler(Workload):
    name = "weighted"
    def __init__(self, items, weights):
        self._items, self._weights = items, weights
    async def next_item(self):
        return random.choices(self._items, weights=self._weights, k=1)[0]
```

That's the whole interface. Add `setup` / `aclose` as needed (the base class
provides default no-ops).

## Custom workload-type

```python
import json
from benchmaker import Request, Sample, WorkloadType

class GraphQLWorkloadType(WorkloadType):
    name = "graphql"
    streaming = False

    def __init__(self, url):
        self.url = url

    async def make_request(self, item):
        return Request(method="POST", url=self.url, json={"query": item})

    async def make_sample(self, item, req, resp, start_ts):
        sample = await super().make_sample(item, req, resp, start_ts)
        obj = json.loads(resp.body or b"{}")
        if obj.get("errors"):
            sample.ok = False
            sample.error = "graphql errors"
        return sample
```

For streaming workloads, set `streaming = True` on the class. The runner will
then populate `Response.stream_chunks` (list of byte chunks) and
`Response.stream_chunk_times` (seconds since request start) so you can
compute per-chunk metrics.
