"""benchmaker: async HTTP benchmarking with pluggable workload-types + workloads (datasets)."""

from benchmaker.core.types import (
    Request,
    Response,
    Sample,
    PreRequestHook,
    PostResponseHook,
)
from benchmaker.workloads.base import WorkloadType
from benchmaker.workloads.datasets import (
    Workload,
    StaticWorkload,
    JsonlWorkload,
    CallableWorkload,
    IterableWorkload,
)
from benchmaker.workloads.http import HttpWorkloadType
from benchmaker.workloads.llm import OpenAIChatWorkloadType
from benchmaker.workloads.sandbox import SandboxWorkloadType
from benchmaker.workloads.hf import HFDatasetWorkload
from benchmaker.workloads.rag import DeepRAGWorkload
from benchmaker.workloads.sglang import SGLangGenerateWorkloadType
from benchmaker.workloads.agentic import AgenticWorkload
from benchmaker.workloads.tracelab import TraceLabWorkload
from benchmaker.workloads.agentx import AgentXWorkload
from benchmaker.workloads.agent import (
    Agent,
    AgentContext,
    AgentResult,
    AgentWorkloadType,
    CallableAgent,
)
from benchmaker.workloads.eval import (
    EvalWorkloadType,
    Scorer,
    correctness_hook,
    extract_openai_text,
    extract_raw_text,
    extract_text,
    exact_match,
    contains,
    regex_match,
    json_valid,
    multiple_choice,
    judge_llm,
    openai_chat_judge,
)
from benchmaker.core.load import (
    LoadModel,
    ConstantRPS,
    PoissonRPS,
    ClosedLoop,
    Sweep,
    Ramp,
    parse_rate_spec,
)
from benchmaker.env import interpolate, load_dotenv
from benchmaker.core.monitors import (
    Monitor,
    FunctionMonitor,
    PrometheusMonitor,
    parse_prometheus,
)
from benchmaker.core.runner import BenchLane, BenchRunner, BenchConfig, BenchResult
from benchmaker.core.trace import (
    ReplayWorkloadType,
    TracePacedLoad,
    TraceRecorder,
    TraceWorkload,
    load_trace,
)
from benchmaker.io.bundle import (
    BUNDLE_VERSION,
    RunMeta,
    default_run_id,
    is_bundle_dir,
    iter_jsonl,
    read_bundle,
    write_bundle,
)

__all__ = [
    "Request",
    "Response",
    "Sample",
    "PreRequestHook",
    "PostResponseHook",
    # workload-types (protocols)
    "WorkloadType",
    "HttpWorkloadType",
    "OpenAIChatWorkloadType",
    "SandboxWorkloadType",
    "HFDatasetWorkload",
    "DeepRAGWorkload",
    "SGLangGenerateWorkloadType",
    "AgenticWorkload",
    "TraceLabWorkload",
    "AgentXWorkload",
    # agent workload (pluggable user-defined agents)
    "Agent",
    "AgentContext",
    "AgentResult",
    "AgentWorkloadType",
    "CallableAgent",
    # eval / correctness
    "EvalWorkloadType",
    "Scorer",
    "correctness_hook",
    "extract_openai_text",
    "extract_raw_text",
    "extract_text",
    "exact_match",
    "contains",
    "regex_match",
    "json_valid",
    "multiple_choice",
    "judge_llm",
    "openai_chat_judge",
    # workloads (datasets / input sources)
    "Workload",
    "StaticWorkload",
    "JsonlWorkload",
    "CallableWorkload",
    "IterableWorkload",
    # load models
    "LoadModel",
    "ConstantRPS",
    "PoissonRPS",
    "ClosedLoop",
    "Sweep",
    "Ramp",
    "parse_rate_spec",
    # monitors
    "Monitor",
    "FunctionMonitor",
    "PrometheusMonitor",
    "parse_prometheus",
    # env
    "load_dotenv",
    "interpolate",
    # runner
    "BenchRunner",
    "BenchConfig",
    "BenchLane",
    "BenchResult",
    # trace: record & replay
    "TraceRecorder",
    "ReplayWorkloadType",
    "TraceWorkload",
    "TracePacedLoad",
    "load_trace",
    # bundle / output layout
    "BUNDLE_VERSION",
    "RunMeta",
    "default_run_id",
    "is_bundle_dir",
    "iter_jsonl",
    "read_bundle",
    "write_bundle",
]

__version__ = "0.1.4"
