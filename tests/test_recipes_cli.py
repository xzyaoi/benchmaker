"""CLI-level tests for the recipe framework (registry + per-recipe commands).

These drive the real click commands via ``CliRunner``. Because each recipe
command calls ``asyncio.run`` internally, the stub HTTP server runs in a
background thread with its own event loop (the async ``stub_server`` fixture in
conftest can't be used here — it would already hold a running loop).
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading

import pytest
from aiohttp import web
from click.testing import CliRunner

from benchmaker.cli import main
from benchmaker.recipes import all_recipes, get
from benchmaker.recipes._factory import SHARED_DESTS, make_command


# ---------------------------------------------------------------- live server


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


async def _hello(request: web.Request) -> web.Response:
    return web.json_response({"ok": True})


async def _sse(request: web.Request) -> web.StreamResponse:
    resp = web.StreamResponse(status=200,
                              headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    for i in range(3):
        chunk = {"choices": [{"index": 0, "delta": {"content": f"t{i} "},
                              "finish_reason": None}]}
        await resp.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
    await resp.write(b"data: " + json.dumps({
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
    }).encode() + b"\n\n")
    await resp.write(b"data: [DONE]\n\n")
    await resp.write_eof()
    return resp


async def _sb_create(request: web.Request) -> web.Response:
    return web.json_response({"id": "sb-0001"})


async def _sb_exec(request: web.Request) -> web.Response:
    return web.json_response({"stdout": "hi\n", "stderr": "", "exit_code": 0,
                              "duration": 0.001})


_cli_files: dict = {}


async def _sb_put_file(request: web.Request) -> web.Response:
    sid = request.match_info["sid"]
    path = request.query.get("path") or "/tmp/benchmaker.bin"
    _cli_files[(sid, path)] = await request.read()
    return web.json_response({"ok": True, "bytes": len(_cli_files[(sid, path)])})


async def _sb_get_file(request: web.Request) -> web.Response:
    sid = request.match_info["sid"]
    path = request.query.get("path") or "/tmp/benchmaker.bin"
    return web.Response(body=_cli_files.get((sid, path), b""),
                        content_type="application/octet-stream")


@pytest.fixture
def live_server():
    """Run a stub aiohttp app in a background thread; yield its base URL."""
    app = web.Application()
    app.router.add_get("/hello", _hello)
    app.router.add_post("/v1/chat/completions", _sse)
    for prefix in ("/sandboxes", "/native/sandboxes"):
        app.router.add_post(prefix, _sb_create)
        app.router.add_post(prefix + "/{sid}/exec", _sb_exec)
        app.router.add_put(prefix + "/{sid}/files", _sb_put_file)
        app.router.add_get(prefix + "/{sid}/files", _sb_get_file)
        app.router.add_delete(prefix + "/{sid}",
                              lambda r: web.json_response({"ok": True}))

    port = _free_port()
    loop = asyncio.new_event_loop()
    ready = threading.Event()
    runner_box: dict = {}

    def _serve():
        asyncio.set_event_loop(loop)
        runner = web.AppRunner(app)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, "127.0.0.1", port)
        loop.run_until_complete(site.start())
        runner_box["runner"] = runner
        ready.set()
        loop.run_forever()

    t = threading.Thread(target=_serve, daemon=True)
    t.start()
    ready.wait(timeout=5)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=5)


# ---------------------------------------------------------------- registry


def test_registry_has_expected_recipes():
    names = {r.name for r in all_recipes()}
    assert names == {"http", "llm", "sglang", "sandbox", "swebench",
                     "swebench-replay", "agentic", "pareval",
                     "tracelab", "agentx"}


def test_recipes_registered_as_subcommands():
    for name in ("http", "llm", "sandbox", "swebench"):
        assert name in main.commands


def test_no_recipe_option_clashes_with_shared():
    # make_command() raises if a recipe declares an option dest in SHARED_DESTS.
    for recipe in all_recipes():
        cmd = make_command(recipe)
        recipe_dests = {p.name for p in cmd.params} - SHARED_DESTS
        assert recipe_dests, f"{recipe.name} declares no recipe-specific options"


# ---------------------------------------------------------------- help layout


def test_llm_help_composes_shared_and_recipe_options():
    res = CliRunner().invoke(main, ["llm", "--help"])
    assert res.exit_code == 0
    for flag in ("--rate", "--duration", "--out-dir", "--prompt", "--model"):
        assert flag in res.output


# ---------------------------------------------------------------- happy paths


def test_http_recipe_runs(live_server):
    res = CliRunner().invoke(main, [
        "http", "--url", f"{live_server}/hello",
        "--rate", "20", "--duration", "0.3s", "--quiet",
    ])
    assert res.exit_code == 0, res.output
    assert "requests" in res.output.lower() or res.output.strip()


def test_llm_recipe_runs(live_server):
    res = CliRunner().invoke(main, [
        "llm", "--url", f"{live_server}/v1/chat/completions",
        "--model", "stub", "--prompt", "hello",
        "--rate", "5", "--duration", "0.3s", "--dotenv", "", "--quiet",
    ])
    assert res.exit_code == 0, res.output


def test_sandbox_recipe_runs(live_server):
    res = CliRunner().invoke(main, [
        "sandbox", "--base-url", live_server, "--command", "echo hi",
        "--rate", "5", "--duration", "0.3s", "--quiet",
    ])
    assert res.exit_code == 0, res.output


def test_sandbox_file_recipe_runs(live_server):
    res = CliRunner().invoke(main, [
        "sandbox", "--base-url", live_server, "--operation", "file",
        "--file-path", "/tmp/cli-blob.bin", "--file-content", "hello-cli",
        "--rate", "5", "--duration", "0.3s", "--quiet",
    ])
    assert res.exit_code == 0, res.output


def test_sandbox_file_options_reach_workload():
    """--file-* flags must be forwarded to the workload-type and recorded in
    the reproducible source_config (and only in file mode)."""
    from benchmaker.recipes import get
    from benchmaker.recipes.base import SharedOpts

    recipe = get("sandbox")
    shared = SharedOpts(
        rate="5", duration="0.3s", max_requests=None, timeout_s=30.0,
        connection_limit=100, dotenv=None, quiet=True,
        out_dir=None, run_id=None, labels=(), notes="",
    )
    built = recipe.build(
        shared, base_url="http://x", operation="file", command=(),
        image=None, spec_json=None, endpoint_prefix="/sandboxes",
        ttl_seconds=None, persistent=False, sandbox_id=None, header=(),
        file_path="/tmp/cli-blob.bin", file_content="hello-cli",
        file_verify_with_exec=True, file_verify_command="md5sum /tmp/cli-blob.bin",
    )
    wt = built.workload_type
    assert wt._file_path == "/tmp/cli-blob.bin"
    assert wt._file_content == b"hello-cli"
    assert wt._file_verify_with_exec is True
    sc = built.source_config["workload_type"]
    assert sc["file_path"] == "/tmp/cli-blob.bin"
    assert sc["file_content"] == "hello-cli"
    assert sc["file_verify_with_exec"] is True


# ---------------------------------------------------------------- validation


def test_llm_prompt_sources_mutually_exclusive(tmp_path, live_server):
    jsonl = tmp_path / "p.jsonl"
    jsonl.write_text('{"prompt": "x"}\n')
    res = CliRunner().invoke(main, [
        "llm", "--url", f"{live_server}/v1/chat/completions", "--model", "m",
        "--prompt", "a", "--prompts-jsonl", str(jsonl), "--dotenv", "",
    ])
    assert res.exit_code != 0
    assert "mutually exclusive" in res.output


def test_quick_alias_warns_deprecated(live_server):
    res = CliRunner().invoke(main, [
        "quick", "--url", f"{live_server}/hello",
        "--rate", "10", "--duration", "0.2s", "--quiet",
    ])
    assert res.exit_code == 0, res.output
    assert "deprecated" in res.output.lower()


# ---------------------------------------------------------------- swebench (harbor)


def test_swebench_is_self_driving():
    """swebench drives harbor itself, so it opts out of the shared load block."""
    recipe = get("swebench")
    assert recipe.wants_load_options is False


def test_swebench_command_exposes_harbor_flags():
    res = CliRunner().invoke(main, ["swebench", "--help"])
    assert res.exit_code == 0, res.output
    for flag in ("--agent", "--dataset", "--concurrency", "--n-tasks", "--dotenv"):
        assert flag in res.output
    # The load/output flags should NOT be on a self-driving recipe.
    for absent in ("--rate", "--out-dir", "--connection-limit"):
        assert absent not in res.output


def test_swebench_default_agent_is_pi():
    res = CliRunner().invoke(main, ["swebench", "--help"])
    assert "pi" in res.output
    # default shown for --agent
    assert "[default: pi]" in res.output


def test_swebench_list_agents_no_cluster():
    """--list-agents short-circuits before any cluster/model checks."""
    res = CliRunner().invoke(main, ["swebench", "--list-agents"])
    assert res.exit_code == 0, res.output
    assert "pi" in res.output and "coding-agent" in res.output


def test_swebench_requires_sandbox_url(monkeypatch):
    """Without FLASH_SANDBOX_URL the recipe errors cleanly (no harbor run)."""
    monkeypatch.delenv("FLASH_SANDBOX_URL", raising=False)
    monkeypatch.delenv("FLASH_SANDBOX_HOST", raising=False)
    res = CliRunner().invoke(main, ["swebench", "--model", "m", "--dotenv", ""])
    assert res.exit_code != 0
    assert "FLASH_SANDBOX_URL" in res.output


# ---------------------------------------------------------------- full-jsonl-row

import glob
import os
import tempfile


def test_llm_full_jsonl_row_records_metadata(live_server):
    with tempfile.TemporaryDirectory() as d:
        rows = os.path.join(d, "rows.jsonl")
        with open(rows, "w") as f:
            f.write(json.dumps({
                "messages": [{"role": "user", "content": "hi"}],
                "conversation_id": "conv-42",
            }) + "\n")
        out = os.path.join(d, "runs")
        res = CliRunner().invoke(main, [
            "llm", "--url", f"{live_server}/v1/chat/completions",
            "--model", "stub", "--prompts-jsonl", rows, "--full-jsonl-row",
            "--rate", "5", "--duration", "0.3s", "--dotenv", "",
            "--out-dir", out, "--quiet",
        ])
        assert res.exit_code == 0, res.output
        samples = glob.glob(os.path.join(out, "**", "samples.jsonl"), recursive=True)
        assert samples, "no samples.jsonl written"
        recs = [json.loads(l) for l in open(samples[0]) if l.strip()]
        assert recs and all(r["meta"].get("conversation_id") == "conv-42"
                            for r in recs)
