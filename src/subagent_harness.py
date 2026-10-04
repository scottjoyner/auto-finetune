"""Shared subagent harness framework.

The framework keeps the model-specific pieces (driver construction, tool-call
dialect, defaults, and MCP tool names) in profiles while providing one tested
JSON-RPC server and task loop for every subagent type.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.bench import ModelDriver, Task, TaskResult, ToolEnv, run_task

DEFAULT_TOOLS = ("bash", "read", "write", "edit", "list")
MAX_TURNS = 100


def _tool_schema(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str] | None = None,
) -> dict:
    return {
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "required": required or [],
        },
    }


def direct_tool_specs(names: tuple[str, ...] | list[str]) -> list[dict]:
    specs = {
        "bash": _tool_schema(
            "bash", "Run a shell command in the sandbox.",
            {"command": {"type": "string"}}, ["command"]),
        "read": _tool_schema(
            "read", "Read file contents.",
            {"filePath": {"type": "string"}, "path": {"type": "string"}},
            ["filePath"]),
        "write": _tool_schema(
            "write", "Write literal content to a file (overwrites).",
            {"filePath": {"type": "string"}, "path": {"type": "string"},
             "content": {"type": "string"}}, ["filePath", "content"]),
        "edit": _tool_schema(
            "edit", "Replace exact text in a file.",
            {"filePath": {"type": "string"}, "path": {"type": "string"},
             "oldText": {"type": "string"}, "oldString": {"type": "string"},
             "newText": {"type": "string"}, "newString": {"type": "string"}},
            ["filePath"]),
        "list": _tool_schema(
            "list", "List directory contents.",
            {"path": {"type": "string"}}, ["path"]),
    }
    return [specs[name] for name in names if name in specs]


def _task_properties(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    props: dict[str, Any] = {
        "prompt": {"type": "string", "description": "Task or instruction."},
        "task_id": {"type": "string", "description": "Optional result task id."},
        "kind": {"type": "string", "enum": ["exec", "replay"],
                 "description": "Benchmark task kind (default exec)."},
        "max_turns": {"type": "integer", "description": "Max agent turns."},
        "checks": {"type": "array", "description": "Verifier checks."},
        "tools": {"type": "array", "description": "Tools available to the task."},
        "gen_max_tokens": {"type": "integer", "description": "Max tokens per model turn."},
    }
    props.update(extra or {})
    return props


def _task_schema(description: str, extra: dict[str, Any] | None = None,
                 required: list[str] | None = None) -> dict:
    props = _task_properties(extra)
    return {
        "type": "object",
        "properties": props,
        "required": required if required is not None else ["prompt"],
    }


@dataclass
class HarnessProfile:
    """Configuration for one model/subagent family."""

    name: str
    server_name: str
    server_version: str
    tool_name: str
    description: str
    input_schema: dict
    default_max_turns: int = 8
    default_model: str = ""
    default_runner: str = ""
    default_gen_max_tokens: int = 512
    driver_arg_names: tuple[str, ...] = ()
    direct_tools: tuple[str, ...] = ()
    transcript_tail: int = 0
    task_id: str = "subagent-task"
    allow_driver_override: bool = True

    def tool_spec(self) -> dict:
        return _tool_schema(
            self.tool_name, self.description,
            self.input_schema.get("properties", {}),
            self.input_schema.get("required", ["prompt"]))

    def model_name(self, args: dict[str, Any]) -> str:
        return (args.get("model_name") or args.get("model_path")
                or args.get("model") or self.default_model or self.server_name)

    def runner_name(self) -> str:
        return self.default_runner or self.name


LOCAL_PROFILE = HarnessProfile(
    name="local",
    server_name="refinedtoolcall-subagent",
    server_version="2.0.0",
    tool_name="run_task",
    description=("Run a task through the optimized local agent loop. The agent "
                 "uses bash/read/write/edit/list tools in a sandbox."),
    input_schema=_task_schema(
        "Run a local model task.",
        {"model_path": {"type": "string", "description": "HF checkpoint dir."},
         "variant": {"type": "string", "enum": ["base", "finetune", "auto"]},
         "rocm": {"type": "boolean"}},
        ["prompt", "model_path"]),
    default_max_turns=12,
    default_model="RefinedToolCallV5-3b",
    default_runner="subagent",
    driver_arg_names=("model_path", "variant", "rocm"),
)

MINICPM5_PROFILE = HarnessProfile(
    name="minicpm5",
    server_name="minicpm5-subagent",
    server_version="2.0.0",
    tool_name="run_task",
    description=("Run a task through the MiniCPM5 endpoint agent loop. The "
                 "server also exposes direct bash and file tools."),
    input_schema=_task_schema(
        "Run a MiniCPM5 task.",
        {"base_url": {"type": "string"}, "model": {"type": "string"},
         "api_key": {"type": "string"}, "temperature": {"type": "number"}},
        ["prompt"]),
    default_model="minicpm5-2b",
    default_runner="minicpm5",
    driver_arg_names=("base_url", "model", "api_key", "temperature"),
    direct_tools=DEFAULT_TOOLS,
    transcript_tail=6,
)

LFM25_PROFILE = HarnessProfile(
    name="lfm25",
    server_name="lfm25-subagent",
    server_version="2.0.0",
    tool_name="lfm_task",
    description=("Run a task through the LFM2.5 endpoint agent loop with "
                 "pythonic tool-call parsing and context compaction."),
    input_schema=_task_schema(
        "Run an LFM2.5 task.",
        {"base_url": {"type": "string"}, "model": {"type": "string"},
         "api_key": {"type": "string"}, "temperature": {"type": "number"}},
        ["prompt"]),
    default_model="lfm2.5-1.2b-instruct",
    default_runner="lfm25",
    driver_arg_names=("base_url", "model", "api_key", "temperature"),
    transcript_tail=6,
)

API_TOOLS_PROFILE = HarnessProfile(
    name="api-tools",
    server_name="api-tools-subagent",
    server_version="2.0.0",
    tool_name="api_task",
    description=("Run a task through an OpenAI-compatible endpoint using "
                 "native structured tool calls."),
    input_schema=_task_schema(
        "Run a native API-tools task.",
        {"base_url": {"type": "string"}, "model": {"type": "string"},
         "api_key": {"type": "string"}, "temperature": {"type": "number"}},
        ["prompt"]),
    default_model="local-model",
    default_runner="api-tools",
    driver_arg_names=("base_url", "model", "api_key", "temperature"),
)

HERMES_PROFILE = HarnessProfile(
    name="hermes",
    server_name="hermes-subagent",
    server_version="2.0.0",
    tool_name="hermes_task",
    description=("Delegate a task to the local Hermes agent harness and return "
                 "its completion result."),
    input_schema=_task_schema(
        "Run a Hermes task.",
        {"hermes_dir": {"type": "string"}, "python": {"type": "string"}},
        ["prompt"]),
    default_model="hermes-agent",
    default_runner="hermes",
    driver_arg_names=("hermes_dir", "python", "extra_args"),
)

HARNESS_PROFILES: dict[str, HarnessProfile] = {
    "local": LOCAL_PROFILE,
    "minicpm5": MINICPM5_PROFILE,
    "lfm25": LFM25_PROFILE,
    "api-tools": API_TOOLS_PROFILE,
    "hermes": HERMES_PROFILE,
}


def _default_root() -> Path:
    return Path(tempfile.mkdtemp(prefix="subagent-"))


def _call_factory(factory: Callable, kwargs: dict[str, Any]):
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory(**kwargs)
    params = signature.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return factory(**kwargs)
    accepted = {name for name, p in params.items()
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                              inspect.Parameter.POSITIONAL_OR_KEYWORD)}
    if set(kwargs) <= accepted:
        return factory(**kwargs)
    positional = [kwargs[name] for name in params
                  if name in kwargs and params[name].kind in (
                      inspect.Parameter.POSITIONAL_ONLY,
                      inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    if positional:
        return factory(*positional)
    filtered = {name: value for name, value in kwargs.items() if name in accepted}
    return factory(**filtered)


def _invoke_run_one(run_one: Callable, driver: ModelDriver, task: Task,
                    model_name: str, runner_name: str, root: Path,
                    gen_max_tokens: int) -> TaskResult:
    try:
        signature = inspect.signature(run_one)
    except (TypeError, ValueError):
        return run_one(driver, task, model_name, runner_name,
                       sandbox_root=root, gen_max_tokens=gen_max_tokens)
    params = signature.parameters
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD
                         for p in params.values())
    if accepts_kwargs or {"sandbox_root", "gen_max_tokens"} <= set(params):
        return run_one(driver, task, model_name, runner_name,
                       sandbox_root=root, gen_max_tokens=gen_max_tokens)
    return run_one(driver, task, model_name, runner_name)


def _coerce_result(result: Any, task: Task, model_name: str,
                   runner_name: str) -> TaskResult:
    if isinstance(result, TaskResult):
        return result
    if isinstance(result, dict):
        return TaskResult(
            task_id=task.id, kind=task.kind, model=model_name,
            runner=runner_name, completed=bool(result.get("completed")),
            turns=int(result.get("api_calls", result.get("turns", 0)) or 0),
            error=str(result.get("error", "")),
        )
    return TaskResult(task_id=task.id, kind=task.kind, model=model_name,
                      runner=runner_name, error="runner returned no result")


def _parse_int(value: Any, default: int, name: str,
               maximum: int = MAX_TURNS) -> int:
    try:
        parsed = default if value is None else int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed < 1 or parsed > maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")
    return parsed


@dataclass
class HarnessContext:
    """Injectable context used by the protocol handler and stdio server."""

    profile: HarnessProfile = field(default_factory=lambda: LOCAL_PROFILE)
    make_driver: Callable[..., ModelDriver] | None = None
    run_one: Callable = run_task
    driver: ModelDriver | None = None
    root: Path | None = None
    root_factory: Callable[[], Path] = _default_root
    default_kwargs: dict[str, Any] = field(default_factory=dict)

    def sandbox_root(self) -> Path:
        if self.root is None:
            self.root = self.root_factory()
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    def resolve_driver(self, args: dict[str, Any]) -> ModelDriver:
        override_keys = set(self.profile.driver_arg_names) & set(args)
        if self.driver is not None and not (
                self.profile.allow_driver_override and override_keys):
            return self.driver
        if self.make_driver is None:
            raise ValueError(f"{self.profile.name} harness has no driver factory")
        kwargs = dict(self.default_kwargs)
        driver_defaults = {
            "variant": "auto", "rocm": False, "temperature": 0.0,
        }
        for name in self.profile.driver_arg_names:
            if name in args:
                kwargs[name] = args[name]
            elif name in driver_defaults and name not in kwargs:
                kwargs[name] = driver_defaults[name]
        return _call_factory(self.make_driver, kwargs)


def run_harness_task(ctx: HarnessContext, args: dict[str, Any]) -> TaskResult:
    """Validate a task request, run it, and return a normalized result."""

    if not isinstance(args, dict):
        raise ValueError("task arguments must be an object")
    effective_args = {**ctx.default_kwargs, **args}
    if "prompt" not in args or args["prompt"] in (None, ""):
        raise ValueError("prompt required")
    for required in ctx.profile.input_schema.get("required", ["prompt"]):
        if required == "prompt":
            continue
        if required not in effective_args or effective_args[required] in (None, ""):
            raise ValueError(f"{required} required")
    prompt = args.get("prompt", "")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt required")
    max_turns = _parse_int(args.get("max_turns"),
                           ctx.profile.default_max_turns, "max_turns")
    gen_max_tokens = _parse_int(args.get("gen_max_tokens"),
                                ctx.profile.default_gen_max_tokens,
                                "gen_max_tokens", maximum=32768)
    checks = args.get("checks", []) or []
    tools = args.get("tools", list(DEFAULT_TOOLS)) or list(DEFAULT_TOOLS)
    if not isinstance(checks, list) or not isinstance(tools, list):
        raise ValueError("checks and tools must be arrays")
    kind = args.get("kind", "exec")
    if kind not in ("exec", "replay"):
        raise ValueError("kind must be exec or replay")
    task = Task(
        id=str(args.get("task_id") or ctx.profile.task_id),
        prompt=prompt,
        kind=kind,
        checks=checks,
        tools=[str(tool) for tool in tools],
        max_turns=max_turns,
        replay_context=args.get("replay_context", []) or [],
    )
    driver = ctx.resolve_driver(args)
    model_name = ctx.profile.model_name(effective_args)
    runner_name = ctx.profile.runner_name()
    result = _invoke_run_one(
        ctx.run_one, driver, task, model_name, runner_name,
        ctx.sandbox_root(), gen_max_tokens)
    return _coerce_result(result, task, model_name, runner_name)


def format_task_result(result: TaskResult, ctx: HarnessContext) -> str:
    lines = [f"completed={result.completed}", f"turns={result.turns}",
             f"success={result.success}"]
    if result.checks_total:
        lines.append(f"checks={result.checks_passed}/{result.checks_total}")
    if result.error:
        lines.append(f"error={result.error}")
    for entry in result.transcript[-ctx.profile.transcript_tail:]:
        role = entry.get("role", "?") if isinstance(entry, dict) else "?"
        body = ""
        if isinstance(entry, dict):
            body = str(entry.get("content") or entry.get("detail") or "")
        text = body[:300].replace("\n", " ")
        lines.append(f"[{role}] {text}")
    return "\n".join(lines)


def result_block(text: str, is_error: bool = False) -> dict:
    return {"content": [{"type": "text", "text": text}],
            "isError": is_error}


def _error(msg_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id,
            "error": {"code": code, "message": message}}


def handle_request(req: dict, ctx: HarnessContext) -> dict | None:
    """Handle one JSON-RPC request for any configured harness profile."""

    method = req.get("method", "") if isinstance(req, dict) else ""
    msg_id = req.get("id") if isinstance(req, dict) else None
    params = req.get("params", {}) or {} if isinstance(req, dict) else {}
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": msg_id,
                "result": {"protocolVersion": "2024-11-05",
                           "capabilities": {"tools": {}},
                           "serverInfo": {"name": ctx.profile.server_name,
                                          "version": ctx.profile.server_version}}}
    if method == "notifications/initialized":
        return None
    if method == "tools/list":
        tools = [ctx.profile.tool_spec()]
        tools.extend(direct_tool_specs(ctx.profile.direct_tools))
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": tools}}
    if method == "tools/call":
        name = params.get("name") if isinstance(params, dict) else None
        args = params.get("arguments", {}) if isinstance(params, dict) else {}
        if name in ctx.profile.direct_tools:
            result = ToolEnv(ctx.sandbox_root()).execute(name, args)
            is_error = str(result).startswith("ERROR") or "blocked" in str(result).lower()
            return {"jsonrpc": "2.0", "id": msg_id,
                    "result": result_block(str(result), is_error)}
        if name != ctx.profile.tool_name:
            return _error(msg_id, -32601, f"unknown tool: {name}")
        try:
            task_result = run_harness_task(ctx, args)
            text = format_task_result(task_result, ctx)
            is_error = bool(task_result.error) or (
                task_result.checks_total > 0 and not task_result.success)
            return {"jsonrpc": "2.0", "id": msg_id,
                    "result": result_block(text, is_error)}
        except ValueError as exc:
            return _error(msg_id, -32602, str(exc))
        except Exception as exc:
            return _error(msg_id, -32603, str(exc))
    if msg_id is not None:
        return _error(msg_id, -32601, f"method not found: {method}")
    return None


def _build_driver(kind: str, **kwargs: Any) -> ModelDriver:
    if kind == "local":
        from src.bench import make_driver
        model_path = kwargs.pop("model_path", None) or kwargs.pop("model", None)
        if not model_path:
            raise ValueError("local harness requires model_path")
        return make_driver("subagent", model_path=model_path,
                           rocm=bool(kwargs.pop("rocm", False)),
                           variant=kwargs.pop("variant", "auto"))
    if kind == "minicpm5":
        from src.subagent_minicpm5 import make_minicpm_driver
        model_path = kwargs.pop("model_path", None)
        model = kwargs.pop("model", None) or model_path or "minicpm5-2b"
        return make_minicpm_driver(
            model_path=model, base_url=kwargs.pop("base_url", None) or
            "http://127.0.0.1:38899/v1",
            api_key=kwargs.pop("api_key", "") or os.environ.get("MINICPM5_API_KEY", ""),
            temperature=float(kwargs.pop("temperature", 0.0)))
    if kind == "lfm25":
        from src.drivers_lfm25 import LFM25Driver
        return LFM25Driver(
            base_url=kwargs.pop("base_url", None) or "http://127.0.0.1:8095",
            model=kwargs.pop("model", None) or "lfm2.5-1.2b-instruct",
            api_key=kwargs.pop("api_key", ""),
            temperature=float(kwargs.pop("temperature", 0.2)))
    if kind == "api-tools":
        from src.drivers_api_tools import ApiToolsDriver
        base_url = kwargs.pop("base_url", None)
        if not base_url:
            raise ValueError("api-tools harness requires base_url")
        return ApiToolsDriver(
            base_url=base_url, model=kwargs.pop("model", None) or "local-model",
            api_key=kwargs.pop("api_key", "") or os.environ.get("OPENAI_API_KEY", ""),
            temperature=float(kwargs.pop("temperature", 0.0)))
    if kind == "hermes":
        from src.bench import HermesDriver
        return HermesDriver(
            hermes_dir=kwargs.pop("hermes_dir", None) or "/home/scott/git/hermes-agent",
            python=kwargs.pop("python", None) or "python3",
            extra_args=kwargs.pop("extra_args", None))
    raise ValueError(f"unknown harness type: {kind}")


def make_default_context(kind: str, *, make_driver: Callable[..., ModelDriver] | None = None,
                         run_one: Callable = run_task,
                         **kwargs: Any) -> HarnessContext:
    """Build a context with the standard driver for a harness type."""

    if kind not in HARNESS_PROFILES:
        raise ValueError(f"unknown harness type: {kind}")
    defaults = dict(kwargs)
    factory = make_driver or (lambda **driver_kwargs: _build_driver(kind, **driver_kwargs))
    return HarnessContext(profile=HARNESS_PROFILES[kind],
                          make_driver=factory, run_one=run_one,
                          default_kwargs=defaults)


def serve_stdio(ctx: HarnessContext) -> None:
    """Serve the JSON-RPC protocol over newline-delimited stdio."""

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = handle_request(req, ctx)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Multi-type subagent harness (MCP stdio)")
    parser.add_argument("--type", "--harness", dest="harness",
                        choices=sorted(HARNESS_PROFILES), required=True)
    parser.add_argument("--model", "--model-path", dest="model")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--variant", default="auto",
                        choices=["base", "finetune", "auto"])
    parser.add_argument("--rocm", action="store_true")
    parser.add_argument("--hermes-dir")
    args = parser.parse_args(argv)
    kwargs: dict[str, Any] = {}
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature
    if args.model:
        kwargs["model_path" if args.harness == "local" else "model"] = args.model
    if args.base_url:
        kwargs["base_url"] = args.base_url
    if args.api_key:
        kwargs["api_key"] = args.api_key
    if args.variant:
        kwargs["variant"] = args.variant
    if args.rocm:
        kwargs["rocm"] = True
    if args.hermes_dir:
        kwargs["hermes_dir"] = args.hermes_dir
    if args.harness == "minicpm5" and not kwargs.get("api_key"):
        kwargs["api_key"] = os.environ.get("MINICPM5_API_KEY", "")
    ctx = make_default_context(args.harness, **kwargs)
    serve_stdio(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
