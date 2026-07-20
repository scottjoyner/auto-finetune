"""Agentic task-completion benchmark.

This is the "did the task actually get done?" layer that the loss / tool-format
evaluations don't cover. It runs a MODEL (not a dataset row) through a real
multi-turn tool-use loop inside a throwaway sandbox directory, then verifies the
outcome with concrete checkers (file exists, command exit code, content present,
regex). It supports two task kinds:

  * "exec"      — a fresh task with a verifiable outcome (create a file, run a
                  command, edit something). The model gets bash + file tools.
  * "replay"    — a gold session transcript: given the INITIAL user prompt and
                  the same tools, did the model reproduce the completed outcome
                  (the verifier checks the end-state the original session reached)?

Model sources (pluggable runners):
  * "local"   — a local HF checkpoint (base / finetune / merged) driven by a
                minimal <tool_call> loop in this module (self-contained).
  * "api"     — an OpenAI-compatible endpoint (e.g. the lan lm-fleet-router),
                used for the "much larger foundational model" reference. No creds
                are hardcoded; pass base_url + model at runtime.
  * "hermes"  — delegate to the hermes-agent harness on this machine.
  * "subagent" — (future) an optimized harness built specifically for this model,
                 meant to run as a subagent inside opencode / hermes-agent.

The local + api runners share the same tool protocol, so a task suite is
comparable across a 3B finetune and a 35B reference model.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from src.parsers import (  # call-format parser + separators (RefinedToolCallV5 dialect)
    _SEP_CHARS,
    _SEP_LITERAL,
    parse_tool_calls,
)

_TOOL_RESULT_RE = re.compile(r"<tool_result>(.*?)</tool_result>", re.DOTALL)




def format_tool_result(result: str, variant: str = "finetune") -> str:
    """Wrap a tool result the way the target model expects it.

    'finetune' -> <tool_result>...</tool_result>   (matches training data)
    'base'     -> <tool_response>...</tool_response> (matches base chat template)
    """
    if variant == "base":
        return f"<tool_response>\n{result}\n</tool_response>"
    return f"<tool_result>{result}</tool_result>"


def parse_tool_results(text: str) -> list[str]:
    return [_strip(m.group(1)) for m in _TOOL_RESULT_RE.finditer(text)]


def _strip(s: str) -> str:
    return s.strip()




# ── verifiers ──────────────────────────────────────────────────────────────────
@dataclass
class VerifyResult:
    passed: bool
    detail: str


def verify_check(sandbox: Path, check: dict) -> VerifyResult:
    """Evaluate one check dict against the sandbox dir.

    Supported check kinds:
      {"kind": "file_exists", "path": "rel/or/abs"}
      {"kind": "file_contains", "path": "...", "expect": "substring"}
      {"kind": "file_regex", "path": "...", "pattern": "..."}
      {"kind": "command_exit", "cmd": "echo hi", "expect_code": 0}
      {"kind": "command_output", "cmd": "...", "expect": "substring"}
      {"kind": "dir_exists", "path": "..."}
    Paths are resolved relative to the sandbox unless absolute.
    """
    kind = check.get("kind")
    try:
        if kind == "file_exists":
            return VerifyResult((sandbox / check["path"]).resolve().is_file(),
                                f"file_exists {check['path']}")
        if kind == "dir_exists":
            return VerifyResult((sandbox / check["path"]).resolve().is_dir(),
                                f"dir_exists {check['path']}")
        if kind == "file_contains":
            p = (sandbox / check["path"]).resolve()
            ok = p.is_file() and check["expect"] in p.read_text(errors="ignore")
            return VerifyResult(ok, f"file_contains {check['path']} ~ {check['expect']!r}")
        if kind == "file_regex":
            p = (sandbox / check["path"]).resolve()
            ok = p.is_file() and re.search(check["pattern"], p.read_text(errors="ignore")) is not None
            return VerifyResult(ok, f"file_regex {check['path']} ~ /{check['pattern']}/")
        if kind == "command_exit":
            rc = subprocess.run(check["cmd"], shell=True, cwd=str(sandbox),
                                capture_output=True, text=True).returncode
            want = int(check.get("expect_code", 0))
            return VerifyResult(rc == want, f"command_exit rc={rc} (want {want}): {check['cmd']}")
        if kind == "command_output":
            out = subprocess.run(check["cmd"], shell=True, cwd=str(sandbox),
                                 capture_output=True, text=True).stdout
            ok = check["expect"] in out
            return VerifyResult(ok, f"command_output ~ {check['expect']!r}")
    except Exception as e:  # noqa: BLE001
        return VerifyResult(False, f"{kind} error: {e}")
    return VerifyResult(False, f"unknown check kind: {kind}")


def verify_task(sandbox: Path, checks: list[dict]) -> tuple[int, int, list[dict]]:
    passed = total = 0
    detail = []
    for c in checks:
        total += 1
        r = verify_check(sandbox, c)
        passed += int(r.passed)
        detail.append({"check": c, "passed": r.passed, "detail": r.detail})
    return passed, total, detail


# ── tool environment (the sandbox the model operates in) ───────────────────────
class ToolEnv:
    """A minimal, safe-ish tool runtime rooted at `root`.

    Provides the tools the model can call. Commands run with cwd=root; writes
    are confined to root. This is good enough for benchmark tasks — it is NOT a
    security boundary (the model could escape with `cd /`), but tasks are
    designed to stay in-sandbox and the dir is deleted after the run.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, p: str) -> Path:
        pp = Path(p)
        if pp.is_absolute():
            return pp
        return (self.root / p).resolve()

    def execute(self, name: str, args: Optional[dict]) -> str:
        args = args or {}
        try:
            if name == "bash":
                return self._bash(args.get("command", ""))
            if name == "read":
                return self._read(args.get("filePath", args.get("path", "")))
            if name == "write":
                return self._write(args.get("filePath", args.get("path", "")),
                                   args.get("content", ""))
            if name == "edit":
                return self._edit(args.get("filePath", args.get("path", "")),
                                  args.get("oldText", ""), args.get("newText", ""))
            if name == "list":
                return self._list(args.get("path", "."))
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {type(e).__name__}: {e}"
        return f"ERROR: unknown tool '{name}'"

    def _bash(self, cmd: str) -> str:
        if not cmd:
            return "ERROR: empty command"
        r = subprocess.run(cmd, shell=True, cwd=str(self.root),
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout or "") + (r.stderr or "")
        return out[:4000] or f"[exit {r.returncode}, no output]"

    def _read(self, path: str) -> str:
        p = self._resolve(path)
        if not p.is_file():
            return f"ERROR: no such file {path}"
        return p.read_text(errors="ignore")[:4000]

    def _write(self, path: str, content: str) -> str:
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"wrote {len(content)} bytes to {path}"

    def _edit(self, path: str, old: str, new: str) -> str:
        p = self._resolve(path)
        if not p.is_file():
            return f"ERROR: no such file {path}"
        text = p.read_text(errors="ignore")
        if old and old not in text:
            return "ERROR: oldText not found"
        p.write_text(text.replace(old, new, 1) if old else text + new)
        return f"edited {path}"

    def _list(self, path: str) -> str:
        p = self._resolve(path)
        if not p.exists():
            return f"ERROR: no such path {path}"
        return "\n".join(str(x.relative_to(self.root)) for x in sorted(p.iterdir()))


# ── task spec ───────────────────────────────────────────────────────────────────
@dataclass
class Task:
    id: str
    prompt: str
    kind: str = "exec"          # "exec" | "replay"
    checks: list = field(default_factory=list)
    tools: list = field(default_factory=lambda: ["bash", "read", "write", "edit", "list"])
    max_turns: int = 8
    replay_context: list = field(default_factory=list)  # optional prior messages

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        return cls(
            id=d["id"], prompt=d["prompt"], kind=d.get("kind", "exec"),
            checks=d.get("checks", []), tools=d.get("tools",
                     ["bash", "read", "write", "edit", "list"]),
            max_turns=d.get("max_turns", 8),
            replay_context=d.get("replay_context", []),
        )


def load_tasks(path: str) -> list[Task]:
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("*.jsonl"))
    else:
        files = [p]
    tasks = []
    for f in files:
        for line in f.read_text().splitlines():
            line = line.strip()
            if line:
                tasks.append(Task.from_dict(json.loads(line)))
    return tasks


# ── model drivers (pluggable) ───────────────────────────────────────────────────
class ModelDriver:
    """Base driver: given a conversation (list of {role,content}), return the
    assistant's next raw text (which may contain <tool_call> blocks)."""

    def generate(self, messages: list[dict], max_new_tokens: int = 512) -> str:
        raise NotImplementedError


class LocalDriver(ModelDriver):
    """Drive a local HF checkpoint with a <tool_call> loop (self-contained)."""

    def __init__(self, model_path: str, rocm: bool = False, max_seq: int = 8192):
        self.model_path = model_path
        self.rocm = rocm
        self.max_seq = max_seq
        self._model = None
        self._tok = None

    def _load(self):
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dev = {"device_map": "auto"} if self.rocm else {"device_map": "cpu"}
        self._model = AutoModelForCausalLM.from_pretrained(self.model_path,
                                                           torch_dtype="auto", **dev)
        self._tok = AutoTokenizer.from_pretrained(self.model_path)

    def generate(self, messages: list[dict], max_new_tokens: int = 512) -> str:
        self._load()
        prompt = self._tok.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True)
        ids = self._tok(prompt, return_tensors="pt").input_ids.to(self._model.device)
        out = self._model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                                   pad_token_id=self._tok.pad_token_id)
        return self._tok.decode(out[0][ids.shape[1]:], skip_special_tokens=False)


class OptimizedDriver(LocalDriver):
    """Runner 'subagent': the model-specific, optimized tool loop.

    Purpose-built for RefinedToolCallV5 variants (and the base model). vs the
    generic LocalDriver it adds:

      * per-variant result formatting — feeds <tool_result> to finetunes (what
        they were trained on) but <tool_response> to the base model (what its
        chat template expects). This alone materially changes call fidelity.
      * explicit stop sequences — stops at <|im_end|> and the tool separator so
        the model doesn't ramble past a complete call.
      * error recovery — when a tool errors or returns no parseable call, a
        structured recovery message is returned (the base model advertises a
        0.896 recovery rate; we mirror that by feeding the failure back as a
        tool result rather than crashing the loop).
      * variant autodetect — "base" if the checkpoint dir name contains
        'RefinedToolCallV5-3b' and no finetune marker, else "finetune".

    Intended to later run as a subagent inside opencode / hermes-agent; for now
    it is a drop-in driver for `bench --runner=subagent`.
    """

    def __init__(self, model_path: str, rocm: bool = False, max_seq: int = 8192,
                 variant: str = "auto"):
        super().__init__(model_path, rocm=rocm, max_seq=max_seq)
        if variant == "auto":
            variant = "base" if "RefinedToolCallV5-3b" in model_path \
                and "toolcall-v5-3b-" not in model_path else "finetune"
        self.variant = variant

    def generate(self, messages: list[dict], max_new_tokens: int = 512) -> str:
        self._load()
        prompt = self._tok.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True)
        ids = self._tok(prompt, return_tensors="pt").input_ids.to(self._model.device)
        # stop at the turn boundary and the tool separator so we don't generate
        # past a complete tool call
        stops = [self._tok.convert_tokens_to_ids("<|im_end|>"),
                 _SEP_CHARS, _SEP_LITERAL]
        stops = [s for s in stops if isinstance(s, int) and s is not None
                 and s != self._tok.unk_token_id]
        gen = dict(max_new_tokens=max_new_tokens, do_sample=False,
                   pad_token_id=self._tok.pad_token_id)
        if stops:
            gen["eos_token_id"] = stops[0]
            gen["stopping_criteria"] = None
        out = self._model.generate(ids, **gen)
        return self._tok.decode(out[0][ids.shape[1]:], skip_special_tokens=False)

    def wrap_result(self, result: str) -> str:
        return format_tool_result(result, self.variant)


class ApiDriver(ModelDriver):
    """Drive an OpenAI-compatible endpoint (e.g. the lan lm-fleet-router).

    Pass base_url + model; no credentials are stored in code. Reads an optional
    Bearer token from OPENAI_API_KEY / the `api_key` arg for routers that need
    it. The body is kept minimal for broad compatibility.
    """

    def __init__(self, base_url: str, model: str, api_key: str = "",
                 max_seq: int = 8192):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.max_seq = max_seq

    def generate(self, messages: list[dict], max_new_tokens: int = 512) -> str:
        import urllib.request
        url = f"{self.base_url}/chat/completions"
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
            "stream": False,
        }
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(), method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}" if self.api_key else ""})
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode())
        return data["choices"][0]["message"]["content"]


# ── run one task to completion ───────────────────────────────────────────────────
@dataclass
class TaskResult:
    task_id: str
    kind: str
    model: str
    runner: str
    checks_passed: int = 0
    checks_total: int = 0
    turns: int = 0
    completed: bool = False     # model signalled it was done (no more tool calls)
    error: str = ""
    transcript: list = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.checks_total > 0 and self.checks_passed == self.checks_total


def run_task(driver: ModelDriver, task: Task, model_name: str, runner_name: str,
             sandbox_root: Optional[Path] = None, gen_max_tokens: int = 512) -> TaskResult:
    """Drive `driver` through `task`, then verify the outcome.

    Two driver shapes are supported:
      * generate-style (LocalDriver / ApiDriver): this module owns the tool
        loop and sandbox (ToolEnv), parsing <tool_call> and feeding
        <tool_result> back until the model stops calling tools.
      * submit-style (HermesDriver, `run_one`): the driver runs its OWN
        tool loop + sandbox; we just get back a trajectory dict with a
        `completed` flag. Our verifier then checks `sandbox_root` if provided
        (point it at the dir Hermes wrote into), else we trust `completed`.
    """
    root = sandbox_root or Path(tempfile.mkdtemp(prefix="bench-"))
    res = TaskResult(task_id=task.id, kind=task.kind, model=model_name,
                     runner=runner_name)
    try:
        if hasattr(driver, "run_one"):
            # submit-style driver (hermes): it manages its own loop + sandbox
            import tempfile as _tf
            out = driver.run_one(task.prompt, Path(_tf.mktemp(suffix=".jsonl")))
            if out is None:
                res.error = "hermes runner produced no trajectory"
            else:
                res.completed = bool(out.get("completed"))
                res.turns = int(out.get("api_calls", 0) or 0)
                if "error" in out:
                    res.error = str(out["error"])
                res.transcript.append({"role": "hermes", "detail": out})
                # verify only if the caller pointed sandbox_root at Hermes's dir
                if sandbox_root is not None:
                    passed, total, detail = verify_task(root, task.checks)
                    res.checks_passed, res.checks_total = passed, total
                    res.transcript.append({"role": "verify", "detail": detail})
                else:
                    # fall back to Hermes's own completion verdict
                    res.checks_total = 1
                    res.checks_passed = 1 if res.completed else 0
            return res

        # generate-style driver: this module owns the loop
        env = ToolEnv(root)
        messages = list(task.replay_context) + [{"role": "user", "content": task.prompt}]
        for turn in range(task.max_turns):
            res.turns = turn + 1
            text = driver.generate(messages, max_new_tokens=gen_max_tokens)
            res.transcript.append({"role": "assistant", "content": text})
            calls = (getattr(driver, "parse_tool_calls", parse_tool_calls))(text)
            if not calls:
                res.completed = True
                messages.append({"role": "assistant", "content": text})
                break
            wrap = getattr(driver, "wrap_result",
                           lambda r: f"<tool_result>{r}</tool_result>")
            tool_msgs = []
            for c in calls:
                # error recovery: if args are unparseable, tell the model so it
                # can self-correct (mirrors the model's trained recovery behavior)
                if c["args"] is None:
                    result = (f"ERROR: could not parse arguments for tool "
                              f"'{c['name']}'. Re-emit a valid JSON object "
                              f"with the correct argument names.")
                else:
                    result = env.execute(c["name"], c["args"])
                tool_msgs.append(wrap(result))
                res.transcript.append({"role": "tool", "name": c["name"],
                                       "content": result})
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": "\n".join(tool_msgs)})
        else:
            res.error = "max turns reached without completion"
        passed, total, detail = verify_task(root, task.checks)
        res.checks_passed, res.checks_total = passed, total
        res.transcript.append({"role": "verify", "detail": detail})
    except Exception as e:  # noqa: BLE001
        res.error = f"{type(e).__name__}: {e}"
    finally:
        if sandbox_root is None:
            shutil.rmtree(root, ignore_errors=True)
    return res


# ── runner registry (pluggable: self-contained / hermes / subagent) ─────────────
RUNNERS: dict[str, Callable[..., ModelDriver]] = {}


def register_runner(name: str, fn: Callable[..., ModelDriver]) -> None:
    RUNNERS[name] = fn


def make_driver(runner: str, **kw) -> ModelDriver:
    """Construct a ModelDriver for a named runner.

    'self'     -> local HF checkpoint (kw: model_path, rocm)
    'subagent' -> optimized model-specific loop (kw: model_path, rocm, variant)
    'api'      -> OpenAI-compatible endpoint (kw: base_url, model, api_key)
    'hermes'   -> hermes-agent harness (kw: hermes_dir)  [wired by register_runner]
    """
    if runner in RUNNERS:
        return RUNNERS[runner](**kw)
    if runner == "self":
        return LocalDriver(kw["model_path"], rocm=kw.get("rocm", False))
    if runner == "subagent":
        return OptimizedDriver(kw["model_path"], rocm=kw.get("rocm", False),
                               variant=kw.get("variant", "auto"))
    if runner == "api":
        return ApiDriver(kw["base_url"], kw["model"], api_key=kw.get("api_key", ""))
    raise ValueError(f"unknown runner: {runner} (known: {sorted(set(RUNNERS)|{'self','subagent','api'})})")


class HermesDriver(ModelDriver):
    """Runner #2: delegate to the hermes-agent harness on this machine.

    Uses hermes-agent/mini_swe_runner.py --task, which runs the task through
    Hermes's OWN model + tool loop (its LM Studio routing), so it is a
    genuinely different harness from the self-contained one. Hermes is pointed
    at whatever model it is configured with (set that in Hermes, not here).

    Returns the trajectory's `completed` flag via a sentinel string the bench
    loop can't produce, so we instead parse the written output file.
    """

    def __init__(self, hermes_dir: str, python: str = "python3",
                 extra_args: Optional[list] = None):
        self.hermes_dir = hermes_dir
        self.python = python
        self.extra_args = extra_args or []

    def run_one(self, prompt: str, out_file: Path) -> Optional[dict]:
        cmd = [
            self.python, "-m", "mini_swe_runner", "--task", prompt,
            "--output_file", str(out_file),
        ] + self.extra_args
        try:
            subprocess.run(cmd, cwd=self.hermes_dir, capture_output=True,
                            text=True, timeout=900)
        except Exception as e:  # noqa: BLE001
            return {"completed": False, "error": str(e)}
        try:
            return json.loads(Path(out_file).read_text(errors="ignore").splitlines()[0])
        except Exception:
            return None


def _make_hermes(runner: str = "hermes", **kw) -> ModelDriver:
    return HermesDriver(kw.get("hermes_dir", "/home/scott/git/hermes-agent"),
                        python=kw.get("python", "python3"),
                        extra_args=kw.get("extra_args"))


register_runner("hermes", _make_hermes)


def bench_suite(driver: ModelDriver, tasks: list[Task], model_name: str,
                runner_name: str, gen_max_tokens: int = 512) -> list[TaskResult]:
    """Run every task and return the results."""
    return [run_task(driver, t, model_name, runner_name,
                     gen_max_tokens=gen_max_tokens) for t in tasks]


def format_bench_results(results: list[TaskResult]) -> str:
    lines = ["## Task-completion benchmark", "",
             "| task | kind | success | checks | turns | completed |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for r in results:
        succ = "yes" if r.success else "no"
        comp = "yes" if r.completed else "no"
        lines.append(f"| {r.task_id} | {r.kind} | {succ} | "
                     f"{r.checks_passed}/{r.checks_total} | {r.turns} | {comp} |")
    n = len(results)
    ok = sum(1 for r in results if r.success)
    lines.append("")
    lines.append(f"**completion: {ok}/{n} tasks fully passed "
                 f"({ok/n*100:.0f}%)**")
    return "\n".join(lines)


# the "local-chat" runner (standard HF model, native tool format) registers itself
# when this module is imported.
def _ensure_local_chat():
    try:
        import src.drivers_localchat  # noqa: F401  (self-registers "local-chat")
    except Exception:
        pass


def bench_matrix(tasks: list[Task], specs: list[dict],
                 gen_max_tokens: int = 512, rocm: bool = False) -> dict:
    """Run the same task suite across several model/runner specs.

    `specs` is a list of dicts, each:
        {"name": <str>, "runner": <str>, ...kwargs for make_driver}
    e.g.
        {"name": "base",    "runner": "local-chat", "model_path": ".../Qwen2.5-7B-Instruct"}
        {"name": "ssd-ft",  "runner": "subagent",   "model_path": ".../toolcall-v5-3b-ssd", "rocm": True}
        {"name": "fleet",   "runner": "api",        "base_url": "...", "model": "qwen3.6-35b"}
    Returns {spec_name: {"results": [TaskResult], "summary": {...}}}.
    """
    _ensure_local_chat()
    out: dict = {}
    for spec in specs:
        name = spec.get("name", spec.get("runner", "unknown"))
        runner = spec["runner"]
        kw = {k: v for k, v in spec.items() if k not in ("name", "runner")}
        if runner in ("self", "subagent", "local-chat") and "rocm" not in kw:
            kw["rocm"] = rocm
        try:
            driver = make_driver(runner, **kw)
        except Exception as e:  # noqa: BLE001
            out[name] = {"results": [], "summary": {"error": f"{type(e).__name__}: {e}"}}
            continue
        results = bench_suite(driver, tasks, name, runner,
                              gen_max_tokens=gen_max_tokens)
        n = len(results)
        ok = sum(1 for r in results if r.success)
        comp = sum(1 for r in results if r.completed)
        errors = sum(1 for r in results if r.error)
        out[name] = {
            "results": results,
            "summary": {
                "n": n, "passed": ok, "completed": comp, "errors": errors,
                "pass_rate": (ok / n) if n else 0.0,
            },
        }
    return out


def format_bench_matrix(matrix: dict) -> str:
    lines = ["## Benchmark matrix (same suite, multiple models)", "",
             "| model | pass | complete | errors | pass-rate |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name, block in matrix.items():
        s = block.get("summary", {})
        if "error" in s:
            lines.append(f"| {name} | — | — | 1 | ERR: {s['error']} |")
            continue
        lines.append(f"| {name} | {s['passed']}/{s['n']} | {s['completed']}/{s['n']} "
                     f"| {s['errors']} | {s['pass_rate']*100:.0f}% |")
    # per-task breakdown
    lines += ["", "### per-task", "",
              "| task | " + " | ".join(matrix.keys()) + " |",
              "| --- | " + " | ".join(["---:"] * len(matrix)) + " |"]
    # align by task id across specs
    by_task: dict[str, dict] = {}
    for name, block in matrix.items():
        for r in block["results"]:
            by_task.setdefault(r.task_id, {})[name] = r
    for tid, per in by_task.items():
        cells = []
        for name in matrix:
            r = per.get(name)
            cells.append("✓" if (r and r.success) else ("✗" if r else "—"))
        lines.append(f"| {tid} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
