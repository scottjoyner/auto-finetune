#!/usr/bin/env python3
"""Continuous agent harness v5 for MiniCPM5 2B (local reasoning_content).

Strategy: Strict tool-call format first; if model generates prose,
fallback to parsing commands from inline/backtick/code blocks.
"""
import json, re, os, time, subprocess, urllib.request


def call_2b(messages, model="minicpm5-2b", max_tokens=512,
            base="http://127.0.0.1:38899/v1/chat/completions"):
    payload = json.dumps({
        "model": model, "messages": messages,
        "max_tokens": max_tokens, "temperature": 0.0, "stream": False
    })
    req = urllib.request.Request(base, data=payload.encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
        elapsed = time.time() - start
        msg = data["choices"][0]["message"]
        content = msg.get("content", "") or msg.get("reasoning_content", "")
        usage = data.get("usage", {})
        return {"content": content, "elapsed": elapsed,
                "finish": data["choices"][0].get("finish_reason", ""),
                "tok_s": usage.get("completion_tokens", 0) / elapsed if elapsed > 0 else 0}
    except Exception as e:
        return {"content": f"[ERROR] {e}", "elapsed": 5.0,
                "finish": "error", "tok_s": 0}


def extract_bash(text):
    """Extract bash commands from XML tool call format (strict)."""
    calls = []
    for pattern in [
        r'<function\s+name="bash"><param\s+name="command">\s*(?:<!\[CDATA\[)?(.+?)(?:\]\]>)?\s*</param>',
        r'<bash>\s*command="([^"]+)"',
        r"<bash>\s*command='([^']+)'",
        r'(?:^|\n)bash:\s*(.+?)(?=\n(?:result|Result|Output|thinking|bash|Bash)|$)',
    ]:
        calls = re.findall(pattern, text, re.DOTALL)
        if calls:
            break
    return [c.strip() for c in calls if c.strip()]


def extract_fallback(text):
    """Extract command from prose (code block, backtick, inline)."""
    # Code block ```bash or ```
    for m in re.finditer(r'```(?:bash|sh)?\n(.+?)```', text, re.DOTALL):
        cmd = m.group(1).strip()
        if any(kw in cmd.split('\n')[0].lower() for kw in
               ['find', 'ls', 'echo', 'cat', 'wc', 'grep', 'tar', 'df', 'ps',
                'mkdir', 'chmod', 'gzip', 'rm', 'cp', 'mv', 'whoami', 'pwd', 'free']):
            return cmd

    # Inline commands like `echo x` or `$ ls`
    for m in re.finditer(r'`([^`]+)`', text):
        cmd = m.group(1).strip()
        if any(kw in cmd.lower() for kw in ['echo', 'ls', 'find', 'wc', 'cat', 'tar', 'df', 'ps']):
            return cmd

    # Inline $ commands
    for m in re.finditer(r'\$\s*((?:find|ls|cat|echo|tar|wc|grep|df|ps|free|nproc|whoami)[^$\n]+)', text):
        return m.group(1).strip()

    return None


def extract_function(text):
    """Extract complete bash function with balanced braces."""
    for m in re.finditer(r'(\w+)\(\)\s*\{', text):
        name = m.group(1)
        start = m.start()
        count, end = 0, start
        for i in range(start, len(text)):
            if text[i] == '{': count += 1
            elif text[i] == '}':
                count -= 1
                if count == 0:
                    end = i + 1
                    break
        return name, text[start:end]
    return None, None


def run_2b_agent(task, model="minicpm5-2b", max_steps=5):
    """Continuous agent loop v5: strict tool-call with prose fallback."""
    print(f"[AGENT-2B-v5] {task[:70]}")
    messages = [
        {"role": "system",
         "content": ("You are a bash coding agent. Emit ONLY tool calls:\n"
                     "<function name=\"bash\"><param name=\"command\">COMMAND</param></function>\n"
                     "No thinking prefix, no prose before/after.")},
        {"role": "user", "content": task},
    ]
    results = []
    saved_functions = []

    for step in range(1, max_steps + 1):
        r = call_2b(messages, model=model)
        content = r["content"]
        tools = extract_bash(content)
        print(f"[AGENT-2B-v5] Step {step}: {len(content)}c, {r['tok_s']:.0f} tok/s, "
              f"tools={len(tools)}, finish={r['finish']}")
        results.append({"step": step, "time": r["elapsed"], "tok_s": r["tok_s"],
                        "tools": len(tools), "finish": r["finish"]})

        # Fallback: parse prose for commands
        if not tools:
            fallback = extract_fallback(content)
            func_name, func_code = extract_function(content)
            if func_code:
                path = f"/tmp/{func_name}.sh"
                with open(path, "w") as f:
                    f.write(func_code + "\n")
                os.chmod(path, 0o755)
                saved_functions.append({"name": func_name, "path": path})
                # Auto-test the function if it's a simple backup/compress
                test_cmd = f"bash -c '. {path} && echo loaded_{func_name}'"
                try:
                    p = subprocess.run(test_cmd, shell=True, capture_output=True,
                                       text=True, timeout=10)
                    print(f"  Func: {func_name} -> {path} | test: {p.stdout.strip()}")
                except:
                    pass
                tools = [{"cmd": test_cmd, "is_func": True}]
            elif fallback:
                print(f"  Fallback: parsed command from prose")
                tools = [{"cmd": fallback}]

        if not tools:
            print(f"[AGENT-2B-v5] Final: {content[:100]}")
            break

        for item in tools:
            cmd = item if isinstance(item, str) else item.get("cmd", "")
            print(f"  Exec: {cmd[:80]}")
            try:
                p = subprocess.run(cmd, shell=True, capture_output=True,
                                   text=True, timeout=15)
                out = p.stdout + p.stderr
                print(f"  Out: {out[:80]}")
                messages.append({"role": "assistant", "content": content})
                feedback = out[:200] if out else "No output"
                messages.append({"role": "user",
                                 "content": f"Result of `{cmd[:50]}`:\n{feedback}\nContinue task."})
            except subprocess.TimeoutExpired:
                # Try simplifying the command
                simple = cmd.split('&&')[0].split('|')[0].strip()
                print(f"  Timeout - trying: {simple}")
                try:
                    p = subprocess.run(simple, shell=True, capture_output=True,
                                       text=True, timeout=10)
                    out = p.stdout + p.stderr
                    print(f"  Simplified: {out[:60]}")
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": f"Result:\n{out[:200]}\nContinue."})
                except Exception as e2:
                    print(f"  Still failed: {e2}")
            except Exception as e:
                print(f"  Error: {str(e)[:60]}")
            time.sleep(0.1)

        if r["finish"] in ("stop", "error"):
            break

    avg = sum(r["tok_s"] for r in results) / len(results) if results else 0
    return {"avg_tok_s": avg, "total_time": sum(r["time"] for r in results),
            "steps": len(results),
            "success": any(r.get("tools", 0) > 0 for r in results),
            "functions_created": saved_functions, "results": results}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="echo v5_test > /tmp/v5.txt")
    p.add_argument("--model", default="minicpm5-2b")
    a = p.parse_args()
    r = run_2b_agent(a.task, a.model)
    print(f"\nRESULT: steps={r['steps']} time={r['total_time']:.2f}s "
          f"tok/s={r['avg_tok_s']:.0f} success={r['success']} "
          f"funcs={r['functions_created']}")
