#!/usr/bin/env python3
"""Continuous agent harness v3 for MiniCPM5 2B (local reasoning_content).

Handles continuously:
  - XML: <function name="bash"><param name="command">cmd</param></function>
  - XML+CDATA: <...>cmd<![CDATA[...]>
  - bash: cmd format
  - Code blocks: ```bash cmd ``` (also ``` cmd ``` without bash)
  - Inline commands: $ cmd
"""

import json, re, time, subprocess, urllib.request


def call_2b(messages, model="minicpm5-2b", max_tokens=512, base="http://127.0.0.1:38899/v1/chat/completions"):
    payload = json.dumps({"model": model, "messages": messages,
                          "max_tokens": max_tokens, "temperature": 0.0, "stream": False})
    req = urllib.request.Request(base, data=payload.encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
        elapsed = time.time() - start
        msg = data["choices"][0]["message"]
        content = msg.get("content", "")
        reasoning = msg.get("reasoning_content", "")
        combined = content if content else reasoning
        usage = data.get("usage", {})
        return {"content": combined, "elapsed": elapsed,
                "finish": data["choices"][0].get("finish_reason", ""),
                "tok_s": usage.get("completion_tokens", 0) / elapsed if elapsed > 0 else 0}
    except Exception as e:
        return {"content": f"[ERROR] {e}", "elapsed": 5.0, "finish": "error", "tok_s": 0, "error": str(e)}


def extract_bash(text):
    """Extract bash commands from ALL 2B formats continuously."""
    calls = []
    # Format 1: XML with optional CDATA
    for m in re.finditer(
        r'<function\s+name="bash"><param\s+name="command">\s*(?:<!\[CDATA\[)?(.+?)(?:\]\]>)?\s*</param>',
        text, re.DOTALL):
        calls.append(m.group(1).strip())

    # Format 2: <bash>command="cmd"</bash> format
    for m in re.finditer(r'<bash>\s*command="([^"]+)"', text, re.DOTALL):
        calls.append(m.group(1).strip())
    for m in re.finditer(r"<bash>\s*command='([^']+)'", text, re.DOTALL):
        calls.append(m.group(1).strip())
    # Format 3: bash: cmd (simple)
    if not calls:
        for m in re.finditer(r'(?:^|\n)bash:\s*(.+?)(?=\n(?:result|Result|Output|thinking|bash|Bash)|$)',
                             text, re.DOTALL):
            cmd = m.group(1).strip()
            if cmd and not cmd.startswith(("<actual>", "result")):
                calls.append(cmd)

    # Format 3: ```bash\nCMD\n``` or ```\nCMD\n``` with bash keywords
    if not calls:
        for m in re.finditer(r'```(?:bash|sh|shell)\n(.+?)```', text, re.DOTALL):
            calls.append(m.group(1).strip())
        # Generic ``` blocks that look like commands
        if not calls:
            for m in re.finditer(r'```\n(.+?)```', text, re.DOTALL):
                cmd = m.group(1).strip()
                if any(s in cmd.split('\n')[0] for s in
                       ['find', 'ls', 'echo', 'cat', 'wc', 'grep', 'tar', 'df', 'ps',
                        'mkdir', 'chmod', 'gzip', 'rm', 'cp', 'mv', 'whoami', 'pwd', 'free']):
                    calls.append(cmd)

    # Filter out placeholder/example commands
    filtered = [c for c in calls if not c.startswith(('echo "Run', 'echo "Output',
                                                       'COMMAND', 'actual_command',
                                                       '<actual', 'EXAMPLE'))]
    return filtered


def run_2b_agent(task, model="minicpm5-2b", max_steps=4):
    """Continuous agent loop with best working prompt."""
    print(f"[AGENT-2B] Task: {task[:70]}")
    messages = [
        {"role": "system",
         "content": ("You are a bash agent. Emit ONLY tool calls:\n"
                     "<function name=\"bash\"><param name=\"command\">COMMAND</param></function>\n"
                     "If you need to show code, put it in the command param.\n"
                     "Do not explain, just act.")},
        {"role": "user", "content": task},
    ]
    results = []
    for step in range(1, max_steps + 1):
        r = call_2b(messages, model=model)
        tools = extract_bash(r["content"])
        print(f"[AGENT-2B] Step {step}: {len(r['content'])} chars, "
              f"{r['tok_s']:.1f} tok/s, tools={len(tools)}, finish={r['finish']}")
        results.append({"step": step, "time": r["elapsed"], "tok_s": r["tok_s"],
                        "tools": len(tools), "finish": r["finish"]})
        if not tools:
            print(f"[AGENT-2B] Final: {r['content'][:100]}")
            break
        for cmd in tools[:3]:  # limit to 3 commands per step
            print(f"  Exec: {cmd[:80]}")
            try:
                p = subprocess.run(cmd, shell=True, capture_output=True,
                                   text=True, timeout=15)
                out = p.stdout + p.stderr
                print(f"  Out: {out[:80]}")
                messages.append({"role": "assistant", "content": r["content"]})
                messages.append({"role": "user",
                                 "content": f"Result of `{cmd}`:\n{out}\nContinue."})
            except subprocess.TimeoutExpired:
                print(f"  Timeout")
                # Try truncated version
                simple = cmd.split('&&')[0].split('|')[0].strip()
                try:
                    p = subprocess.run(simple, shell=True, capture_output=True,
                                       text=True, timeout=10)
                    out = p.stdout + p.stderr
                    print(f"  Simplified: {out[:80]}")
                    messages.append({"role": "assistant", "content": r["content"]})
                    messages.append({"role": "user", "content": f"Result:\n{out}\nContinue."})
                except:
                    pass
            except Exception as e:
                print(f"  Error: {str(e)[:60]}")
            time.sleep(0.1)
        if r["finish"] in ("stop", "error"):
            break
    avg = sum(r["tok_s"] for r in results) / len(results) if results else 0
    total = sum(r["time"] for r in results)
    return {"avg_tok_s": avg, "total_time": total, "steps": len(results),
            "success": any(r.get("tools", 0) > 0 for r in results), "results": results}


def extract_function(text, name="compress"):
    """Extract bash function with balanced braces."""
    start = text.find(f"{name}()")
    if start < 0:
        return None
    brace_count, end = 0, start
    for i, c in enumerate(text[start:], start):
        if c == '{': brace_count += 1
        elif c == '}':
            brace_count -= 1
            if brace_count == 0:
                end = i + 1
                break
    return text[start:end]


if __name__ == "__main__":
    import argparse, os
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="echo continuous_2b > /tmp/2b_final.txt")
    p.add_argument("--model", default="minicpm5-2b")
    args = p.parse_args()
    r = run_2b_agent(args.task, args.model)
    print(f"\n=== RESULT === steps={r['steps']} time={r['total_time']:.2f}s "
          f"tok/s={r['avg_tok_s']:.1f} success={r['success']}")
    if os.path.exists("/tmp/2b_final.txt"):
        print(f"Verify: {open('/tmp/2b_final.txt').read().strip()}")
