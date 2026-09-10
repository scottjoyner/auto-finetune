#!/usr/bin/env python3
"""Continuous agent harness for MiniCPM5 2B (local reasoning_content).

Handles all 2B output formats continuously:
  - XML: <function name="bash"><param name="command">cmd</param></function>
  - XML+CDATA: <function name="bash"><param name="command"><![CDATA[cmd]]></param></function>
  - Simple: bash: cmd -> result: ...
"""

import json, re, time, subprocess, urllib.request


def call_2b(messages, model="minicpm5-2b", max_tokens=512,
            base="http://127.0.0.1:38899/v1/chat/completions"):
    payload = json.dumps({"model": model, "messages": messages,
                          "max_tokens": max_tokens, "temperature": 0.1, "stream": False})
    req = urllib.request.Request(base, data=payload.encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.time()
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode())
    elapsed = time.time() - start
    msg = data["choices"][0]["message"]
    content = msg.get("content", "")
    reasoning = msg.get("reasoning_content", "")
    combined = content if content else reasoning
    usage = data.get("usage", {})
    return {
        "content": combined,
        "elapsed": elapsed,
        "finish": data["choices"][0].get("finish_reason", ""),
        "tok_s": usage.get("completion_tokens", 0) / elapsed if elapsed > 0 else 0,
    }


def extract_bash(text):
    """Extract bash commands from XML+CDATA, XML, or simple bash: format."""
    calls = []
    # Format 1: XML with CDATA <function name="bash"><param name="command"><![CDATA[cmd]]></param>
    for m in re.finditer(
        r'<function\s+name="bash">.*?<param\s+name="command">\s*(?:<!\[CDATA\[)?(.+?)(?:\]\]>)?\s*</param>',
        text, re.DOTALL
    ):
        cmd = m.group(1).strip()
        if cmd:
            calls.append(cmd)
    # Format 2: simple bash: cmd
    if not calls:
        for m in re.finditer(r'(?:^|\n)bash:\s*(.+?)(?=\n(?:result|Result|thinking|bash|Bash)|$)', text, re.DOTALL):
            cmd = m.group(1).strip()
            if cmd and not cmd.startswith(("<actual>", "result")):
                calls.append(cmd)
    return calls


def run_2b_agent(task, model="minicpm5-2b", max_steps=4):
    """Continuous agent loop for 2B with strict tool-call format."""
    print(f"[AGENT-2B] {task[:70]}")
    messages = [
        {"role": "system", "content":
            "You are an agent. Use tools by emitting ONLY this format:\n"
            "<function name=\"bash\"><param name=\"command\">COMMAND</param></function>\n"
            "No thinking, no prose, only tool calls. After running, summarize results."},
        {"role": "user", "content": task},
    ]
    results = []
    for step in range(1, max_steps + 1):
        r = call_2b(messages, model=model)
        tools = extract_bash(r["content"])
        print(f"[AGENT-2B] Step {step}: {len(r['content'])} chars, {r['tok_s']:.1f} tok/s, tools={len(tools)}")
        results.append({"step": step, "time": r["elapsed"], "tok_s": r["tok_s"]})
        if not tools:
            print(f"[AGENT-2B] Final: {r['content'][:100]}")
            break
        for cmd in tools:
            print(f"  Exec: {cmd[:80]}")
            try:
                p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15)
                out = p.stdout + p.stderr
                print(f"  Out: {out[:80]}")
                messages.append({"role": "assistant", "content": r["content"]})
                messages.append({"role": "user", "content": f"Result of `{cmd}`:\n{out}\nContinue."})
            except Exception as e:
                print(f"  Error: {e}")
            time.sleep(0.2)
        if r["finish"] == "stop":
            break
    avg = sum(r["tok_s"] for r in results) / len(results)
    total = sum(r["time"] for r in results)
    return {"avg_tok_s": avg, "total_time": total, "steps": len(results), "results": results}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="echo continuous_2b > /tmp/2b_final.txt")
    p.add_argument("--model", default="minicpm5-2b")
    args = p.parse_args()
    r = run_2b_agent(args.task, args.model)
    print(f"\n=== RESULT === Steps={r['steps']} | Time={r['total_time']:.2f}s | Tok/s={r['avg_tok_s']:.1f}")
    # Verify continuous
    try:
        with open("/tmp/2b_final.txt") as f:
            print(f"Verification: {f.read().strip()}")
    except:
        print("Verify: file not written continuously")
