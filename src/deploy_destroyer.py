#!/usr/bin/env python3
"""Deploy and test MiniCPM5-1B on destroyer node continuously."""
import json, urllib.request, time, subprocess, sys, os

sys.path.insert(0, "/home/scott/git/auto-finetune")
from src.config import scratch_dir
from src.locking import atomic_write_json

DESTROYER_IP = "100.81.57.77"
# Local artifact: honour TMPDIR / the data mount rather than /tmp, which is a
# separate filesystem cleared on reboot. (The server log inside the ssh command
# stays on the remote node's own /tmp -- that path belongs to the destroyer.)
RESULTS_PATH = os.path.join(scratch_dir(), "destroyer_deploy_results.json")
PASSWORD = "gluhlaf8"
SERVER_URL = f"http://{DESTROYER_IP}:8300/v1/chat/completions"
LOCAL_URL = "http://127.0.0.1:38899/v1/chat/completions"

def ssh_run(cmd, password=PASSWORD):
    """Run command on destroyer with password auth."""
    full_cmd = f"sshpass -p '{password}' ssh -o StrictHostKeyChecking=no scott@{DESTROYER_IP} '{cmd}'"
    r = subprocess.run(full_cmd, shell=True, capture_output=True, text=True, timeout=60)
    return r.stdout + r.stderr

def start_server():
    """Start MiniCPM5 server on destroyer port 8300."""
    print("=== Starting MiniCPM5-1B on destroyer (port 8300) ===")
    cmd = f'''
    LIB_DIR="/home/scott/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-avx2-2.33.0"
    export LD_LIBRARY_PATH="$LIB_DIR:/home/scott/fleet-data/lib:$LD_LIBRARY_PATH"
    pkill -f "port 8300" 2>/dev/null
    sleep 1
    /home/scott/fleet-data/bin/llama-server \\
      --model /home/scott/.lmstudio/models/openbmb/MiniCPM5-1B-GGUF/MiniCPM5-1B-Q4_K_M.gguf \\
      --host 0.0.0.0 --port 8300 --threads 8 -ngl 0 --ctx-size 2048 \\
      > /tmp/minicpm5_8300.log 2>&1 &
    echo $!
    sleep 6
    echo "Health check:"
    curl -s "http://localhost:8300/health" | head -c 100
    echo ""
    echo "Server status:"
    ps aux | grep "port 8300" | grep -v grep | head -1
    '''
    result = ssh_run(cmd)
    print(result)
    return result

def test_inference(url, label):
    """Test inference on a given URL."""
    print(f"\n=== Inferencing: {label} ({url}) ===")
    payload = json.dumps({
        "model": "minicpm5-1b",
        "messages": [{"role": "user", "content": "Write a bash function called 'compress' that gzips $1"}],
        "max_tokens": 256,
        "temperature": 0.0
    })
    req = urllib.request.Request(url, data=payload.encode(),
                                  headers={"Content-Type": "application/json"})
    start = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        data = json.loads(resp.read().decode())
        elapsed = time.time() - start
        msg = data.get("choices", [{}])[0].get("message", {})
        content = msg.get("content", "")
        reasoning = msg.get("reasoning_content", "")
        combined = content if content else reasoning
        usage = data.get("usage", {})
        gen_tok = usage.get("completion_tokens", 0)
        tok_s = gen_tok / elapsed if elapsed > 0 else 0
        print(f"✅ {label}: {len(combined)} chars | {elapsed:.2f}s | {tok_s:.1f} tok/s | gen={gen_tok}tok")
        return {"success": True, "content_len": len(combined), "tok_s": tok_s,
                "reasoning_len": len(reasoning), "gen_tok": gen_tok}
    except Exception as e:
        elapsed = time.time() - start
        print(f"❌ {label}: ERROR {str(e)[:80]} ({elapsed:.1f}s)")
        return {"success": False, "error": str(e), "tok_s": 0}

if __name__ == "__main__":
    # Start destroyer server
    start_server()
    time.sleep(3)
    
    # Test locally
    local_res = test_inference(LOCAL_URL, "localhost (xwing GPU)")
    
    # Test destroyer
    destroyer_res = test_inference(SERVER_URL, "destroyer (100.81.57.77:8300 CPU)")
    
    # Compare
    print("\n=== COMPARISON ===")
    print(f"{'Endpoint':<25} {'Tok/s':>8} {'Success':>8} {'Content':>8}")
    print("-" * 50)
    if local_res["success"]:
        print(f"{'localhost (xwing)':<25} {local_res['tok_s']:>8.1f} {'✅':>8} {local_res['content_len']:>8}")
    if destroyer_res["success"]:
        print(f"{'destroyer (port 8300)':<25} {destroyer_res['tok_s']:>8.1f} {'✅':>8} {destroyer_res['content_len']:>8}")
    
    # Save results
    results = {
        "localhost": local_res,
        "destroyer": destroyer_res,
        "comparison": {
            "tok_s_ratio": destroyer_res.get("tok_s", 0) / local_res.get("tok_s", 1) if local_res.get("tok_s", 0) else 0,
            "destroyer_status": "DEPLOYED" if destroyer_res["success"] else "FAILED",
        }
    }
    atomic_write_json(RESULTS_PATH, results)
    print(f"\nResults saved to {RESULTS_PATH}")
