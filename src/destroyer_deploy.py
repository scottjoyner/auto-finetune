#!/usr/bin/env python3
"""Destroyer node continuous deployment manager.

Manages deployment of the continuous inference hub on destroyer node (100.81.57.77).
Handles the constraint that SSH key auth is not pre-configured.

Usage:
  python src/destroyer_deploy.py --check           # check connectivity
  python src/destroyer_deploy.py --prepare         # prepare deploy package
  python src/destroyer_deploy.py --deploy          # attempt deploy via SSH
  python src/destroyer_deploy.py --status          # check deployment status
"""

import subprocess, json, socket, time
from pathlib import Path
from datetime import datetime

NODE_IP = "100.81.57.77"
NODE_NAME = "destroyer"
SSH_KEY = "~/.ssh/id_ed25519"
DEPLOY_DIR = "/tmp/destroyer_deploy"

# Deploy package files
DEPLOY_FILES = [
    f"{DEPLOY_DIR}/setup_destroyer_agent_hub.sh",
    f"{DEPLOY_DIR}/deploy_destroyer.sh",
    f"{DEPLOY_DIR}/subagent_2b_agent.py",
]

# Node specifications
NODE_SPEC = {
    "name": NODE_NAME,
    "ip": NODE_IP,
    "cpu": "i7-10610U (4C/8T)",
    "ram": "31GB",
    "gpu": "None (integrated graphics only)",
    "os": "Ubuntu (assumed)",
    "inference_mode": "CPU-only",
    "expected_tok_s": "15-25 tok/s (CPU, Q4_KM)",
    "primary_model": "MiniCPM5-1B-Q4_K_M.gguf (657MB)",
    "server_port": 8300,
    "ssh_status": "key_auth_blocked",
    "deployment_status": "pending_manual",
}


def check_connectivity():
    """Check if node is reachable via ping and tailscale ssh."""
    results = {"ping": False, "tailscale": False, "ssh": False, "port": False}
    try:
        r = subprocess.run(["ping", "-c", 2, "-W", "2", NODE_IP],
                           capture_output=True, timeout=5)
        results["ping"] = r.returncode == 0
    except: pass
    try:
        r = subprocess.run(["tailscale", "ping", "-c", NODE_IP],
                           capture_output=True, timeout=5)
        results["tailscale"] = r.returncode == 0
    except: pass
    try:
        r = subprocess.run(["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
                            "-o", "StrictHostKeyChecking=no", "-i", SSH_KEY,
                            f"scott@{NODE_IP}", "hostname"],
                           capture_output=True, timeout=7)
        results["ssh"] = r.returncode == 0
    except: pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(3)
        s.connect((NODE_IP, 8300))
        results["port"] = True
        s.close()
    except: pass
    return results


def prepare_deploy_package():
    """Prepare deploy scripts for manual deployment."""
    Path(DEPLOY_DIR).mkdir(parents=True, exist_ok=True)
    # Ensure deploy scripts exist
    scripts = []
    for f in DEPLOY_FILES:
        if Path(f).exists():
            scripts.append(f)
    return {"deploy_dir": DEPLOY_DIR, "scripts_ready": len(scripts), "files": scripts}


def deploy_status():
    """Check deployment status of destroyer node."""
    connectivity = check_connectivity()
    deploy_package = prepare_deploy_package()

    return {
        "node": NODE_SPEC,
        "connectivity": connectivity,
        "deploy_package": deploy_package,
        "deployment_status": "pending_manual" if not connectivity["ssh"] else "ready",
        "timestamp": datetime.now().isoformat(),
        "next_steps": [
            "1. Ensure SSH key authorized on destroyer (ssh-copy-id)",
            "2. Run deploy_destroyer.sh script on destroyer",
            "3. Verify server on port 8300",
            "4. Add to fleet for continuous inference cycle",
        ] if not connectivity["ssh"] else [
            "1. Run: ssh scott@{ip} 'bash -s' < setup_destroyer_agent_hub.sh",
            "2. Verify inference: curl http://{ip}:8300/v1/chat/completions",
        ],
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true", help="Check connectivity")
    p.add_argument("--prepare", action="store_true", help="Prepare deploy package")
    p.add_argument("--deploy", action="store_true", help="Attempt deploy")
    p.add_argument("--status", action="store_true", help="Check deployment status")
    args = p.parse_args()

    if args.check or args.status:
        result = deploy_status()
        print(json.dumps(result, indent=2))
    elif args.prepare:
        result = prepare_deploy_package()
        print(json.dumps(result, indent=2))
    elif args.deploy:
        print("Deploy requires manual SSH access. Use --status to check readiness.")
        status = deploy_status()
        print(json.dumps(status["next_steps"], indent=2))
