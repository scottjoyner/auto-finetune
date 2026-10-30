#!/usr/bin/env python3
"""Execute bounded LFM3 fleet probes and retain raw, independently reviewable receipts."""
import argparse, datetime, hashlib, json, pathlib, platform, shlex, subprocess, sys, time
ROOT = pathlib.Path(__file__).resolve().parent
EXPECTED = {
 "LFM3-1.7B-Q4_K_M.gguf": "3793e1e6e2266919d6e015287221b41100175a98eddaba92e111aed3b05c93c4",
 "LFM3-1.7B-QAD-Q4_0.gguf": "43a7d09a0e9cba91b63eb7052488b9564569bddfd65754d2e426465e572a7452",
}
NODES = {
 "xwing": ("/home/scott/llama.cpp-prism/build-vulkan-gfx1151-20261016/bin/llama-bench", "LFM3-1.7B-QAD-Q4_0.gguf"),
 "deathstar-xps-8920": ("/home/scott/lfm3-research/build-vulkan/bin/llama-bench", "LFM3-1.7B-Q4_K_M.gguf"),
 "joyner": ("/home/scott/lfm3-research/runtime-vulkan/bin/llama-bench", "LFM3-1.7B-Q4_K_M.gguf"),
}
def ssh(node, command, timeout):
 return subprocess.run(["ssh","-o","BatchMode=yes","-o","ConnectTimeout=6","-o","StrictHostKeyChecking=yes",node,command],
     capture_output=True,text=True,timeout=timeout)
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--node",required=True,choices=NODES)
 p.add_argument("--prompt",type=int,default=512)
 p.add_argument("--binary",help="override existing runtime executable")
 p.add_argument("--model-name",choices=EXPECTED,help="override installed model checkpoint")
 p.add_argument("--gpu-layers",type=int,default=99)
 p.add_argument("--decode",type=int,default=64)
 p.add_argument("--repetitions",type=int,default=3)
 p.add_argument("--min-available-mib",type=int,default=2048)
 p.add_argument("--timeout",type=int,default=240)
 p.add_argument("--paired",action="store_true")
 p.add_argument("--cache",choices=["f16","q8_0"],default="f16")
 a=p.parse_args()
 if not(1<=a.prompt<=4096 and 1<=a.decode<=256 and 1<=a.repetitions<=5):p.error("unsafe benchmark bounds")
 binary,filename=NODES[a.node]
 if a.binary:binary=a.binary
 if a.model_name:filename=a.model_name
 model="/home/scott/lfm3-research/gguf/"+filename
 ts=datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
 receipt={"schema":"lfm3.fleet-benchmark.v1","created_utc":ts,"orchestrator":platform.node(),"node":a.node,
          "model":model,"expected_model_sha256":EXPECTED[filename],"prompt":a.prompt,"decode":a.decode,
          "repetitions":a.repetitions,"gpu_layers":a.gpu_layers,"mode":"paired" if a.paired else "separate","cache_type":a.cache,"status":"not_run"}
 path=ROOT/("fleet-"+a.node+"-"+ts+".json")
 try:
  checks=ssh(a.node,"hostname; awk '/MemAvailable:/ {print int($2/1024)}' /proc/meminfo; sha256sum "+shlex.quote(model),30)
  receipt["preflight"]={"returncode":checks.returncode,"stdout":checks.stdout[-3000:],"stderr":checks.stderr[-1500:]}
  lines=checks.stdout.strip().splitlines()
  if checks.returncode or len(lines)<3:raise RuntimeError("remote preflight failed")
  available=int(lines[1]); observed=lines[2].split()[0]
  receipt["available_mib"]=available;receipt["observed_model_sha256"]=observed
  if available<a.min_available_mib:raise RuntimeError("remote available RAM below safety limit")
  if observed!=EXPECTED[filename]:raise RuntimeError("model SHA-256 mismatch")
  cmd=["env","LD_LIBRARY_PATH="+str(pathlib.Path(binary).parent),binary,"-m",model,"-ngl",str(a.gpu_layers),"-t","4","-ctk",a.cache,"-ctv",a.cache]
  cmd+=["-p","0","-n","0","-pg",f"{a.prompt},{a.decode}"] if a.paired else ["-p",str(a.prompt),"-n",str(a.decode)]
  cmd+=["-r",str(a.repetitions),"-o","json"]
  receipt["command"]=["nice","-n","10","timeout",str(a.timeout)]+cmd
  started=time.monotonic()
  result=ssh(a.node,shlex.join(receipt["command"]),a.timeout+30)
  receipt["elapsed_seconds"]=round(time.monotonic()-started,3)
  receipt["returncode"]=result.returncode
  receipt["stdout"]=result.stdout[-120000:];receipt["stderr"]=result.stderr[-12000:]
  if result.returncode:raise RuntimeError("llama-bench returned nonzero exit")
  rows=json.loads(result.stdout)
  if not isinstance(rows,list) or not rows:raise RuntimeError("empty or invalid benchmark JSON")
  receipt["backend"]=sorted(set(r.get("backends","") for r in rows))
  if a.gpu_layers>0 and all("CPU" in b for b in receipt["backend"]):raise RuntimeError("GPU configuration silently used CPU")
  receipt["status"]="completed"
 except Exception as exc:
  receipt["status"]="skipped" if "below safety limit" in str(exc) else "failed"
  receipt["reason"]=str(exc)
 finally:
  path.write_text(json.dumps(receipt,indent=2)+"\n")
  print(json.dumps({"receipt":str(path),"status":receipt["status"],"backend":receipt.get("backend"),
       "reason":receipt.get("reason"),"available_mib":receipt.get("available_mib")}))
 return 0 if receipt["status"] in ("completed","skipped") else 2
if __name__=="__main__":sys.exit(main())
