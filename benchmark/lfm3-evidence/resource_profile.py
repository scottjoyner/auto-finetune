#!/usr/bin/env python3
"""Read-only process and DRM memory sampling for an isolated llama-bench child."""
import argparse, datetime, hashlib, json, os, pathlib, psutil, subprocess, sys, time
ROOT=pathlib.Path(__file__).resolve().parent
MODEL='/home/scott/lfm3-research/gguf/LFM3-1.7B-Q4_K_M.gguf'
SHA='3793e1e6e2266919d6e015287221b41100175a98eddaba92e111aed3b05c93c4'
BIN='/home/scott/lfm3-research/llama-build/bin/llama-bench'
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def gpu_mem():
 v={}
 for path in pathlib.Path('/sys/class/drm').glob('card*/device/mem_info_*_used'):
  try:v[str(path)]=int(path.read_text())
  except (ValueError,PermissionError,OSError):pass
 return v
def main():
 p=argparse.ArgumentParser()
 p.add_argument('--cache',choices=['f16','q8_0'],default='f16')
 p.add_argument('--prompt',type=int,default=8192)
 p.add_argument('--decode',type=int,default=64)
 p.add_argument('--repetitions',type=int,default=2)
 a=p.parse_args()
 if not (128<=a.prompt<=8192 and 16<=a.decode<=128 and 1<=a.repetitions<=3):p.error('bounds exceeded')
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
 r={'schema':'lfm3.mem-profile.v1','created_utc':stamp,'cache':a.cache,'model':MODEL,
    'expected_sha256':SHA,'status':'not_run','samples':0,'sampling_period_s':.1}
 path=ROOT/('resource-'+a.cache+'-'+stamp+'.json')
 try:
  if sha(MODEL)!=SHA:raise RuntimeError('model SHA mismatch')
  cmd=['nice','-n','12',BIN,'-m',MODEL,'-ngl','99','-t','4','-ctk',a.cache,'-ctv',a.cache,
       '-p','0','-n','0','-pg',str(a.prompt)+','+str(a.decode),'-r',str(a.repetitions),'-o','json']
  r['command']=cmd
  baseline=gpu_mem();r['gpu_baseline_bytes']=baseline
  before=psutil.virtual_memory().available;r['ram_available_before_bytes']=before
  started=time.monotonic()
  proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  pproc=psutil.Process(proc.pid)
  max_rss=0;max_memory={k:0 for k in baseline}
  while proc.poll() is None:
   try:max_rss=max(max_rss,pproc.memory_info().rss)
   except psutil.Error:pass
   for k,v in gpu_mem().items():max_memory[k]=max(max_memory.get(k,0),v)
   r['samples']+=1
   if time.monotonic()-started>180:
    proc.terminate();raise RuntimeError('process timed out')
   time.sleep(.1)
  stdout,stderr=proc.communicate(timeout=6)
  r['elapsed_s']=round(time.monotonic()-started,3)
  r['returncode']=proc.returncode
  r['stdout']=stdout.decode(errors='replace')[-50000:]
  r['stderr']=stderr.decode(errors='replace')[-10000:]
  r['peak_rss_bytes']=max_rss
  r['gpu_peak_bytes']=max_memory
  r['gpu_peak_delta_from_baseline_bytes']={k:max(0,max_memory.get(k,0)-v) for k,v in baseline.items()}
  r['ram_available_after_bytes']=psutil.virtual_memory().available
  if proc.returncode!=0:raise RuntimeError('nonzero llama-bench exit')
  parsed=json.loads(r['stdout'])
  if len(parsed)!=1 or parsed[0].get('n_prompt')!=a.prompt or parsed[0].get('n_gen')!=a.decode:
   raise RuntimeError('unexpected benchmark output')
  r['status']='completed';r['throughput_tps']=parsed[0]['avg_ts']
 except Exception as e:r.update(status='failed',reason=str(e))
 finally:
  path.write_text(json.dumps(r,indent=2)+'\n')
  print(json.dumps({'file':str(path),'status':r['status'],'reason':r.get('reason'),
       'throughput_tps':r.get('throughput_tps'),'peak_rss_mib':round(r.get('peak_rss_bytes',0)/(1024*1024),2),
       'gpu_peak_delta_mib':{k:round(v/(1024*1024),2) for k,v in r.get('gpu_peak_delta_from_baseline_bytes',{}).items()}},indent=2))
 return 0 if r['status']=='completed' else 2
if __name__=='__main__':sys.exit(main())
