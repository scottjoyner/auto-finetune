#!/usr/bin/env python3
"""Isolated localhost llama-server burst test; stops only its own process."""
import argparse, concurrent.futures, datetime, hashlib, json, os, pathlib, signal, socket
import statistics, subprocess, sys, threading, time, urllib.error, urllib.request
ROOT=pathlib.Path(__file__).resolve().parent
MODEL='/home/scott/lfm3-research/gguf/LFM3-1.7B-Q4_K_M.gguf'
SHA='3793e1e6e2266919d6e015287221b41100175a98eddaba92e111aed3b05c93c4'
SERVER='/home/scott/lfm3-research/llama-build/bin/llama-server'
def http(url,data=None,timeout=6):
 req=urllib.request.Request(url,data=json.dumps(data).encode() if data is not None else None,
   headers={'Content-Type':'application/json'} if data is not None else {})
 with urllib.request.urlopen(req,timeout=timeout) as response:
  return json.loads(response.read(1000000))
def quantile(xs,p):
 ys=sorted(xs);return ys[min(len(ys)-1,max(0,int((len(ys)-1)*p+.999999)))]
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 a=argparse.ArgumentParser()
 a.add_argument('--max-parallel',type=int,default=4)
 a.add_argument('--repetitions',type=int,default=3)
 a.add_argument('--tokens',type=int,default=48)
 a.add_argument('--port',type=int,default=29617)
 a.add_argument('--cache',choices=['f16','q8_0'],default='f16')
 arg=a.parse_args()
 if not(1<=arg.max_parallel<=4 and 1<=arg.repetitions<=5 and 8<=arg.tokens<=128 and 1025<=arg.port<=65535):a.error('bounds exceeded')
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
 receipt={'schema':'lfm3.concurrent.v1','created_utc':stamp,'hostname':socket.gethostname(),
  'model':MODEL,'model_sha256_expected':SHA,'cache':arg.cache,
  'max_parallel':arg.max_parallel,'repetitions':arg.repetitions,'tokens':arg.tokens,'status':'not_run','results':[]}
 path=ROOT/('concurrency-'+stamp+'.json');proc=None;out=None;stop=threading.Event()
 try:
  if sha(MODEL)!=SHA:raise RuntimeError('model SHA mismatch')
  if pathlib.Path('/proc/meminfo').read_text().split('MemAvailable:')[1].split()[0].isdigit():
   avail=int(pathlib.Path('/proc/meminfo').read_text().split('MemAvailable:')[1].split()[0])//1024
   receipt['preflight_available_mib']=avail
   if avail<4096:raise RuntimeError('memory safety gate below 4 GiB')
  sock=socket.socket();sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);sock.bind(('127.0.0.1',arg.port));sock.close()
  cmd=['nice','-n','12',SERVER,'-m',MODEL,'-ngl','99','-t','4','-c','8192','-np',str(arg.max_parallel),
   '-ctk',arg.cache,'-ctv',arg.cache,'--host','127.0.0.1','--port',str(arg.port),'--log-disable']
  receipt['command']=cmd
  log=ROOT/('concurrency-'+stamp+'.log')
  out=open(log,'w')
  proc=subprocess.Popen(cmd,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
  receipt['server_pid']=proc.pid
  base='http://127.0.0.1:'+str(arg.port)
  deadline=time.monotonic()+75
  while time.monotonic()<deadline:
   if proc.poll() is not None:raise RuntimeError('server exited '+str(proc.returncode))
   try:
    health=http(base+'/health',timeout=2)
    if health.get('status')=='ok':break
   except Exception:pass
   time.sleep(.6)
  else:raise RuntimeError('server readiness timeout')
  receipt['startup_ready']=True
  def task(i):
   prompt=('You are a concise technical assistant. Explain why careful benchmarking must record hashes, '
     'model inputs, and measured latency, not invented values. ')*3+f' Request index {i}.'
   body={'prompt':prompt,'n_predict':arg.tokens,'temperature':0,'seed':20+i,'cache_prompt':False,'stream':False}
   start=time.monotonic()
   resp=http(base+'/completion',body,timeout=90)
   return {'latency_seconds':round(time.monotonic()-start,4),'predicted_n':resp.get('timings',{}).get('predicted_n'),
    'predicted_ms':resp.get('timings',{}).get('predicted_ms'),'prompt_n':resp.get('timings',{}).get('prompt_n'),
    'content_len':len(resp.get('content','')),'stop':resp.get('stop'),'error':resp.get('error')}
  warm=task(9999);receipt['warmup']=warm
  if not warm['predicted_n'] or warm.get('error'):raise RuntimeError('warmup returned no tokens')
  for n in [1,2,4]:
   if n>arg.max_parallel:continue
   waves=[]
   for w in range(arg.repetitions):
    start=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
     fut=[pool.submit(task,w*1000+j) for j in range(n)]
     results=[f.result(timeout=95) for f in fut]
    wall=time.monotonic()-start
    if any((not x.get('predicted_n') or x.get('error')) for x in results):raise RuntimeError('invalid completion response')
    waves.append({'requests':results,'wall_seconds':round(wall,4),'aggregate_tokens_per_second':round(sum(x['predicted_n'] for x in results)/wall,3)})
   lat=[z['latency_seconds'] for w in waves for z in w['requests']]
   receipt['results'].append({'concurrency':n,'waves':waves,'median_latency_seconds':round(statistics.median(lat),3),
     'p95_latency_seconds':round(quantile(lat,.95),3),
     'mean_aggregate_decode_tokens_per_second':round(statistics.mean(x['aggregate_tokens_per_second'] for x in waves),3)})
  receipt['status']='completed'
 except Exception as e:receipt.update(status='failed',reason=str(e))
 finally:
  if proc:
   try:os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=12)
   except Exception:
    try:os.killpg(proc.pid,signal.SIGKILL)
    except ProcessLookupError:pass
  if out:out.close()
  path.write_text(json.dumps(receipt,indent=2)+'\n')
  print(json.dumps({'file':str(path),'status':receipt['status'],'reason':receipt.get('reason'),
   'results':[{k:x[k] for k in ('concurrency','median_latency_seconds','p95_latency_seconds','mean_aggregate_decode_tokens_per_second')} for x in receipt['results']]},indent=2))
 return 0 if receipt['status']=='completed' else 2
if __name__=='__main__':sys.exit(main())
