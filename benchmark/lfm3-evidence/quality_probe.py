#!/usr/bin/env python3
"""Deterministic synthetic retrieval probes; never present substring matching as verified accuracy."""
import argparse, datetime, hashlib, json, os, pathlib, signal, subprocess, sys, time, urllib.request
from concurrency import http, sha
ROOT=pathlib.Path(__file__).resolve().parent
MODEL='/home/scott/lfm3-research/gguf/LFM3-1.7B-Q4_K_M.gguf'
SHA='3793e1e6e2266919d6e015287221b41100175a98eddaba92e111aed3b05c93c4'
BIN='/home/scott/lfm3-research/llama-build/bin/llama-server'
def main():
 p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=29618)
 p.add_argument('--cache',choices=['f16','q8_0'],default='f16')
 a=p.parse_args()
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
 path=ROOT/('quality-'+a.cache+'-'+stamp+'.json')
 r={'schema':'lfm3.quality-probe.v1','created_utc':stamp,'cache_type':a.cache,
   'checkpoint_sha256':SHA,'status':'not_run','results':[],'score_description':'substring inclusion in generated text, NOT adjudicated answer correctness'}
 proc=None;log=None
 try:
  if sha(MODEL)!=SHA:raise RuntimeError('checkpoint mismatch')
  cmd=['nice','-n','12',BIN,'-m',MODEL,'-ngl','99','-t','4','-c','8192','-np','1',
       '-ctk',a.cache,'-ctv',a.cache,'--host','127.0.0.1','--port',str(a.port),'--log-disable']
  r['command']=cmd
  log=open(ROOT/('quality-'+a.cache+'-'+stamp+'.log'),'w')
  proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  url='http://127.0.0.1:'+str(a.port)
  deadline=time.monotonic()+75
  while time.monotonic()<deadline:
   if proc.poll() is not None:raise RuntimeError('server exited before readiness')
   try:
    if http(url+'/health',timeout=2).get('status')=='ok':break
   except Exception:pass
   time.sleep(.4)
  else:raise RuntimeError('server readiness timeout')
  for idx,position in enumerate([4,32,64]):
   key=['JQ7X-2649','LP5B-7103','ZD4N-8192'][idx]
   context=[]
   for i in range(72):
    text=f'Inventory row {i:03d}: standard components approved; status pending routine validation. '
    if i==position:text=f'Inventory row {i:03d}: verification code {key}; status approved. '
    context.append(text)
   prompt='Read this inventory excerpt and answer the question using only its contents.\n'+''.join(context)+f'\nQuestion: What is the verification code for inventory row {position:03d}?\nAnswer:'
   req={'prompt':prompt,'temperature':0,'seed':42,'n_predict':128,'stream':False,'cache_prompt':False}
   t=time.monotonic();response=http(url+'/completion',req,timeout=90)
   content=response.get('content','')
   if not isinstance(content,str):raise RuntimeError('bad response content')
   r['results'].append({'row':position,'expected':key,'prompt_chars':len(prompt),
      'generated':content[:6000],'substring_indicator':key in content,
      'exact_indicator':content.strip().strip(' \"\'.,')==key,
      'returned_tokens':response.get('timings',{}).get('predicted_n'),
      'prompt_tokens':response.get('timings',{}).get('prompt_n'),
      'elapsed_s':round(time.monotonic()-t,3),'stop':response.get('stop'),'error':response.get('error')})
  r['status']='completed'
 except Exception as exc:r.update(status='failed',reason=str(exc))
 finally:
  if proc:
   try:os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=10)
   except Exception:
    try:os.killpg(proc.pid,signal.SIGKILL)
    except ProcessLookupError:pass
  if log:log.close()
  path.write_text(json.dumps(r,indent=2)+'\n')
  print(json.dumps({'path':str(path),'status':r['status'],'reason':r.get('reason'),
      'indicators':[{'row':i['row'],'contains':i['substring_indicator'],'exact':i['exact_indicator'],
       'tokens':i['returned_tokens']} for i in r['results']]},indent=2))
 return 0 if r['status']=='completed' else 2
if __name__=='__main__':sys.exit(main())
