#!/usr/bin/env python3
"""Non-destructive llama-bench runner; refuses to fabricate measurements."""
import argparse, datetime, hashlib, json, os, pathlib, platform, shutil, subprocess, sys, time
p=argparse.ArgumentParser()
p.add_argument('--model', default='/home/scott/lfm3-research/gguf/LFM3-1.7B-Q4_K_M.gguf')
p.add_argument('--binary',default='llama-bench')
p.add_argument('--out',default='/home/scott/lfm3-research/bench-receipts')
p.add_argument('--ngl',type=int,default=0)
p.add_argument('--threads',type=int,default=4)
p.add_argument('--prompt',type=int,default=512)
p.add_argument('--decode',type=int,default=128)
p.add_argument('--repetitions',type=int,default=3)
p.add_argument('--paired',action='store_true',help='measure prefill and decode in one request')
p.add_argument('--cache',choices=['f16','q8_0'],default='f16')
p.add_argument('--timeout',type=int,default=240)
a=p.parse_args()
out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
model=pathlib.Path(a.model)
if not (1<=a.prompt<=8192 and 1<=a.decode<=256 and 1<=a.repetitions<=5):p.error('benchmark bounds exceeded')
cmd=[a.binary,'-m',str(model),'-ngl',str(a.ngl),'-t',str(a.threads),'-ctk',a.cache,'-ctv',a.cache]
cmd+=['-p','0','-n','0','-pg',f'{a.prompt},{a.decode}'] if a.paired else ['-p',str(a.prompt),'-n',str(a.decode)]
cmd+=['-r',str(a.repetitions),'-o','json']
receipt={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':platform.node(),'platform':platform.platform(),'model':str(model),'model_exists':model.is_file(),'command':cmd,'status':'not_run','mode':'paired' if a.paired else 'separate','cache_type':a.cache}
if model.is_file():
 h=hashlib.sha256()
 with model.open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 receipt['model_sha256']=h.hexdigest()
if not model.is_file():receipt['reason']='model missing'
elif not shutil.which(a.binary) and not pathlib.Path(a.binary).is_file():receipt['reason']='llama-bench binary missing'
else:
 t=time.monotonic()
 try:
  r=subprocess.run(['nice','-n','10']+cmd,capture_output=True,text=True,timeout=a.timeout)
  receipt.update(status='completed' if r.returncode==0 else 'failed',returncode=r.returncode,elapsed_seconds=round(time.monotonic()-t,3),stdout=r.stdout[-100000:],stderr=r.stderr[-20000:])
 except Exception as e:receipt.update(status='failed',reason=str(e))
path=out/('receipt-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
path.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'receipt':str(path),'status':receipt['status'],'reason':receipt.get('reason')},indent=2))
sys.exit(0 if receipt['status']=='completed' else 2)