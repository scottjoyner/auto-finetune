#!/usr/bin/env python3
"""Summarize completed and blocked tests without silently dropping failures."""
import json, pathlib, sys
ROOT=pathlib.Path(__file__).resolve().parent
def parse(p):
 r=json.loads(p.read_text())
 if p.name.startswith('fleet-'):
  base={k:r.get(k) for k in ('node','status','reason','model','observed_model_sha256','available_mib','repetitions','elapsed_seconds','backend')}
 else:
  base={'node':'x1-370','status':r.get('status'),'reason':r.get('reason'),'model':r.get('model'),
    'observed_model_sha256':r.get('model_sha256'),'repetitions':int(r['command'][r['command'].index('-r')+1]),
    'mode':r.get('mode','separate'),'cache_type':r.get('cache_type')}
 base['receipt']=p.name
 if base['status']=='completed':
  try:
   rows=json.loads(r['stdout'])
   base['measurements']=[{k:x.get(k) for k in
      ('model_type','build_commit','backends','gpu_info','n_gpu_layers','n_prompt','n_gen',
       'type_k','type_v','avg_ts','stddev_ts','avg_ns','stddev_ns')}
       for x in rows]
  except (KeyError,ValueError,TypeError) as e:
   base['status']='invalid_receipt';base['reason']=str(e)
 return base
def main():
 data=[parse(p) for p in sorted(ROOT.glob('receipt-*.json'))+sorted(p for p in ROOT.glob('fleet-*.json') if p.name!='fleet-summary.json')]
 out=ROOT/'fleet-summary.json'
 out.write_text(json.dumps({'schema':'lfm3.fleet-evidence.v1','results':data},indent=2)+'\n')
 print('wrote',out,'total',len(data),'completed',sum(x['status']=='completed' for x in data),
       'blocked',sum(x['status']!='completed' for x in data))
 for x in data:
  print(x['node'],x['status'],x['receipt'],x.get('reason') or '')
 return 0 if data else 1
if __name__=='__main__':sys.exit(main())
