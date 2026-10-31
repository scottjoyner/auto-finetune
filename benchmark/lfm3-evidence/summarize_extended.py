#!/usr/bin/env python3
"""Concise reference index for measured concurrency, memory and retrieval probes."""
import json, pathlib, sys
ROOT=pathlib.Path(__file__).resolve().parent
def entry(p):
 r=json.loads(p.read_text())
 x={'receipt':p.name,'schema':r.get('schema'),'status':r.get('status'),
    'cache_type':r.get('cache') or r.get('cache_type'),'reason':r.get('reason')}
 if x['status']=='completed':
  if p.name.startswith('concurrency-'):
   x['results']=[{k:z[k] for k in ('concurrency','median_latency_seconds','p95_latency_seconds','mean_aggregate_decode_tokens_per_second')} for z in r['results']]
  elif p.name.startswith('resource-'):
   x['throughput_tps']=r.get('throughput_tps')
   x['peak_rss_bytes']=r.get('peak_rss_bytes')
   x['gpu_peak_delta_from_baseline_bytes']=r.get('gpu_peak_delta_from_baseline_bytes')
  elif p.name.startswith('quality-'):
   x['substring_indicators']=[z['substring_indicator'] for z in r['results']]
   x['exact_indicators']=[z['exact_indicator'] for z in r['results']]
   x['important']='substring indicators are not adjudicated behavioral accuracy'
 return x
def main():
 paths=[p for pat in ('concurrency-*.json','resource-*.json','quality-*.json') for p in sorted(ROOT.glob(pat)) if p.name != 'quality-scored.json']
 records=[entry(p) for p in paths]
 result={'schema':'lfm3.extended-evidence.v1','warning':'Exploratory runs were not thermally controlled or randomized; no cross-run performance claims.',
  'results':records}
 p=ROOT/'extended-summary.json';p.write_text(json.dumps(result,indent=2)+'\n')
 print('wrote',p,len(records),'experiments',sum(r['status']=='completed' for r in records),'completed')
 return 0
if __name__=='__main__':sys.exit(main())
