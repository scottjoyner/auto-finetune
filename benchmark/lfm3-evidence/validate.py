#!/usr/bin/env python3
"""Fail-closed validation of measured LFM3 benchmark evidence."""
import json, pathlib, sys, hashlib
ROOT=pathlib.Path(__file__).resolve().parent
EXPECTED={
 'LFM3-1.7B-Q4_K_M.gguf':'3793e1e6e2266919d6e015287221b41100175a98eddaba92e111aed3b05c93c4',
 'LFM3-1.7B-QAD-Q4_0.gguf':'43a7d09a0e9cba91b63eb7052488b9564569bddfd65754d2e426465e572a7452',
}
def require(ok, why):
 if not ok:raise ValueError(why)
def check_receipt(p):
 r=json.loads(p.read_text())
 require(r.get('status') in ('completed','skipped','failed','not_run'),f'{p.name}: missing status')
 model=pathlib.PurePath(r.get('model','')).name
 if model in EXPECTED:
  fingerprint=r.get('model_sha256') or r.get('observed_model_sha256')
  require(fingerprint in (None,EXPECTED[model]),f'{p.name}: checkpoint mismatch')
 if r['status']!='completed':return
 require(r.get('returncode')==0,f'{p.name}: nonzero process status')
 rows=json.loads(r['stdout'])
 require(isinstance(rows,list) and len(rows)>0,f'{p.name}: no benchmark measurements')
 for row in rows:
  require(row.get('avg_ts',0)>0,f'{p.name}: invalid tokens/s')
  require(row.get('n_prompt',0)>0 or row.get('n_gen',0)>0,f'{p.name}: no workload')
  require(row.get('model_filename','').endswith(model),f'{p.name}: model path mismatch')
  if p.name.startswith('fleet-') and r.get('gpu_layers',0)>0:
   require('CPU' not in row.get('backends',''),f'{p.name}: GPU silently fell back to CPU')
def check_experiment(p):
 r=json.loads(p.read_text())
 require(r.get('status') in ('completed','failed','skipped'),f'{p.name}: no status')
 if r['status']!='completed':return
 if p.name.startswith('concurrency-'):
  require(r.get('results'),f'{p.name}: no clients')
  for batch in r['results']:
   require(batch['median_latency_seconds']>0 and batch['p95_latency_seconds']>=batch['median_latency_seconds'],f'{p.name}: invalid latency')
   for wave in batch['waves']:
    require(len(wave['requests'])==batch['concurrency'],f'{p.name}: incomplete wave')
    require(all(x.get('predicted_n',0)>0 for x in wave['requests']),f'{p.name}: missing tokens')
 if p.name.startswith('resource-'):
  require(r.get('samples',0)>0 and r.get('throughput_tps',0)>0,f'{p.name}: unmeasured resources')
  require(r.get('peak_rss_bytes',0)>0,f'{p.name}: no RSS')
 if p.name.startswith('quality-'):
  require(len(r.get('results',[]))==3,f'{p.name}: incomplete fixture')
  require(all(isinstance(x.get('substring_indicator'),bool) and isinstance(x.get('exact_indicator'),bool) for x in r['results']),f'{p.name}: invalid scoring')
def main():
 base=sorted(ROOT.glob('receipt-*.json'))+[p for p in sorted(ROOT.glob('fleet-*.json')) if p.name!='fleet-summary.json']
 extra=[p for pat in ('concurrency-*.json','resource-*.json','quality-*.json') for p in sorted(ROOT.glob(pat)) if p.name != 'quality-scored.json']
 for p in base:check_receipt(p)
 for p in extra:check_experiment(p)
 summary=ROOT/'fleet-summary.json'
 if summary.is_file():
  names={x['receipt'] for x in json.loads(summary.read_text())['results']}
  require(names=={p.name for p in base},'fleet-summary.json missing or duplicating receipts')
 smokes=list(ROOT.glob('rocm-gfx*-smoke-*.json'))
 for p in smokes:
  r=json.loads(p.read_text())
  require(r.get('schema')=='lfm3.rocm-compatibility.v1',f'{p.name}: invalid ROCm receipt schema')
  x=r.get('xwing',{})
  require(x.get('hipcc_returncode')==0,f'{p.name}: invalid xwing HIP compiler result')
  require(x.get('smoke_returncode')==0 and x.get('smoke_passed') is True and 'mismatches=0' in x.get('smoke_stdout',''),f'{p.name}: HIP smoke did not pass')
  source=ROOT/'rocm_gfx1151_smoke.cpp'
  require(source.is_file() and hashlib.sha256(source.read_bytes()).hexdigest()==x.get('smoke_source_sha256'),f'{p.name}: missing or mismatched HIP source')
  require('NO LFM3 HIP inference result' in x.get('llama_hip_build',''),f'{p.name}: model benchmark status not explicit')
 toolchains=sorted(ROOT.glob('rocm-*-toolchain-*.json'))
 for p in toolchains:
  r=json.loads(p.read_text())
  require(r.get('schema')=='lfm3.rocm-toolchain-audit.v1',f'{p.name}: invalid toolchain audit')
  require(r.get('installed_packages',{}).get('exit_code')==0,f'{p.name}: package listing failed')
  require('hipcc 5.7' in r['installed_packages']['stdout'] and 'hip-runtime-amd 7.2' in r['installed_packages']['stdout'],
          f'{p.name}: expected observed mixed toolchain versions')
  require(r.get('toolchain_path_exists',{}).get('hipcc_implicit_clang17') is False,
          f'{p.name}: missing-clang-17 observation changed')
 comparisons=sorted(ROOT.glob('backend-compare-*.json'))
 for p in comparisons:
  r=json.loads(p.read_text())
  require(r.get('schema')=='lfm3.xwing-matched-backends.v1',f'{p.name}: invalid matched-ABBA schema')
  require(r.get('order')==['Vulkan','HIP','HIP','Vulkan'],f'{p.name}: invalid comparison order')
  require(r.get('status') in ('completed','failed'),f'{p.name}: invalid comparison status')
  runs=r.get('runs',[])
  require(len(runs)<=4,f'{p.name}: too many benchmark runs')
  for row in runs:
   receipt=ROOT/row.get('receipt','')
   require(receipt.is_file() and receipt.name.startswith('fleet-xwing-'),f'{p.name}: missing source receipt')
   source=json.loads(receipt.read_text())
   require(source.get('observed_model_sha256')==r.get('expected_model_sha256'),f'{p.name}: incompatible model hashes')
   require(row.get('status')==source.get('status'),f'{p.name}: mismatch source status')
  if r['status']=='completed':
   require(len(runs)==4 and all(x.get('status')=='completed' for x in runs),f'{p.name}: incomplete success')
   require(all(r['order'][i]==x['backend_expected'] for i,x in enumerate(runs)),f'{p.name}: mismatched backend sequence')
   for row in runs:
    expected=row['backend_expected'].upper()
    observed=[b.upper() for b in row.get('backend_observed',[])]
    require(observed and all((expected in b if expected=='VULKAN' else 'HIP' in b or 'ROCM' in b) for b in observed),
            f'{p.name}: wrong backend')
    require(all(m.get('avg_ts',0)>0 for m in row.get('measurements',[])),f'{p.name}: invalid throughput')
 fingerprints=sorted(ROOT.glob('backend-fingerprint-*.json'))
 require(len(fingerprints)<=1, 'ambiguous environment fingerprints')
 for p in fingerprints:
  r=json.loads(p.read_text())
  require(r.get('schema')=='lfm3.xwing-backend-fingerprint.v1' and r.get('status')=='completed',f'{p.name}: invalid fingerprint')
  require(r.get('source_commit')=='8ca4060b7e30e76baaa5670ee1ec1b760b7f8552',f'{p.name}: unexpected source commit')
  items=r.get('source_files',[])
  require(len(items)==6 and all(x.get('match') and x.get('git_tree_file_sha256')==x.get('xwing_snapshot_sha256') for x in items),
          f'{p.name}: source file mismatch')
  for k in ['Vulkan','HIP']:
   binary=r.get('runtime_binaries',{}).get(k,{})
   require(len(binary.get('binary_sha256',''))==64,f'{p.name}: missing {k} binary SHA')
  require(any('GGML_HIP:BOOL=ON' in x for x in r['runtime_binaries']['HIP'].get('cmake_flags',[])),f'{p.name}: HIP not built')
  require(any('GGML_VULKAN:BOOL=ON' in x for x in r['runtime_binaries']['Vulkan'].get('cmake_flags',[])),f'{p.name}: Vulkan not built')
 if comparisons:
  require(len(fingerprints)==1 and (ROOT/'backend-comparison-summary.json').is_file(),
          'comparison summary or source provenance is missing')
  summary=json.loads((ROOT/'backend-comparison-summary.json').read_text())
  require(summary.get('schema')=='lfm3.xwing-backend-comparison-summary.v1', 'invalid comparison summary')
  reported={x['comparison_receipt'] for x in summary['comparisons']}
  blocked={x['comparison_receipt'] for x in summary.get('failed_comparisons',[])}
  require(not reported.intersection(blocked) and reported.union(blocked)=={p.name for p in comparisons},
          'comparison summary missing or double-counting evidence')
 print('PASS',len(base),'core receipts +',len(extra),'extended experiment receipts +',
       len(smokes),'ROCm smoke +',len(toolchains),'toolchain audit +',len(comparisons),'matched comparisons +',len(fingerprints),'source fingerprint')
 return 0
if __name__=='__main__':
 try:sys.exit(main())
 except (ValueError,KeyError,TypeError,json.JSONDecodeError) as e:print('FAIL',str(e),file=sys.stderr);sys.exit(1)
