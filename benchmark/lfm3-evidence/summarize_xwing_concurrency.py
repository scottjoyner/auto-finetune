#!/usr/bin/env python3
"""Fail-closed xwing FA candidate concurrency receipt index, including failed runs."""
import json
import pathlib
import statistics
import sys

ROOT=pathlib.Path(__file__).resolve().parent
MODEL='43a7d09a0e9cba91b63eb7052488b9564569bddfd65754d2e426465e572a7452'
SERVER='85f3baf1a389ff45e914beef69b1e3e5d524a262472bbac10aa19759547bc662'


def require(test,message):
    if not test:raise ValueError(message)


def check(p):
    r=json.loads(p.read_text())
    require(r.get('schema')=='lfm3.xwing-hip-fa-concurrency.v1',p.name+': schema')
    require(r.get('model_sha256_expected')==MODEL,p.name+': unexpected checkpoint')
    require(r.get('status') in ('completed','failed','skipped'),p.name+': invalid status')
    if r.get('server_binary_sha256'):
        require(r['server_binary_sha256']==SERVER,p.name+': unexpected server executable')
    entry={'receipt':p.name,'status':r['status'],'reason':r.get('reason'),
           'model_sha256':r.get('model_sha256_observed'),
           'server_sha256':r.get('server_binary_sha256'),
           'results':[]}
    if r['status']!='completed':return entry
    require(r.get('model_sha256_observed')==MODEL,p.name+': checkpoint hash mismatch')
    require(r.get('server_binary_sha256')==SERVER,p.name+': executable hash mismatch')
    require(r.get('server_ready') is True,p.name+': no healthy server')
    require(r.get('flash_attn')=='on' and r.get('cache_type')=='q8_0',p.name+': unexpected cache or attention')
    cmd=r.get('server_command',[])
    for flag in ['127.0.0.1','--no-webui','--log-disable','-ctk','-ctv','-fa']:
        require(flag in cmd,p.name+': server command missing '+flag)
    rows=r.get('results',[])
    require([row.get('concurrency') for row in rows]==[1,2,4],p.name+': missing concurrency levels')
    for row in rows:
        n=row['concurrency']
        waves=row.get('waves',[])
        require(len(waves)==r['waves_per_load'],p.name+': missing workload waves')
        require(row.get('samples')==n*len(waves),p.name+': unexpected sample count')
        latencies=[]
        tps=[]
        for wave in waves:
            jobs=wave['requests']
            require(len(jobs)==n,p.name+': invalid wave size')
            require(wave['wall_seconds']>0,p.name+': invalid wall time')
            tokens=sum(x['predicted_tokens'] for x in jobs)
            require(tokens>0,p.name+': zero generated tokens')
            computed=round(tokens/wave['wall_seconds'],3)
            require(abs(computed-wave['aggregate_generated_tps'])<=0.2,
                    p.name+': aggregate rate inconsistent with rounded wall time')
            tps.append(wave['aggregate_generated_tps'])
            for job in jobs:
                require(job['elapsed_seconds']>0 and job['predicted_tokens']>0,
                        p.name+': invalid request timing')
                latencies.append(job['elapsed_seconds'])
        expected_median=round(statistics.median(latencies),4)
        require(abs(expected_median-row['median_client_latency_s'])<=0.0001,p.name+': median wrong')
        require(row['sample_p95_client_latency_s']>=row['median_client_latency_s'],
                p.name+': sample p95 less than median')
        require(abs(statistics.mean(tps)-row['mean_aggregate_generated_tps'])<=0.002,
                p.name+': mean throughput wrong')
        entry['results'].append({
            'concurrency':n,'samples':len(latencies),
            'median_client_latency_s':row['median_client_latency_s'],
            'sample_p95_client_latency_s':row['sample_p95_client_latency_s'],
            'mean_aggregate_generated_tps':row['mean_aggregate_generated_tps']})
    return entry


def main():
    inputs=sorted(p for p in ROOT.glob('xwing-concurrency-*.json') if p.name!='xwing-concurrency-summary.json')
    if not inputs:
        require(not (ROOT/'xwing-concurrency-summary.json').exists(),'orphan concurrency summary')
        print('No xwing concurrency receipts')
        return 0
    attestation=ROOT/'hip-fa-server-attestation-20261031.json'
    require(attestation.is_file(),'missing server attestation')
    att=json.loads(attestation.read_text())
    require(att.get('schema')=='lfm3.xwing-hip-fa-server-provenance.v1',
            'invalid server attestation')
    require(att.get('sha256',{}).get('model')==MODEL and att['sha256'].get('server')==SERVER,
            'unrecognized attested server')
    require(att.get('sha256',{}).get('backend_library')==
            '0a8573c81efa6a3383ac97112d73564e0b4d4b468ada32025c230584c8354ea3',
            'backend library incompatible')
    entries=[check(p) for p in inputs]
    summary={'schema':'lfm3.xwing-hip-fa-concurrency-summary.v1',
             'model_sha256':MODEL,'server_sha256':SERVER,
             'warning':'Small synthetic localhost-only samples. Sample p95 is descriptive, not a production SLA. Includes failed port reuse.',
             'receipts':entries}
    out=ROOT/'xwing-concurrency-summary.json'
    out.write_text(json.dumps(summary,indent=2)+'\n')
    print('wrote',out,'completed',sum(x['status']=='completed' for x in entries),
          'failed',sum(x['status']=='failed' for x in entries))
    for x in entries:
        print(x['receipt'],x['status'],[(y['concurrency'],y['mean_aggregate_generated_tps']) for y in x['results']])
    return 0


if __name__=='__main__':
    try:sys.exit(main())
    except (ValueError,KeyError,TypeError,json.JSONDecodeError) as exc:
        print('FAIL',exc,file=sys.stderr);sys.exit(1)
