"""Practical regression screening, not a statistical equivalence proof."""
import argparse
import collections
import csv
import json
import statistics
from pathlib import Path

ENV_KEYS = ['gpu','driver','torch','transformers','cuda_runtime','cpu_threads','revision',
            'attention','tf32','seed','image_id']

def delta_percent(old, new):
    return None if old == 0 else (new-old)/old*100

def cv(record):
    values = [x['elapsed_ms'] for x in record.get('raw_invocations', [])]
    return statistics.stdev(values)/statistics.mean(values) if len(values)>1 else 0

def metrics(record):
    result = {}
    for name,key in [('latency','latency_ms'),('ttft','ttft_ms'),('tpot','tpot_ms')]:
        values = record.get(key) or {}
        for statistic in ('median','p90'):
            if values.get(statistic) is not None:
                result[name+'_'+statistic+'_ms'] = values[statistic]
    t = record.get('throughput') or {}
    for name in ('requests_per_second','output_tokens_per_second'):
        if t.get(name) is not None:
            result[name] = t[name]
    for name in ('inference_peak_allocated_bytes','inference_peak_reserved_bytes','nvml_process_sampled_peak_bytes'):
        if record.get(name) is not None:
            result[name] = record[name]
    for name in ('parameter_storage_bytes','buffer_storage_bytes'):
        if (record.get('memory_baseline') or {}).get(name) is not None:
            result[name] = record['memory_baseline'][name]
    snapshots = record.get('component_snapshots') or []
    if snapshots:
        for name in ('self_kv_storage_bytes','cross_kv_storage_bytes'):
            result[name] = max(x[name] for x in snapshots)
    return result

def compare_record(old, new, latency_pct=5, token_pct=10, allocated_pct=1, reserved_pct=5):
    problems = []
    if old['case'] != new['case']:
        problems.append('case_definition_changed')
    if old['status'] != 'ok' or new['status'] != 'ok':
        return [], ['status:'+old['status']+'->'+new['status']]
    for name in ENV_KEYS:
        if old.get('environment',{}).get(name) != new.get('environment',{}).get(name):
            problems.append('environment:'+name)
    if len(old.get('raw_invocations',[])) != len(new.get('raw_invocations',[])):
        problems.append('iteration_count_changed')
    if old.get('scope') != new.get('scope') or old.get('actual_input_shapes') != new.get('actual_input_shapes'):
        problems.append('scope_or_shape_changed')
    if new['mode']=='memory' and new.get('nvml_status') != 'ok':
        problems.append('nvml_not_fully_available')
    if any((new.get(key) or {}).get('thermal_slowdown_at_boundary') is True
           for key in ('gpu_health_before_measurement','gpu_health_after_measurement')):
        problems.append('thermal_slowdown_at_measurement_boundary')
    rows=[]
    a,b=metrics(old),metrics(new)
    for metric in sorted(a.keys()|b.keys()):
        x,y=a.get(metric),b.get(metric)
        if x is None or y is None:
            rows.append({'metric':metric,'old':x,'new':y,'delta_pct':None,'tolerance_abs':0,'screen':'missing'})
            continue
        if metric in ('parameter_storage_bytes','buffer_storage_bytes','self_kv_storage_bytes','cross_kv_storage_bytes'):
            tolerance=0
        elif metric=='inference_peak_allocated_bytes':
            tolerance=max(2**20,abs(x)*allocated_pct/100)
        elif metric.endswith('_bytes'):
            tolerance=max(32*2**20,abs(x)*reserved_pct/100)
        else:
            pct=token_pct if metric.startswith(('ttft','tpot')) else latency_pct
            tolerance=abs(x)*pct/100
        rows.append({'metric':metric,'old':x,'new':y,'delta_pct':delta_percent(x,y),
                     'tolerance_abs':tolerance,'screen':'within_screen' if abs(y-x)<=tolerance else 'investigate'})
    if max(cv(old),cv(new))>.05 and new['mode'] in ('performance','first_logits'):
        problems.append('within_run_cv_over_5pct')
    return rows, problems

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--latency-pct',type=float,default=5)
    p.add_argument('--token-pct',type=float,default=10)
    p.add_argument('--allocated-pct',type=float,default=1)
    p.add_argument('--reserved-pct',type=float,default=5)
    a=p.parse_args()
    candidate_index=json.loads((a.candidate/'index.json').read_text(encoding='utf-8'))
    all_rows,issues=[],[]
    compared=set()
    for item in candidate_index:
        new=json.loads((a.candidate/item['file']).read_text(encoding='utf-8'))
        if new['status']=='not_applicable':
            continue
        # The existing reference has r0 only; do not invent independent reference repeats.
        path=a.reference/f'{item["case_id"]}__{item["mode"]}__r0.json'
        if not path.exists():
            issues.append({'case_id':item['case_id'],'mode':item['mode'],'issue':'missing_reference'})
            continue
        old=json.loads(path.read_text(encoding='utf-8'))
        rows,problems=compare_record(old,new,a.latency_pct,a.token_pct,a.allocated_pct,a.reserved_pct)
        for r in rows:
            all_rows.append({'case_id':item['case_id'],'mode':item['mode'],'repeat':item['repeat'],**r})
        for issue in problems:
            issues.append({'case_id':item['case_id'],'mode':item['mode'],'issue':issue})
        compared.add(item['case_id'])
    fields=['case_id','mode','repeat','metric','old','new','delta_pct','tolerance_abs','screen']
    with (a.candidate/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(all_rows)
    report={'compared_cases':len(compared),'metric_counts':dict(collections.Counter(r['screen'] for r in all_rows)),
            'issues':issues,'thresholds':{'latency_pct':a.latency_pct,'token_pct':a.token_pct,
            'allocated_pct':a.allocated_pct,'reserved_pct':a.reserved_pct,'allocated_floor_bytes':2**20,
            'reserved_nvml_floor_bytes':32*2**20},
            'interpretation':'Initial practical regression screen only. Not proof of equivalence; inspect variability, temperature, and flagged changes before full execution.'}
    (a.candidate/'comparison-summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    print('Metric details:',a.candidate/'comparison.csv')

if __name__=='__main__':
    main()
