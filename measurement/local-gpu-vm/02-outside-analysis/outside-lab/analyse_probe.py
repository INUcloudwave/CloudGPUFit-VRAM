#!/usr/bin/env python3
"""Summarise process/stage records without treating time samples as repeats."""
import argparse
import collections
import csv
import json
from pathlib import Path
from statistics import median
from reanalyse import csv_write, describe

MIB=2**20

def clean(row):
    flags=[]
    if row.get('outside_bytes') is None:flags.append('missing_pair')
    elif row['outside_bytes']<0:flags.append('negative_outside')
    if row.get('query_end_ns',0)-row.get('query_begin_ns',0)>5_000_000:flags.append('query_over_5ms')
    if row.get('other_gpu_pids'):flags.append('other_process')
    if row.get('reserved_bracket_changed') or row.get('nvml_bracket_changed'):flags.append('bracket_changed')
    if row.get('errors'):flags.append('query_error')
    for key in ('nvml','nvml_after'):
        if row.get(key,{}).get('errors'):flags.append(key+'_error')
    return flags

def analyse(root,out):
    out.mkdir(parents=True,exist_ok=True);stage_rows=[];flag_rows=[];runs=[]
    for mp in sorted(root.rglob('manifest.json')):
        m=json.loads(mp.read_text());sample_file=mp.with_name('samples.jsonl')
        runs.append(dict(path=str(mp),status=m.get('status'),gpu=m.get('gpu'),args=m.get('args')))
        if not sample_file.exists():continue
        stages=collections.defaultdict(list)
        for line in sample_file.read_text().splitlines():
            r=json.loads(line);flags=clean(r)
            stages[(r['stage'],r['kind'])].append((r,flags))
            if flags:flag_rows.append(dict(run_id=m['run_id'],stage=r['stage'],query_begin_ns=r['query_begin_ns'],
                outside_bytes=r.get('outside_bytes'),flags=';'.join(flags)))
        for (stage,kind),pairs in stages.items():
            good=[r for r,flags in pairs if not flags];vals=[r['outside_bytes']/MIB for r in good]
            args=m['args'];g=m.get('gpu',{})
            stage_rows.append(dict(run_id=m['run_id'],repeat=m['independent_repeat'],pid=m['pid'],
                suite=str(mp.parent.parent),scenario=args['scenario'],case_id=args.get('case_id'),
                history=args['history'],sampling=args['sampling'],api_probe=args['api_probe'],
                query_order=args['query_order'],op_order=args['op_order'] if args['scenario']=='ops' else '',
                gpu_uuid=g.get('uuid'),gpu=g.get('name'),image_id=m.get('image_id'),driver=g.get('driver'),
                module_loading=m.get('environment',{}).get('CUDA_MODULE_LOADING'),
                stage=stage,kind=kind,n_raw=len(pairs),n_clean=len(good),**describe(vals),
                raw_negative=sum(r.get('outside_bytes') is not None and r['outside_bytes']<0 for r,_ in pairs),
                last10_span_mib=max(vals[-10:])-min(vals[-10:]) if len(vals)>=10 else None,
                last10_duration_ms=(good[-1]['query_end_ns']-good[-10]['query_begin_ns'])/1e6 if len(good)>=10 else None,
                first_mib=vals[0] if vals else None,last_mib=vals[-1] if vals else None,
                first_to_last_mib=vals[-1]-vals[0] if vals else None))
    csv_write(out/'process_stage.csv',stage_rows);csv_write(out/'quality_flags.csv',flag_rows)
    groups=collections.defaultdict(list)
    # Includes exact suite identity: never silently pool different environment sweeps.
    keys=('suite','gpu_uuid','image_id','driver','scenario','case_id','history','sampling','api_probe','query_order','op_order','stage','kind')
    for r in stage_rows:
        if r['kind']=='stable' and r['n_clean']>=10 and r['median'] is not None:
            groups[tuple(r[k] for k in keys)].append(r)
    repeat_rows=[]
    for k,rs in groups.items():
        vals=[r['median'] for r in rs]
        repeat_rows.append(dict(**dict(zip(keys,k)),n_processes=len(rs),
            process_ids_json=json.dumps([r['run_id'] for r in rs]),**{f'process_median_{x}':v for x,v in describe(vals).items()},
            process_median_range_mib=max(vals)-min(vals),
            spread_pass_exploratory=(max(vals)-min(vals)<=16) if len(vals)>=3 else None,
            unstable_process_stages=sum(r['last10_span_mib'] is not None and r['last10_span_mib']>8 for r in rs)))
    csv_write(out/'repeat_groups.csv',repeat_rows)
    (out/'audit.json').write_text(json.dumps(dict(runs=runs,n_stage_rows=len(stage_rows),n_flagged=len(flag_rows),
        rules=dict(query_window_max_ms=5,min_clean_stable_samples=10,plateau_span_last10_mib=8,
            repeat_process_median_range_mib=16),
        caveats=['Stage extrema are sampled; GPU transients can be missed.',
            'Quality flags never delete raw samples. Review missing pre-init separately.',
            'Process repetitions on one host do not measure between-host or GPU-model variation.',
            'No confidence interval or p-value inferred from time samples.']),indent=2))
    print(json.dumps(dict(n_runs=len(runs),n_stage_rows=len(stage_rows),n_repeat_groups=len(repeat_rows),n_flagged=len(flag_rows))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();analyse(a.input,a.out)
