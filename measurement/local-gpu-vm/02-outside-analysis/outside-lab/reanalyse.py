#!/usr/bin/env python3
"""CPU-only audit of immutable bench-v3 JSON. Standard library only."""
import argparse
import collections
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import median

MIB = 2**20
ARCHIVE_INSTANCES = {'g4dn-2xlarge':'g4dn.2xlarge','g5-2xlarge':'g5.2xlarge',
    'g6-xlarge':'g6.xlarge','g6-2xlarge':'g6.2xlarge','g6-4xlarge':'g6.4xlarge','g6e-2xlarge':'g6e.2xlarge'}

def quantile(xs, p):
    if not xs:
        return None
    s = sorted(xs); pos = (len(s)-1)*p; i = int(pos)
    return s[i] + (s[min(i+1, len(s)-1)]-s[i])*(pos-i)

def describe(xs):
    xs = [v for v in xs if v is not None]
    return dict(n=len(xs), min=min(xs) if xs else None,
                p05=quantile(xs,.05), median=quantile(xs,.5),
                p95=quantile(xs,.95), max=max(xs) if xs else None)

def csv_write(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'wt', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

def audit(root, out):
    out.mkdir(parents=True, exist_ok=True)
    summaries, samples, manifests, statuses = [], [], [], collections.Counter()
    for mp in sorted(root.rglob('run-manifest.json')):
        m=json.loads(mp.read_text()); run_id=mp.parent.name
        archive=mp.relative_to(root).parts[0]
        instance=ARCHIVE_INSTANCES.get(archive.split('-all')[0],archive)
        manifests.append(dict(run_id=run_id, manifest_path=str(mp.relative_to(root)),
            manifest_sha256=hashlib.sha256(mp.read_bytes()).hexdigest(), **m))
        for fp in sorted(mp.parent.glob('*__r*.json')):
            j=json.loads(fp.read_text()); c=j.get('case',{}); env=j.get('environment',{})
            mode=j.get('mode'); statuses[(instance,mode,j.get('status'))]+=1
            base=dict(run_id=run_id,instance=instance,archive=archive,
                instance_source='attachment_filename_and_user_table_unverified',
                declared_instance=m.get('instance_label'),declared_env_instance=env.get('instance_label'),gpu=env.get('gpu'),
                case_id=c.get('case_id'),model=c.get('model'),dtype=c.get('dtype'),
                params_json=json.dumps(c.get('params',{}),sort_keys=True),mode=mode,
                repeat=j.get('repeat'),status=j.get('status'),
                model_reused=j.get('model_reused'),source=str(fp.relative_to(root)))
            for key in ('startup_memory_samples','memory_samples','component_snapshots'):
                rows=j.get(key,[])
                if not rows:
                    continue
                vals=[]; durations=[]; gaps=[]; previous=None; nulls=negatives=mismatch=0
                for ix,r in enumerate(rows):
                    n=r.get('nvml_process_bytes'); v=r.get('reserved_bytes')
                    outside=n-v if n is not None and v is not None else None
                    if outside is None: nulls+=1
                    else:
                        vals.append(outside/MIB); negatives+=outside<0
                    if r.get('outside_reserved_bytes') is not None and outside!=r['outside_reserved_bytes']:
                        mismatch+=1
                    b=r.get('query_begin_ns'); e=r.get('query_end_ns')
                    if b is not None and e is not None: durations.append((e-b)/1e6)
                    if b is not None and previous is not None: gaps.append((b-previous)/1e6)
                    previous=b
                    samples.append(dict(**base,sample_group=key,sample_index=ix,
                        query_begin_ns=b,query_end_ns=e,query_ms=(e-b)/1e6 if b is not None and e is not None else None,
                        nvml_process_bytes=n,allocated_bytes=r.get('allocated_bytes'),reserved_bytes=v,
                        outside_bytes=outside,reported_outside_bytes=r.get('outside_reserved_bytes')))
                summaries.append(dict(**base,sample_group=key,n_samples=len(rows),n_valid=len(vals),
                    n_null=nulls,n_negative=negatives,n_mismatch=mismatch,
                    outside_min_mib=min(vals) if vals else None,outside_median_mib=median(vals) if vals else None,
                    outside_max_mib=max(vals) if vals else None,outside_p95_mib=quantile(vals,.95),
                    first_mib=next((r['nvml_process_bytes']-r['reserved_bytes'] for r in rows
                        if r.get('nvml_process_bytes') is not None),None)/MIB if vals else None,
                    last_mib=next((r['nvml_process_bytes']-r['reserved_bytes'] for r in reversed(rows)
                        if r.get('nvml_process_bytes') is not None),None)/MIB if vals else None,
                    query_ms_median=quantile(durations,.5),query_ms_p95=quantile(durations,.95),
                    gap_ms_median=quantile(gaps,.5),gap_ms_max=max(gaps) if gaps else None,
                    n_invocations=len(j.get('raw_invocations',[])),
                    sampling_errors_json=json.dumps({k:j[k] for k in j if 'sampling_errors' in k})))
    csv_write(out/'legacy_samples.csv.gz',samples)
    csv_write(out/'legacy_case_stage.csv',summaries)
    csv_write(out/'status_counts.csv',[dict(instance=k[0],mode=k[1],status=k[2],n=v) for k,v in statuses.items()])
    groups=collections.defaultdict(list)
    for row in summaries:
        groups[(row['instance'],row['mode'],row['sample_group'])].append(row)
    agg=[]
    for key,rs in sorted(groups.items()):
        stat=describe([r['outside_median_mib'] for r in rs])
        agg.append(dict(instance=key[0],mode=key[1],sample_group=key[2],**stat,
            n_samples=sum(r['n_samples'] for r in rs),n_negative=sum(r['n_negative'] for r in rs),
            n_null=sum(r['n_null'] for r in rs),n_le2=sum(r['n_samples']<=2 for r in rs),
            max_within_span_mib=max((r['outside_max_mib']-r['outside_min_mib'] for r in rs if r['n_valid']),default=None),
            median_sample_count=median(r['n_samples'] for r in rs),
            median_query_ms=quantile([r['query_ms_median'] for r in rs if r['query_ms_median'] is not None],.5),
            median_gap_ms=quantile([r['gap_ms_median'] for r in rs if r['gap_ms_median'] is not None],.5)))
    csv_write(out/'legacy_groups.csv',agg)
    detailed=[]
    for fields in [('instance','model','dtype'),('instance','model')]:
        g=collections.defaultdict(list)
        for r in summaries:
            if r['sample_group']=='memory_samples': g[tuple(r[f] for f in fields)].append(r)
        for k,rs in sorted(g.items()):
            detailed.append(dict(**dict(zip(fields,k)),**describe([r['outside_median_mib'] for r in rs])))
    csv_write(out/'legacy_model_dtype.csv',detailed)
    paired=collections.defaultdict(dict)
    for r in summaries:
        if r['outside_median_mib'] is not None:
            paired[(r['instance'],r['run_id'],r['case_id'],r['repeat'])][r['sample_group']]=r['outside_median_mib']
    csv_write(out/'legacy_stage_pairs.csv',[dict(instance=k[0],run_id=k[1],case_id=k[2],repeat=k[3],**v,
        components_minus_memory_mib=v['component_snapshots']-v['memory_samples'])
        for k,v in paired.items() if 'component_snapshots' in v and 'memory_samples' in v])
    (out/'legacy_audit.json').write_text(json.dumps(dict(manifests=manifests,
        groups=agg,n_samples=len(samples),n_case_stage=len(summaries),
        negative_samples=sum(r['outside_bytes'] is not None and r['outside_bytes']<0 for r in samples),
        outside_mismatch=sum(r['n_mismatch'] for r in summaries),
        caveats=['No per-API timestamps, raw PID, UUID, module-loading env or pre-init observations.',
                 'memory startup is reused model; performance/first_logits have no NVML timeline.',
                 'Cases have one repeat per archive; samples are not independent replicates.',
                 'Fixed case ordering confounds condition with run time.']),ensure_ascii=False,indent=2))
    print(json.dumps(agg,indent=2)); return agg

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();audit(a.input,a.out)
