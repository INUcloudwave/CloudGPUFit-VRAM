"""Same workloads and repeat counts; shared model only within one case/repeat."""
import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path

MODES = ["performance", "memory", "first_logits", "tokens", "components"]
COLUMNS = ["case_id", "model", "dtype", "params", "mode", "repeat", "status",
           "latency_p90_ms", "req_s", "output_tokens_s", "ttft_p90_ms", "tpot_p90_ms",
           "torch_peak_reserved_mib", "nvml_sampled_peak_mib", "weight_storage_mib",
           "self_kv_snapshot_peak_mib", "cross_kv_snapshot_peak_mib", "nvml_status",
           "model_reused", "worker_main_seconds"]

def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def select_cases(plan, profile, model=None, limit=None, case_set="all", case_ids=None):
    cases = [c for c in plan["cases"] if c["suite"] == profile["suite"]]
    if model:
        cases = [c for c in cases if c["model"] == model]
    if case_set in ("baseline9", "baseline27"):
        selected = {}
        for c in cases:
            if case_set == "baseline9" and c["dtype"] != "fp16":
                continue
            selected.setdefault((c["model"], c["dtype"]), c)
        cases = list(selected.values())
    if case_ids:
        known = {c["case_id"] for c in cases}
        missing = set(case_ids)-known
        if missing:
            raise ValueError(f"Requested cases not in selected suite: {sorted(missing)}")
        cases = [c for c in cases if c["case_id"] in set(case_ids)]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        cases = cases[:limit]
    if not cases or len({c['case_id'] for c in cases}) != len(cases):
        raise ValueError("No cases or duplicate case IDs")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", c['case_id']) for c in cases):
        raise ValueError("Unsafe case ID")
    return cases

def mode_groups(modes, isolation):
    if isolation == 'separate-mode':
        return [[m] for m in modes]
    if isolation == 'shared-case':
        return [list(modes)]
    shared = [m for m in modes if m != 'first_logits']
    return ([shared] if shared else []) + ([['first_logits']] if 'first_logits' in modes else [])

def shared_worker(group, isolation):
    return isolation == 'shared-case' or (isolation == 'hybrid' and group != ['first_logits'])

def model_load_count(cases, modes, isolation, repeats):
    return repeats * sum(any(c['workload_type'] != 'vision_cls' or m not in ('first_logits', 'tokens')
                            for m in group)
                         for c in cases for group in mode_groups(modes, isolation))

def summary_row(data):
    mem = data.get("memory_baseline") or {}
    snapshots = data.get("component_snapshots") or []
    def mib(x):
        return "" if x is None else x/2**20
    return {
        "case_id": data["case"]["case_id"], "model": data["case"]["model"],
        "dtype": data["case"]["dtype"], "params": json.dumps(data["case"]["params"]),
        "mode": data["mode"], "repeat": data["repeat"], "status": data["status"],
        "latency_p90_ms": (data.get("latency_ms") or {}).get("p90", ""),
        "req_s": (data.get("throughput") or {}).get("requests_per_second", ""),
        "output_tokens_s": (data.get("throughput") or {}).get("output_tokens_per_second", ""),
        "ttft_p90_ms": (data.get("ttft_ms") or {}).get("p90", ""),
        "tpot_p90_ms": (data.get("tpot_ms") or {}).get("p90", ""),
        "torch_peak_reserved_mib": mib(data.get("inference_peak_reserved_bytes")),
        "nvml_sampled_peak_mib": mib(data.get("nvml_process_sampled_peak_bytes")),
        "weight_storage_mib": mib(mem.get("parameter_storage_bytes")),
        "self_kv_snapshot_peak_mib": mib(max((x["self_kv_storage_bytes"] for x in snapshots), default=None)),
        "cross_kv_snapshot_peak_mib": mib(max((x["cross_kv_storage_bytes"] for x in snapshots), default=None)),
        "nvml_status": data.get("nvml_status", ""), "model_reused": data.get("model_reused", ""),
        "worker_main_seconds": data.get("worker_main_seconds", ""),
    }

class Progress:
    def __init__(self, cases, repeats, modes, reference, interval):
        self.start = time.perf_counter()
        self.case_done = self.mode_done = 0
        self.total_cases, self.total_modes = len(cases)*repeats, len(cases)*repeats*len(modes)
        self.current, self.phase = "preparation", "startup"
        self.weights = {}
        for case in cases:
            weight = 0
            if reference:
                for mode in modes:
                    f = reference/f'{case["case_id"]}__{mode}__r0.json'
                    if f.exists():
                        old = json.loads(f.read_text(encoding="utf-8"))
                        weight += sum(x['elapsed_ms']/1000 for x in old.get('raw_invocations', []))
            self.weights[case['case_id']] = max(weight, .1)
        self.total_weight = sum(self.weights.values())*repeats
        self.done_weight = 0
        self.case_clock = None
        self.interval = interval
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

    def text(self):
        with self.lock:
            elapsed = time.perf_counter()-self.start
            eta = "calibrating"
            if self.done_weight and self.case_clock:
                active = time.perf_counter()-self.case_clock
                estimate = active/self.done_weight*self.total_weight
                eta = f"~{max(0, estimate-active)/60:.1f}m (estimate)"
            return (f"[cases {self.case_done}/{self.total_cases} | modes {self.mode_done}/{self.total_modes}] "
                    f"elapsed={elapsed/60:.1f}m ETA={eta} case={self.current} phase={self.phase}")

    def start_heartbeat(self):
        if self.interval == 0:
            self.thread = None
            return
        def loop():
            while not self.stop_event.wait(self.interval):
                print(self.text(), flush=True)
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join()

def main():
    entry_begin = time.perf_counter()
    p = argparse.ArgumentParser()
    p.add_argument('--profile', choices=['local_smoke','local_core','aws_core'], default='local_core')
    p.add_argument('--plan', default='benchmark_plan_v2.json')
    p.add_argument('--image', default='cloudgpufit-bench:cu124-v1')
    p.add_argument('--instance-label', default='inugpu-rtx3090')
    p.add_argument('--modes', nargs='+', choices=MODES, default=MODES)
    p.add_argument('--isolation', choices=['hybrid','shared-case','separate-mode'], default='hybrid')
    p.add_argument('--model')
    p.add_argument('--limit', type=int)
    p.add_argument('--case-set', choices=['all','baseline9','baseline27'], default='all')
    p.add_argument('--case-id', nargs='+')
    p.add_argument('--reference', type=Path)
    p.add_argument('--threads', type=int)
    p.add_argument('--max-start-temp', type=float)
    p.add_argument('--no-temperature-wait', action='store_true')
    p.add_argument('--warmup', type=int)
    p.add_argument('--repeats', type=int)
    p.add_argument('--performance-iterations', type=int)
    p.add_argument('--memory-iterations', type=int)
    p.add_argument('--first-logits-iterations', type=int)
    p.add_argument('--token-iterations', type=int)
    p.add_argument('--idle-timeout', type=float, default=180)
    p.add_argument('--progress-seconds', type=float, default=30)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    root = Path(__file__).resolve().parent
    plan = json.loads(Path(a.plan).resolve().read_text(encoding='utf-8'))
    profile = dict(plan['execution_profiles'][a.profile])
    cases = select_cases(plan, profile, a.model, a.limit, a.case_set, a.case_id)
    modes = [m for m in MODES if m in a.modes]
    warmup = a.warmup if a.warmup is not None else profile.get('warmup', profile.get('warmup_invocations', 3))
    iterations = profile.get('iterations', profile.get('measured_invocations', 20))
    mode_iterations = dict(profile.get('mode_iterations', {}))
    for mode, value in [('performance',a.performance_iterations),('memory',a.memory_iterations),
                        ('first_logits',a.first_logits_iterations),('tokens',a.token_iterations)]:
        if value is not None:
            if value <= 0:
                raise ValueError('Per-mode iterations must be positive')
            mode_iterations[mode] = value
    repeats = a.repeats if a.repeats is not None else profile['repeats']
    profile.update(warmup=warmup,repeats=repeats,mode_iterations=mode_iterations)
    plan['execution_profiles'][a.profile] = profile
    if repeats <= 0 or iterations <= 0 or warmup < 0 or a.progress_seconds < 0:
        raise ValueError('Invalid repetitions or progress interval')
    applicable = sum(1 for c in cases for m in modes if c['workload_type']!='vision_cls' or m not in ('first_logits','tokens'))*repeats
    print(f'Cases: {len(cases)} Repeats: {repeats} Modes: {modes} Applicable measurements: {applicable}', flush=True)
    print(f'Isolation: {a.isolation}; planned model loads <= {model_load_count(cases,modes,a.isolation,repeats)}', flush=True)
    if a.dry_run:
        print(json.dumps({'case_ids':[c['case_id'] for c in cases], 'warmup':warmup, 'iterations':iterations,
                          'mode_iterations':mode_iterations,
                          'temperature_wait':a.max_start_temp is not None and a.max_start_temp>0 and not a.no_temperature_wait,
                          'mode_groups':mode_groups(modes,a.isolation),
                          'allocator_cache_policy':'retain_within_case' if a.isolation=='hybrid' else 'v2_control'}, indent=2))
        return
    if os.name != 'posix':
        raise RuntimeError('Run GPU jobs on the Linux VM; --dry-run is available elsewhere')
    threads = a.threads or len(os.sched_getaffinity(0))
    if threads <= 0:
        raise ValueError('threads must be positive')
    cache = root/'hf-cache'
    cache.mkdir(exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    folder = root/'results'/f'{stamp}_v3_{uuid.uuid4().hex[:8]}'
    folder.mkdir(parents=True)
    plan['cases'] = cases
    plan['execution_profiles'][a.profile]['modes'] = modes
    plan['runner_settings'].update(fresh_process_per_case_repeat_mode=a.isolation=='separate-mode',
                                   fresh_process_per_case_repeat=True, model_reused_between_modes=a.isolation!='separate-mode',
                                   first_logits_fresh_process=a.isolation!='shared-case',
                                   allocator_cache_policy='retain_within_case' if a.isolation=='hybrid' else 'v2_control')
    write(folder/'selected-plan.json', plan)
    worker = root/'bench_worker_v3.py'
    legacy = root/'bench_worker_legacy.py'
    image_id = subprocess.check_output(['sudo','docker','image','inspect',a.image,'--format','{{.Id}}'], text=True).strip()
    def gpu_state():
        x = subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.used,temperature.gpu,utilization.gpu',
                                     '--format=csv,noheader,nounits'], text=True).strip()
        return [float(v.strip()) for v in x.split(',')]
    baseline = gpu_state()
    max_temp = 0 if a.no_temperature_wait or a.max_start_temp is None else a.max_start_temp
    def wait_idle():
        begin = time.perf_counter()
        stable = 0
        while time.perf_counter()-begin < a.idle_timeout:
            state = gpu_state()
            good = state[0] <= baseline[0]+64 and (max_temp <= 0 or state[1] <= max_temp) and state[2] <= 5
            stable = stable+1 if good else 0
            if stable >= (1 if max_temp <= 0 else 3):
                return state, time.perf_counter()-begin
            time.sleep(1)
        raise RuntimeError('GPU idle/temperature timeout; no limits were relaxed')
    def base(selected_worker):
        return ['sudo','docker','run','--rm','--gpus','device=0','--pid=host',
                '--user',f'{os.getuid()}:{os.getgid()}',
                '-e',f'BENCH_IMAGE_ID={image_id}','-e',f'BENCH_INSTANCE_LABEL={a.instance_label}',
                '-e','HF_HUB_DISABLE_PROGRESS_BARS=1',
                '-e',f'BENCH_PROGRESS={1 if a.progress_seconds else 0}',
                '-v',f'{selected_worker}:/work/bench_worker.py:ro',
                '-v',f'{folder/"selected-plan.json"}:/plan.json:ro',
                '-v',f'{cache}:/cache/huggingface','-v',f'{folder}:/results']
    progress = Progress(cases,repeats,modes,a.reference,a.progress_seconds)
    progress.start_heartbeat()
    events = (folder/'events.jsonl').open('a',encoding='utf-8')
    host_timings = []
    case_timings = []
    manifest = None
    try:
        progress.phase = 'model_cache_preparation'
        begin = time.perf_counter()
        subprocess.run(base(worker)+[image_id,'python','/work/bench_worker.py','--prepare','/plan.json'],check=True)
        prepare_seconds = time.perf_counter()-begin
        progress.phase = 'environment_capture'
        lock = subprocess.check_output(['sudo','docker','run','--rm',image_id,'cat','/opt/environment.lock.txt'],text=True)
        (folder/'environment.lock.txt').write_text(lock,encoding='utf-8')
        manifest = {'profile':a.profile,'profile_settings':profile,'modes':modes,'threads':threads,
                    'image_id':image_id,'instance_label':a.instance_label,'host_os':Path('/etc/os-release').read_text(),
                    'runner_version':3,'isolation':a.isolation,
                    'mode_groups':mode_groups(modes,a.isolation),
                    'allocator_cache_policy':'retain_within_case' if a.isolation=='hybrid' else 'v2_control',
                    'worker_sha256':hashlib.sha256(worker.read_bytes()).hexdigest(),
                    'legacy_worker_sha256':hashlib.sha256(legacy.read_bytes()).hexdigest(),
                    'initial_gpu_state':baseline,'maximum_start_temperature_c':max_temp,
                    'reference':str(a.reference) if a.reference else None,
                    'prepare_seconds':prepare_seconds,'case_order':[c['case_id'] for c in cases],
                    'progress_policy':'stage boundaries; host heartbeat; no printing in timed invocation',
                    'counts_unchanged':not mode_iterations and a.warmup is None and a.repeats is None,
                    'components_iterations':1,
                    'temperature_wait_disabled':max_temp<=0}
        write(folder/'run-manifest.json',manifest)
        guard, unsupported, index = {}, set(), []
        fraction = plan['runner_settings'].get('memory_stop_fraction',.85)
        csv_file = (folder/'summary.csv').open('w',encoding='utf-8-sig',newline='')
        writer = csv.DictWriter(csv_file,fieldnames=COLUMNS)
        writer.writeheader()
        progress.case_clock = time.perf_counter()
        try:
            for case in cases:
                params = case['params']
                key = (case['model'],case['dtype'],tuple(sorted((k,v) for k,v in params.items() if k!='batch_size')))
                for repeat in range(repeats):
                    progress.current, progress.phase = case['case_id'], 'case_start'
                    case_begin = time.perf_counter()
                    case_start_wall_ns = time.time_ns()
                    name = lambda mode: f'{case["case_id"]}__{mode}__r{repeat}.json'
                    groups = mode_groups(modes,a.isolation)
                    invalid = None
                    for group in groups:
                        reported_modes = set()
                        reason = ('unsupported_dtype' if case['dtype'] in unsupported else
                                  'skipped_memory_guard' if params['batch_size']>=guard.get(key,float('inf')) else invalid)
                        active = [m for m in group if not(case['workload_type']=='vision_cls' and m in ('first_logits','tokens'))]
                        executed = not reason and bool(active)
                        state, idle_seconds, process_seconds = None, 0, 0
                        if executed:
                            progress.phase = 'gpu_idle_wait'
                            state,idle_seconds = wait_idle()
                            progress.phase = 'container_start_and_import'
                            use_shared_worker = shared_worker(group,a.isolation)
                            target_worker = worker if use_shared_worker else legacy
                            container_name = 'cgfit-'+uuid.uuid4().hex[:12]
                            command = base(target_worker)+['--name',container_name,'--network','none','-e','HF_HUB_OFFLINE=1',image_id,
                                'python','/work/bench_worker.py','--case',case['case_id'],'--repeat',str(repeat),
                                '--threads',str(threads),'--warmup',str(profile.get('warmup_by_type',{}).get(case['workload_type'],warmup)),
                                '--iterations',str(iterations if use_shared_worker else mode_iterations.get(group[0],iterations)),
                                '--memory-fraction',str(fraction)]
                            if use_shared_worker:
                                pattern = f'/results/{case["case_id"]}__{{mode}}__r{repeat}.json'
                                command += ['--modes',*group,'--output',pattern,'--max-start-temp',str(max_temp),
                                            '--idle-timeout',str(a.idle_timeout),'--mode-iterations-json',json.dumps(mode_iterations)]
                                if a.isolation=='hybrid':
                                    command += ['--keep-allocator-cache']
                            else:
                                command += ['--mode',group[0],'--output','/results/'+name(group[0])]
                            begin = time.perf_counter()
                            proc = subprocess.Popen(command,stdout=subprocess.PIPE,text=True,bufsize=1)
                            try:
                                for line in proc.stdout:
                                    if line.startswith('BENCH_EVENT '):
                                        event = json.loads(line[len('BENCH_EVENT '):])
                                        event['host_elapsed_seconds'] = time.perf_counter()-progress.start
                                        events.write(json.dumps(event)+'\n'); events.flush()
                                        progress.phase = event.get('phase',event['event'])
                                        if event['event']=='mode_complete':
                                            reported_modes.add(event['mode'])
                                            progress.mode_done += 1
                                            if a.progress_seconds:
                                                print(progress.text()+' mode='+event['mode']+' status='+event['status'],flush=True)
                                        elif a.progress_seconds:
                                            print(progress.text(),flush=True)
                                    else:
                                        print(line,end='',flush=True)
                                returncode = proc.wait()
                            except BaseException:
                                subprocess.run(['sudo','docker','stop','-t','5',container_name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                                proc.wait()
                                raise
                            process_seconds = time.perf_counter()-begin
                        for mode in group:
                            path = folder/name(mode)
                            inactive = case['workload_type']=='vision_cls' and mode in ('first_logits','tokens')
                            if not executed:
                                write(path,{'case':case,'mode':mode,'repeat':repeat,'status':'not_applicable' if inactive else reason})
                            elif not path.exists():
                                write(path,{'case':case,'mode':mode,'repeat':repeat,'status':'worker_failed','exit_code':returncode})
                            data = json.loads(path.read_text(encoding='utf-8'))
                            mode_state = state
                            idle = data.get('between_mode_idle')
                            if idle:
                                mode_state = [idle['device_used_mib'],idle['temperature_c'],idle['utilization_percent']]
                            data.update(case_start_gpu_state=state,start_gpu_state=mode_state,
                                        worker_exit_code=returncode if executed else None)
                            write(path,data)
                            status = data['status']
                            if status=='unsupported_dtype':
                                unsupported.add(case['dtype'])
                            if status in ('gpu_oom','skipped_memory_guard'):
                                guard[key] = min(guard.get(key,float('inf')),params['batch_size'])
                                invalid = 'skipped_memory_guard'
                            elif data.get('memory_guard_exceeded'):
                                guard[key] = min(guard.get(key,float('inf')),params['batch_size']+1)
                            if status in ('nonfinite_output','invalidated_by_component_check'):
                                invalid = 'skipped_invalid_output'
                            writer.writerow(summary_row(data)); csv_file.flush()
                            index.append({'case_id':case['case_id'],'mode':mode,'repeat':repeat,'status':status,'file':name(mode)})
                            if mode not in reported_modes:
                                progress.mode_done += 1
                            if status in ('gpu_busy','worker_failed','runtime_error','input_mismatch_between_modes'):
                                raise RuntimeError(f'Stop on {status}; inspect {path}')
                        host_timings.append({'case_id':case['case_id'],'repeat':repeat,'modes':group,
                                             'gpu_idle_wait_seconds':idle_seconds,'container_wall_seconds':process_seconds})
                        write(folder/'host-timings.json',host_timings)
                    write(folder/'index.json',index)
                    case_timings.append({'case_id':case['case_id'],'repeat':repeat,
                                         'wall_seconds':time.perf_counter()-case_begin,
                                         'start_wall_ns':case_start_wall_ns,'end_wall_ns':time.time_ns()})
                    write(folder/'case-timings.json',case_timings)
                    progress.case_done += 1
                    progress.done_weight += progress.weights[case['case_id']]
                    progress.phase = 'case_complete'
                    print(progress.text()+f' case_wall={time.perf_counter()-case_begin:.1f}s',flush=True)
        finally:
            csv_file.close()
        manifest.update(total_wall_seconds=time.perf_counter()-entry_begin,
                        completed_cases=progress.case_done,status='complete',
                        status_counts=dict(collections.Counter(x['status'] for x in index)))
        write(folder/'run-manifest.json',manifest)
    except BaseException as exc:
        if manifest is not None:
            manifest.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                            error=str(exc),total_wall_seconds=time.perf_counter()-entry_begin,
                            completed_cases=progress.case_done,completed_modes=progress.mode_done)
            write(folder/'run-manifest.json',manifest)
        raise
    finally:
        progress.close()
        events.close()
        print('Results:',folder,flush=True)

if __name__=='__main__':
    main()
