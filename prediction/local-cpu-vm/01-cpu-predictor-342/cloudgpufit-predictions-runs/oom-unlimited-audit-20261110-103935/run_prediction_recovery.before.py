#!/usr/bin/env python3
"""Run selected CPU predictions with host-memory-gated concurrency in disposable, memory-limited containers.

The prepared xmem-cpu environment is snapshotted locally. Its mounted source and
all old results remain untouched. Each invocation requires a fresh output path.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from run_prediction_suite import ROOT, CONFIG, prediction_json, request, save, select_cases
from prediction_scheduler import JobRegistry, host_memory, job_ram_bytes, run_cases

SOURCES = ('predict_vram.py', 'cpu_tensor_trace.py', 'inference_trace_runtime.py',
           'vram_policy.py', 'gpu_overheads.json')
GIB = 1024 ** 3


def docker(*args, **kwargs):
    return subprocess.run(['docker', *args], check=True, text=True, encoding='utf-8',
                          errors='replace', capture_output=True, **kwargs)


def available_bytes():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('Cannot determine host MemAvailable')


def last_phase(log):
    result = None
    with Path(log).open(errors='replace') as source:
        for line in source:
            if line.startswith('CGF_PROGRESS '):
                try:
                    result = json.loads(line[len('CGF_PROGRESS '):]).get('phase', result)
                except ValueError:
                    pass
    return result


def classify(exit_code, state, log, *, timed_out=False, cancelled=False):
    """Return process evidence even when no predictor JSON was emitted."""
    result = {'status': 'error', 'exit_code': exit_code,
              'container_exit_code': state.get('ExitCode'),
              'oom_killed': bool(state.get('OOMKilled')), 'stage': last_phase(log)}
    if cancelled:
        result.update(error_kind='cancelled', error='Prediction cancelled; job container stopped')
    elif timed_out:
        result.update(error_kind='timeout', error='Prediction exceeded configured timeout')
    elif result['oom_killed']:
        result.update(error_kind='cgroup_oom', error='Job container exceeded its memory limit')
    else:
        try:
            parsed = prediction_json(Path(log).read_text(errors='replace'))
        except ValueError:
            result.update(error_kind='missing_prediction_json' if exit_code == 0 else 'process_exit_without_json',
                          error=f'Predictor JSON missing; docker exit={exit_code}; OOMKilled={result["oom_killed"]}')
        else:
            result.update(parsed)
            # Runtime evidence is authoritative even if a stale/misleading JSON says ok.
            result.update(exit_code=exit_code, container_exit_code=state.get('ExitCode'),
                          oom_killed=bool(state.get('OOMKilled')))
            if exit_code or state.get('ExitCode', 0):
                result.update(status='error', error_kind='process_exit',
                              error=parsed.get('error', f'Predictor process exited unsuccessfully: {exit_code}'))
            elif parsed['status'] != 'ok':
                result['error_kind'] = 'predictor_error'
    return result


def stop_job(name):
    try:
        docker('stop', '--time', '3', name, timeout=10)
    except (subprocess.SubprocessError, OSError):
        # Only the uniquely named container created for this attempt is removed.
        docker('kill', name, timeout=15)


def kill_job(name):
    """Immediate SIGKILL so the kernel reclaims the job's RAM without a grace period."""
    docker('kill', name, timeout=15)


def retry_reason(result, attempt, solo_memory_bytes, memory_bytes):
    """Why a failed first attempt deserves one solo re-run, or None."""
    if attempt != 1 or result.get('status') == 'ok' or result.get('error_kind') in ('cancelled', 'timeout'):
        return None
    if result.get('error_kind') == 'host_memory_pressure_killed':
        return 'killed_as_largest_job_under_host_memory_pressure'
    if result.get('error_kind') == 'cgroup_oom':
        # Its own limit was hit; retrying alone only helps with a larger limit.
        return 'job_limit_raised_for_solo_retry' if solo_memory_bytes > memory_bytes else None
    if 137 in (result.get('exit_code'), result.get('container_exit_code')):
        # SIGKILL not sent by this runner: most likely the host kernel OOM killer.
        return 'sigkill_while_sharing_host'
    return None


def run_job(name, command, payload, log, timeout, cancel_event=None, on_start=None):
    """The container PID 1 is the predictor, so Docker can attribute OOM kills."""
    proc = None
    timed_out = cancelled = False
    state = {}
    created = False
    try:
        docker('create', '--name', name, *command, timeout=30)
        created = True
        info = json.loads(docker('inspect', name, timeout=15).stdout)[0]
        metadata = {'container_id': info['Id'], 'container_name': name,
                    'memory_limit_bytes': info['HostConfig']['Memory'],
                    'memory_swap_bytes': info['HostConfig']['MemorySwap'], 'mounts': info['Mounts'],
                    'nano_cpus': info['HostConfig'].get('NanoCpus'),
                    'network_mode': info['HostConfig'].get('NetworkMode')}
        with Path(log).open('x') as output:
            proc = subprocess.Popen(['docker', 'start', '-a', '-i', name], stdin=subprocess.PIPE,
                                    stdout=output, stderr=subprocess.STDOUT, text=True)
            try:
                for _ in range(20):
                    active = json.loads(docker('inspect', '--format', '{{json .State}}', name, timeout=15).stdout)
                    host_pid = active.get('Pid', 0)
                    if host_pid:
                        metadata['host_pid'] = host_pid
                        try:
                            metadata['host_cgroup'] = Path(f'/proc/{host_pid}/cgroup').read_text()
                        except OSError:
                            pass
                        break
                    if 'Running' not in active or proc.poll() is not None:
                        break
                    time.sleep(0.05)
                if on_start is not None:
                    on_start(metadata)
                if cancel_event is None:
                    proc.communicate(json.dumps(payload), timeout=timeout)
                else:
                    deadline = time.monotonic() + timeout
                    first = True
                    while True:
                        if cancel_event.is_set():
                            cancelled = True
                            stop_job(name)
                            break
                        remaining = deadline-time.monotonic()
                        if remaining <= 0:
                            timed_out = True
                            stop_job(name)
                            break
                        try:
                            proc.communicate(json.dumps(payload) if first else None, timeout=min(1, remaining))
                            break
                        except subprocess.TimeoutExpired:
                            first = False
            except subprocess.TimeoutExpired:
                timed_out = True
                stop_job(name)
            except KeyboardInterrupt:
                cancelled = True
                stop_job(name)
            finally:
                if proc.poll() is None:
                    try:
                        proc.communicate(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=10)
        state = json.loads(docker('inspect', '--format', '{{json .State}}', name, timeout=15).stdout)
        result = classify(proc.returncode, state, log, timed_out=timed_out, cancelled=cancelled)
        result.update(container=metadata, container_state=state)
        return result
    finally:
        # Also cover errors while inspecting or starting the Docker CLI.
        try:
            if created:
                docker('rm', '-f', name, timeout=15)
        finally:
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)


def job_command(image, work, cache, memory_bytes, threads):
    # oom-score-adj 1000: if the host kernel OOM killer still fires, it picks a job
    # process (the one with the most RAM among equally adjusted jobs), not sshd/dockerd.
    args = ['-i', '--entrypoint', 'python', '--network', 'none', '--memory', str(memory_bytes),
            '--memory-swap', str(memory_bytes), '--oom-score-adj', '1000', '--cpus', str(threads),
            '--mount', f'type=bind,src={work},dst=/work,readonly',
            '--mount', f'type=bind,src={cache},dst=/root/.cache/huggingface,readonly']
    # A separate directory avoids creating missing mountpoints under read-only /work.
    args += ['--mount', f'type=bind,src={ROOT},dst=/recovery,readonly']
    for value in (f'OMP_NUM_THREADS={threads}', f'MKL_NUM_THREADS={threads}',
                  'OPENBLAS_NUM_THREADS=1', 'PYTHONDONTWRITEBYTECODE=1', 'HF_HUB_OFFLINE=1',
                  'PYTHONPATH=/recovery:/work/xMem'):
        args += ['-e', value]
    return args + ['-w', '/work/xMem', image, '-u', '/recovery/predict_vram.py', '--json', '-']


def choose_cases(plan, ids, remaining_from):
    cases = select_cases(plan, 'all')
    if ids:
        selected = set(ids)
        unknown = selected - {c['case_id'] for c in cases}
        if unknown:
            raise ValueError('Unknown case IDs: ' + ', '.join(sorted(unknown)))
        cases = [c for c in cases if c['case_id'] in selected]
    elif remaining_from:
        def succeeded(case):
            path = remaining_from / (case['case_id'] + '.prediction.json')
            return path.exists() and json.loads(path.read_text()).get('status') == 'ok'
        cases = [c for c in cases if not succeeded(c)]
    else:
        raise ValueError('Specify --case-id for validation or --remaining-from after validation')
    return cases


def compare_reference(case_id, payload, result, root, revision=None):
    if root is None or result.get('status') != 'ok':
        return {'status': 'not_requested' if root is None else 'prediction_failed'}
    prediction = root / (case_id + '.prediction.json')
    inputs = root / (case_id + '.input.json')
    if not prediction.exists() or not inputs.exists():
        return {'status': 'no_successful_reference'}
    old = json.loads(prediction.read_text())
    if old.get('status') != 'ok':
        return {'status': 'no_successful_reference'}
    old_input = json.loads(inputs.read_text())
    if old_input.get('request') != payload or (revision is not None and
            (old_input.get('revision') != revision or old.get('revision') != revision)):
        return {'status': 'incompatible_input'}
    names = ('predicted_allocated_bytes', 'predicted_reserved_bytes')
    deltas = {name: result[name] - old[name] for name in names}
    return {'status': 'equal' if all(v == 0 for v in deltas.values()) else 'mismatch',
            'delta_bytes': deltas, 'reference': str(prediction.resolve())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', required=True, type=Path)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--all', action='store_true', help='Run all 342 core cases, including old successes')
    selection.add_argument('--case-id', action='append', help='Exact case ID; repeat to select multiple cases')
    selection.add_argument('--remaining-from', type=Path, help='Read-only old results; select every non-successful case')
    parser.add_argument('--container', default='xmem-cpu')
    parser.add_argument('--snapshot-image', help='Reuse an explicit local snapshot after verifying its Python environment')
    parser.add_argument('--compare-with', type=Path, help='Read-only old results for exact allocated/reserved comparison')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--jobs', type=int, default=2, help='Maximum concurrent job containers')
    parser.add_argument('--start-below-used-gib', type=float, default=20, help='Admit a job only when host MemTotal-MemAvailable is at most this value')
    parser.add_argument('--poll-seconds', type=float, default=1)
    parser.add_argument('--launch-interval-seconds', type=float, default=5, help='Minimum stagger between concurrent starts; does not delay an idle queue')
    parser.add_argument('--memory-gib', type=float, default=32)
    parser.add_argument('--host-reserve-gib', type=float, default=6)
    parser.add_argument('--kill-below-available-gib', type=float, default=4,
                        help='While jobs run, if host MemAvailable falls below this, SIGKILL only the job using the most RAM; 0 disables')
    parser.add_argument('--solo-memory-gib', type=float,
                        help='Per-job limit when a pressure-killed/OOM case is re-run alone after the queue drains; default --memory-gib')
    parser.add_argument('--timeout-seconds', type=float, default=3600)
    args = parser.parse_args()
    if args.solo_memory_gib is None:
        args.solo_memory_gib = args.memory_gib
    if (args.threads < 1 or args.jobs < 1 or args.memory_gib <= 0 or args.host_reserve_gib < 0
            or args.timeout_seconds <= 0 or args.start_below_used_gib < 0
            or args.poll_seconds <= 0 or args.launch_interval_seconds < 0
            or args.kill_below_available_gib < 0 or args.solo_memory_gib < args.memory_gib):
        parser.error('Invalid resource limits')
    if args.kill_below_available_gib and args.kill_below_available_gib >= args.host_reserve_gib:
        parser.error('--kill-below-available-gib must be below --host-reserve-gib, or admission and kill would oscillate')
    out = args.output_root.resolve()
    if out.exists():
        parser.error('Use a new output directory; existing results are never overwritten')
    plan = json.loads((ROOT/'benchmarks/aws342/selected-plan.json').read_text())
    pins = json.loads((ROOT/'benchmarks/aws342/revisions.json').read_text())
    cases = select_cases(plan, 'all') if args.all else choose_cases(plan, args.case_id, args.remaining_from)
    if not cases:
        print('No incomplete cases selected')
        return 0
    lock = Path(f'/tmp/cloudgpufit-cpu-recovery-{os.getuid()}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Do not snapshot an actively computing or stopped environment.
    info = json.loads(docker('inspect', args.container, timeout=15).stdout)[0]
    if not info['State']['Running']:
        raise RuntimeError('The prepared CPU container must be running')
    top = docker('top', args.container, '-eo', 'pid,args', timeout=15).stdout
    if 'predict_vram.py' in top or 'run_prediction' in top:
        raise RuntimeError('A prediction is still running; existing work was left untouched')
    mounts = {m['Destination']: m['Source'] for m in info['Mounts'] if m['Type'] == 'bind'}
    work, cache = mounts['/work'], mounts['/root/.cache/huggingface']
    allocator_lock = json.loads((ROOT/'benchmarks/aws342/xmem-allocator-lock.json').read_text())
    check = 'import json,hashlib,pathlib,sys; names=json.load(sys.stdin); print(json.dumps({n:hashlib.sha256((pathlib.Path("/work/xMem")/n).read_bytes()).hexdigest() for n in names}))'
    hashes = json.loads(docker('exec', '-i', args.container, 'python', '-c', check,
                              input=json.dumps(list(allocator_lock['sha256'])), timeout=30).stdout)
    if hashes != allocator_lock['sha256']:
        raise RuntimeError('Prepared xMem allocator differs from pinned source')
    envcode = 'import json,torch,transformers,sys; print(json.dumps(dict(torch=torch.__version__,transformers=transformers.__version__,python=sys.version.split()[0],cuda=torch.version.cuda,gpu_available=torch.cuda.is_available())))'
    environment = json.loads(docker('exec', args.container, 'python', '-c', envcode, timeout=60).stdout)
    if (environment['torch'], environment['transformers'], environment['python'], environment['cuda'], environment['gpu_available']) != ('2.6.0+cu124', '4.52.4', '3.11.11', '12.4', False):
        raise RuntimeError('Prepared CPU environment differs from pinned versions: ' + str(environment))
    if args.snapshot_image:
        snapshot_environment = json.loads(docker('run', '--rm', '--network', 'none', '--memory', str(2*GIB),
            '--memory-swap', str(2*GIB), '--entrypoint', 'python', args.snapshot_image, '-c', envcode, timeout=60).stdout)
        if snapshot_environment != environment:
            raise RuntimeError('Snapshot environment differs from the prepared container')
    configs = {model: json.loads(docker('exec', args.container, 'python', '-c', CONFIG,
                                      model, pins[model]['revision'], timeout=120).stdout)
               for model in dict.fromkeys(c['model'] for c in cases)}
    memory_bytes = int(args.memory_gib * GIB)
    solo_memory_bytes = int(args.solo_memory_gib * GIB)
    reserve_bytes = int(args.host_reserve_gib * GIB)
    threshold_bytes = int(args.start_below_used_gib * GIB)
    kill_below_bytes = int(args.kill_below_available_gib * GIB)
    if reserve_bytes >= host_memory()['total_bytes']:
        raise RuntimeError('Host reserve must be smaller than total host RAM')
    token = uuid.uuid4().hex
    image = 'cloudgpufit-cpu-recovery:' + token
    out.mkdir(parents=True)
    git_commit = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    manifest = {'status': 'preparing', 'git_commit': git_commit, 'total_cases': len(cases), 'completed_cases': 0, 'success_count': 0,
                'environment': environment, 'base_image_id': info['Image'], 'allocator_lock': allocator_lock,
                'identity': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES},
                'plan_sha256': hashlib.sha256((ROOT/'benchmarks/aws342/selected-plan.json').read_bytes()).hexdigest(),
                'revisions_sha256': hashlib.sha256((ROOT/'benchmarks/aws342/revisions.json').read_bytes()).hexdigest(),
                'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'threads': args.threads, 'memory_limit_bytes': memory_bytes, 'host_reserve_bytes': reserve_bytes,
                'timeout_seconds': args.timeout_seconds, 'jobs': args.jobs,
                'admission_used_limit_bytes': threshold_bytes,
                'memory_usage_definition': 'MemTotal-MemAvailable',
                'poll_seconds': args.poll_seconds, 'launch_interval_seconds': args.launch_interval_seconds,
                'scheduler_sha256': hashlib.sha256((ROOT/'tools/prediction_scheduler.py').read_bytes()).hexdigest(),
                'pressure_kill_below_available_bytes': kill_below_bytes,
                'pressure_kill_policy': 'sigkill_largest_job_anon_ram_one_at_a_time',
                'solo_retry_memory_limit_bytes': solo_memory_bytes,
                'job_oom_score_adj': 1000,
                'pressure_kills': 0, 'solo_retries': 0,
                'reused_snapshot': args.snapshot_image,
                'selection': [c['case_id'] for c in cases], 'started_unix': time.time()}
    save(out/'manifest.json', manifest)
    start = time.monotonic()
    snapshot = False
    interrupted = False
    validation_failed = False
    try:
        if args.snapshot_image:
            image_id = docker('image', 'inspect', '--format', '{{.Id}}', args.snapshot_image, timeout=15).stdout.strip()
        else:
            docker('commit', '--pause=false', args.container, image, timeout=900)
            snapshot = True
            # Some Docker versions emit deprecation warnings on stdout.
            image_id = docker('image', 'inspect', '--format', '{{.Id}}', image, timeout=15).stdout.strip()
        manifest.update(status='running', snapshot_image_id=image_id)
        save(out/'manifest.json', manifest)
        numbers = {c['case_id']: i for i, c in enumerate(cases, 1)}
        registry = JobRegistry()
        first_attempts = {}

        def job_ram(job):
            pid = job.get('host_pid')
            if not pid:
                # The PID may not have been visible during the short start-up poll.
                try:
                    pid = int(docker('inspect', '--format', '{{.State.Pid}}', job['name'], timeout=15).stdout.strip())
                except (subprocess.SubprocessError, OSError, ValueError):
                    return None
            return job_ram_bytes(pid)

        def execute(case, cancel_event):
            cid = case['case_id']
            attempt = case.get('attempt', 1)
            limit = memory_bytes if attempt == 1 else solo_memory_bytes
            suffix = '' if attempt == 1 else f'.attempt{attempt}'
            payload = request(case, configs[case['model']], args.threads)
            if attempt == 1:
                save(out/(cid+'.input.json'), {'case_id': cid, 'revision': pins[case['model']]['revision'], 'request': payload})
            print(f'START {cid}' + ('' if attempt == 1 else f' (solo retry, limit {limit/GIB:.1f} GiB)'), flush=True)
            case_start = time.monotonic()
            name = 'cgf-cpu-' + token[:12] + '-' + str(numbers[cid]) + ('' if attempt == 1 else f'-a{attempt}')
            try:
                result = run_job(name, job_command(image_id, work, cache, limit, args.threads),
                                 payload, out/(cid+suffix+'.log'), args.timeout_seconds, cancel_event,
                                 on_start=lambda meta: registry.register(cid, name, meta.get('host_pid')))
            except Exception as exc:
                result = {'status': 'error', 'error_kind': 'runner_error', 'error': f'{type(exc).__name__}: {exc}'}
                if isinstance(exc, subprocess.CalledProcessError):
                    result['docker_stderr'] = (exc.stderr or '')[-4000:]
            finally:
                registry.unregister(cid)
            if registry.pop_pressure_killed(cid) and result.get('status') != 'ok':
                result.update(status='error', error_kind='host_memory_pressure_killed',
                              error='Killed by the runner as the largest-RAM job while host MemAvailable '
                                    f'was below {args.kill_below_available_gib:g} GiB')
            result.update(case_id=cid, revision=pins[case['model']]['revision'], attempt=attempt,
                          memory_limit_bytes=limit, runner_elapsed_seconds=time.monotonic()-case_start)
            return result

        def retry_solo(case, result):
            reason = retry_reason(result, case.get('attempt', 1), solo_memory_bytes, memory_bytes)
            if reason is None:
                return False
            first_attempts[case['case_id']] = {key: result.get(key) for key in (
                'status', 'error_kind', 'error', 'exit_code', 'container_exit_code', 'oom_killed',
                'stage', 'memory_limit_bytes', 'runner_elapsed_seconds')}
            first_attempts[case['case_id']].update(retry_reason=reason, log=case['case_id']+'.log')
            manifest['solo_retries'] += 1
            return True

        def record(case, result):
            cid = case['case_id']
            if cid in first_attempts:
                result['previous_attempt'] = first_attempts.pop(cid)
            payload = request(case, configs[case['model']], args.threads)
            comparison = compare_reference(cid, payload, result, args.compare_with, pins[case['model']]['revision'])
            result['reference_comparison'] = comparison
            save(out/(cid+'.prediction.json'), result)
            manifest['completed_cases'] += 1
            manifest['success_count'] += result['status'] == 'ok'
            manifest['elapsed_seconds'] = time.monotonic()-start
            save(out/'manifest.json', manifest)
            print(f'[{manifest["completed_cases"]}/{len(cases)}] {result["status"]} {cid} {result.get("error_kind", "")}', flush=True)
            if comparison['status'] in ('mismatch', 'incompatible_input'):
                print('STOP reference comparison: ' + json.dumps(comparison), flush=True)
                return False
            return True

        def notify(event, **fields):
            evidence = {'event': event, 'unix_time': time.time(), **fields}
            with (out/'scheduler.jsonl').open('a') as log:
                log.write(json.dumps(evidence)+'\n')
            manifest.update(running_cases=fields.get('running', []), pending_cases=fields.get('pending', 0))
            if 'memory' in fields:
                manifest['last_admission_memory'] = fields['memory']
            save(out/'manifest.json', manifest)
            if event == 'wait' and fields['reason'] == 'memory':
                print(f'WAIT host used={fields["memory"]["used_bytes"]/GIB:.2f} GiB; '
                      f'admission limit={args.start_below_used_gib:.2f} GiB; '
                      f'running={len(fields["running"])} pending={fields["pending"]}', flush=True)
            elif event == 'memory_pressure_kill':
                manifest['pressure_kills'] += 1
                save(out/'manifest.json', manifest)
                print(f'KILL {fields["case_id"]}: largest job ({fields["job_memory_bytes"]/GIB:.2f} GiB) while host '
                      f'available={fields["memory"]["available_bytes"]/GIB:.2f} GiB < '
                      f'{args.kill_below_available_gib:g} GiB; other jobs keep running', flush=True)
            elif event == 'requeued_solo':
                print(f'REQUEUE {fields["case_id"]} ({fields["error_kind"]}): re-run alone after the queue drains', flush=True)

        flags = run_cases(cases, execute, record, jobs=args.jobs,
                          threshold_bytes=threshold_bytes, reserve_bytes=reserve_bytes,
                          poll_seconds=args.poll_seconds, launch_interval_seconds=args.launch_interval_seconds,
                          notify=notify, kill_below_available_bytes=kill_below_bytes, registry=registry,
                          kill_job=kill_job, job_memory_reader=job_ram, retry_solo=retry_solo)
        interrupted, validation_failed = flags['interrupted'], flags['validation_failed']
    except KeyboardInterrupt:
        interrupted = True
    finally:
        # Safety net: no job container of this run may outlive the runner.
        try:
            leftovers = subprocess.run(['docker', 'ps', '-aq', '--filter', f'name=cgf-cpu-{token[:12]}-'],
                                       capture_output=True, text=True, timeout=30).stdout.split()
            if leftovers:
                subprocess.run(['docker', 'rm', '-f', *leftovers], capture_output=True, timeout=60)
            manifest['leftover_containers_removed'] = len(leftovers)
        except (subprocess.SubprocessError, OSError) as exc:
            manifest['leftover_container_check_error'] = f'{type(exc).__name__}: {exc}'
        if snapshot:
            subprocess.run(['docker', 'image', 'rm', image], capture_output=True, timeout=60)
        manifest.update(status='interrupted' if interrupted else 'validation_mismatch' if validation_failed else
                        'complete' if manifest['completed_cases'] == len(cases) and manifest['success_count'] == len(cases)
                        else 'complete_with_errors' if manifest['completed_cases'] == len(cases) else 'runner_failed',
                        elapsed_seconds=time.monotonic()-start)
        save(out/'manifest.json', manifest)
    return 130 if interrupted else 2 if validation_failed else 0 if manifest['status'] == 'complete' else 1


if __name__ == '__main__':
    def interrupt(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGHUP, interrupt)
    raise SystemExit(main())
