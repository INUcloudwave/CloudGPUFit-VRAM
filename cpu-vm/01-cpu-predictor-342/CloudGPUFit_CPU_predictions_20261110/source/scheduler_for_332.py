"""Admit concurrent predictions using observed host RAM, not case reservations."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
import threading
import time

GIB = 1024 ** 3


def host_memory():
    values = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, _, value = line.partition(':')
        if key in ('MemTotal', 'MemAvailable'):
            values[key] = int(value.split()[0]) * 1024
    total, available = values['MemTotal'], values['MemAvailable']
    return {'total_bytes': total, 'available_bytes': available,
            'used_bytes': total - available}


def admission_allowed(memory, threshold_bytes, reserve_bytes):
    return memory['used_bytes'] <= threshold_bytes and memory['available_bytes'] >= reserve_bytes


def _cgroup_dirs(host_pid):
    """Return (version, directory) pairs for the memory cgroup of a host PID."""
    dirs = []
    for line in Path(f'/proc/{host_pid}/cgroup').read_text().splitlines():
        _, controllers, path = line.split(':', 2)
        if controllers == '':
            dirs.append(('v2', Path('/sys/fs/cgroup') / path.lstrip('/')))
        elif 'memory' in controllers.split(','):
            dirs.append(('v1', Path('/sys/fs/cgroup/memory') / path.lstrip('/')))
    return dirs


def job_ram_bytes(host_pid):
    """Anonymous (non-reclaimable) RAM of the job's cgroup; page cache is excluded.

    Falls back to cgroup usage and then to the PID's VmRSS. Returns None when the
    process is gone, so a finished job is never selected as a kill victim.
    """
    if not host_pid:
        return None
    try:
        for version, directory in _cgroup_dirs(host_pid):
            stat_keys = ('anon',) if version == 'v2' else ('total_rss', 'rss')
            try:
                stats = dict(line.split() for line in (directory / 'memory.stat').read_text().splitlines())
                for key in stat_keys:
                    if key in stats:
                        return int(stats[key])
            except (OSError, ValueError):
                pass
            usage = 'memory.current' if version == 'v2' else 'memory.usage_in_bytes'
            try:
                return int((directory / usage).read_text())
            except (OSError, ValueError):
                pass
        for line in Path(f'/proc/{host_pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        return None
    return None


class JobRegistry:
    """Thread-safe map of running case IDs to their container name and host PID."""

    def __init__(self):
        self._lock = threading.Lock()
        self._jobs = {}
        self._pressure_killed = set()

    def register(self, case_id, name, host_pid=None):
        with self._lock:
            self._jobs[case_id] = {'case_id': case_id, 'name': name, 'host_pid': host_pid}

    def unregister(self, case_id):
        with self._lock:
            self._jobs.pop(case_id, None)

    def snapshot(self):
        with self._lock:
            return [dict(job) for job in self._jobs.values()]

    def mark_pressure_killed(self, case_id):
        with self._lock:
            self._pressure_killed.add(case_id)

    def pop_pressure_killed(self, case_id):
        """True once per pressure kill, so a solo retry is not mislabelled."""
        with self._lock:
            killed = case_id in self._pressure_killed
            self._pressure_killed.discard(case_id)
            return killed


def choose_victim(jobs, job_memory_reader):
    """Largest measured RAM user among registered jobs, or None if none is measurable."""
    measured = []
    for job in jobs:
        usage = job_memory_reader(job)
        if usage is not None:
            measured.append((usage, job))
    if not measured:
        return None
    usage, job = max(measured, key=lambda item: item[0])
    return dict(job, memory_bytes=usage)


def run_cases(cases, execute, record, *, jobs, threshold_bytes, reserve_bytes,
              poll_seconds=1, launch_interval_seconds=5, memory_reader=host_memory,
              notify=lambda event, **fields: None,
              kill_below_available_bytes=0, registry=None, kill_job=None,
              job_memory_reader=lambda job: job_ram_bytes(job.get('host_pid')),
              retry_solo=lambda case, result: False, kill_grace_seconds=30):
    """Only this thread admits/writes results; workers own their job containers.

    record() returns False to stop admission (e.g. an exact-reference mismatch).
    Cancellation reaches every worker, then all completed/cancelled results drain.
    The threshold is an admission rule, not a cap on future aggregate RAM growth.

    Memory-pressure guard: while jobs run, if host MemAvailable drops below
    kill_below_available_bytes, only the registered job using the most RAM is
    killed (one at a time). retry_solo(case, result) decides whether that case is
    re-run later with no other job active; each case is retried at most once.
    """
    pending = list(cases)
    solo = []
    retried = set()
    active = {}
    cancel = threading.Event()
    interrupted = validation_failed = False
    last_launch = float('-inf')
    waiting = None
    kill_in_flight = None
    guard = bool(kill_below_available_bytes) and registry is not None and kill_job is not None
    pool = ThreadPoolExecutor(max_workers=jobs)

    def running():
        return [c['case_id'] for c in active.values()]

    try:
        while pending or solo or active:
            try:
                for future in [f for f in active if f.done()]:
                    case = active.pop(future)
                    result = future.result()
                    if kill_in_flight and kill_in_flight[0] == case['case_id']:
                        kill_in_flight = None
                    if (not cancel.is_set() and case['case_id'] not in retried
                            and retry_solo(case, result)):
                        retried.add(case['case_id'])
                        solo.append(dict(case, attempt=case.get('attempt', 1)+1))
                        notify('requeued_solo', case_id=case['case_id'], error_kind=result.get('error_kind'),
                               running=running(), pending=len(pending)+len(solo))
                        continue
                    if record(case, result) is False:
                        validation_failed = True
                        cancel.set()
                    notify('finished', case_id=case['case_id'],
                           running=running(), pending=len(pending)+len(solo))
                if cancel.is_set():
                    pending.clear()
                    solo.clear()
                if guard and active:
                    if kill_in_flight and time.monotonic()-kill_in_flight[1] > kill_grace_seconds:
                        kill_in_flight = None
                    if kill_in_flight is None:
                        memory = memory_reader()
                        if memory['available_bytes'] < kill_below_available_bytes:
                            victim = choose_victim(registry.snapshot(), job_memory_reader)
                            if victim is not None:
                                registry.mark_pressure_killed(victim['case_id'])
                                kill_in_flight = (victim['case_id'], time.monotonic())
                                notify('memory_pressure_kill', case_id=victim['case_id'],
                                       container=victim['name'], job_memory_bytes=victim['memory_bytes'],
                                       memory=memory, running=running(), pending=len(pending)+len(solo))
                                try:
                                    kill_job(victim['name'])
                                except Exception as exc:  # the container may already be gone
                                    notify('memory_pressure_kill_failed', case_id=victim['case_id'],
                                           error=f'{type(exc).__name__}: {exc}', running=running(),
                                           pending=len(pending)+len(solo))
                # Solo retries start only after the normal queue is drained and nothing else runs.
                queue = pending if pending else (solo if not active else None)
                if queue and len(active) < jobs and kill_in_flight is None:
                    memory = memory_reader()
                    allowed = admission_allowed(memory, threshold_bytes, reserve_bytes)
                    staggered = active and time.monotonic()-last_launch < launch_interval_seconds
                    reason = 'memory' if not allowed else 'launch_interval' if staggered else None
                    if reason is None:
                        case = queue.pop(0)
                        future = pool.submit(execute, case, cancel)
                        active[future] = case
                        last_launch = time.monotonic()
                        waiting = None
                        notify('start', case_id=case['case_id'], attempt=case.get('attempt', 1), memory=memory,
                               running=running(), pending=len(pending)+len(solo))
                    elif waiting != reason:
                        waiting = reason
                        notify('wait', reason=reason, memory=memory,
                               running=running(), pending=len(pending)+len(solo))
                if active:
                    wait(active, timeout=poll_seconds, return_when=FIRST_COMPLETED)
                elif pending or solo:
                    time.sleep(poll_seconds)
            except KeyboardInterrupt:
                interrupted = True
                cancel.set()
                pending.clear()
                solo.clear()
                notify('cancellation_requested', running=running(), pending=0)
    finally:
        cancel.set()
        pool.shutdown(wait=True)
    return {'interrupted': interrupted, 'validation_failed': validation_failed}
