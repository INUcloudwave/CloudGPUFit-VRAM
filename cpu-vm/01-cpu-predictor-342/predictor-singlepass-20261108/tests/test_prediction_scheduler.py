import importlib.util
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import prediction_scheduler as scheduler


class SchedulerTests(unittest.TestCase):
    def test_threshold_inclusive_and_reserve(self):
        self.assertTrue(scheduler.admission_allowed({'used_bytes':20,'available_bytes':30},20,6))
        self.assertFalse(scheduler.admission_allowed({'used_bytes':21,'available_bytes':30},20,6))
        self.assertFalse(scheduler.admission_allowed({'used_bytes':10,'available_bytes':5},20,6))

    def test_memory_accounting_uses_available_not_free(self):
        with patch.object(scheduler.Path,'read_text',return_value='MemTotal: 100 kB\nMemFree: 10 kB\nMemAvailable: 60 kB\n'):
            self.assertEqual(scheduler.host_memory(),{'total_bytes':102400,'available_bytes':61440,'used_bytes':40960})

    def test_high_memory_waits_then_resumes_without_failed_case(self):
        admissions=[];events=[];reads=0
        def memory():
            nonlocal reads
            reads+=1
            return {'used_bytes':21 if reads<3 else 20,'available_bytes':30}
        def execute(case,cancel):
            admissions.append(reads);return {'status':'ok'}
        recorded=[]
        flags=scheduler.run_cases([{'case_id':'a'}],execute,lambda c,r:recorded.append((c,r)),
            jobs=2,threshold_bytes=20,reserve_bytes=6,poll_seconds=.001,
            launch_interval_seconds=0,memory_reader=memory,notify=lambda event,**kw:events.append((event,kw)))
        self.assertEqual(admissions,[3]);self.assertEqual(len(recorded),1)
        self.assertEqual(recorded[0][1]['status'],'ok')
        self.assertEqual([e[1]['reason'] for e in events if e[0]=='wait'],['memory'])
        self.assertFalse(flags['interrupted'])

    def test_jobs_overlap_and_maximum_is_enforced(self):
        lock=threading.Lock();active=peak=0
        def execute(case,cancel):
            nonlocal active,peak
            with lock:active+=1;peak=max(peak,active)
            time.sleep(.02)
            with lock:active-=1
            return {'status':'ok'}
        completed=[]
        scheduler.run_cases([{'case_id':str(i)} for i in range(6)],execute,lambda c,r:completed.append(c['case_id']),
            jobs=2,threshold_bytes=20,reserve_bytes=6,poll_seconds=.001,
            launch_interval_seconds=0,memory_reader=lambda:{'used_bytes':2,'available_bytes':40})
        self.assertEqual(peak,2);self.assertEqual(sorted(completed),list('012345'))

    def test_memory_rise_blocks_third_start_and_does_not_cancel_running_jobs(self):
        active=0;reads=[];lock=threading.Lock();release=threading.Event();done=[]
        def memory():
            with lock:count=active
            reads.append(count)
            if count==2:release.set()
            return {'used_bytes':21 if count>=2 else 2,'available_bytes':30}
        def execute(case,cancel):
            nonlocal active
            with lock:active+=1
            release.wait(timeout=2)
            self.assertFalse(cancel.is_set())
            time.sleep(.01)
            with lock:active-=1
            return {'status':'ok'}
        scheduler.run_cases([{'case_id':str(i)} for i in range(3)],execute,lambda c,r:done.append(c['case_id']),
            jobs=3,threshold_bytes=20,reserve_bytes=6,poll_seconds=.001,
            launch_interval_seconds=.003,memory_reader=memory)
        self.assertIn(2,reads);self.assertEqual(len(done),3)

    def test_reference_stop_cancels_and_drains_peer_without_launching_pending(self):
        started=[];results=[];peer_started=threading.Event()
        def execute(case,cancel):
            started.append(case['case_id'])
            if case['case_id']=='a':
                peer_started.wait(timeout=2);return {'status':'ok'}
            peer_started.set()
            self.assertTrue(cancel.wait(timeout=2))
            return {'status':'error','error_kind':'cancelled'}
        def record(case,result):
            results.append((case['case_id'],result));return case['case_id']!='a'
        flags=scheduler.run_cases([{'case_id':c} for c in 'abc'],execute,record,
            jobs=2,threshold_bytes=20,reserve_bytes=6,poll_seconds=.001,
            launch_interval_seconds=0,memory_reader=lambda:{'used_bytes':2,'available_bytes':40})
        self.assertEqual(started,['a','b']);self.assertEqual(len(results),2)
        self.assertTrue(flags['validation_failed']);self.assertFalse(flags['interrupted'])

    def test_interrupt_reaches_all_active_workers_and_records_results(self):
        started=[];results=[];both=threading.Event()
        def execute(case,cancel):
            started.append(case['case_id'])
            if len(started)==2:both.set()
            self.assertTrue(cancel.wait(timeout=2))
            return {'status':'error','error_kind':'cancelled'}
        original_wait=scheduler.wait;raised=False
        def wait(*args,**kwargs):
            nonlocal raised
            if both.is_set() and not raised:
                raised=True;raise KeyboardInterrupt
            return original_wait(*args,**kwargs)
        with patch.object(scheduler,'wait',side_effect=wait):
            flags=scheduler.run_cases([{'case_id':c} for c in 'abc'],execute,lambda c,r:results.append(r),
                jobs=2,threshold_bytes=20,reserve_bytes=6,poll_seconds=.001,
                launch_interval_seconds=0,memory_reader=lambda:{'used_bytes':2,'available_bytes':40})
        self.assertEqual(started,['a','b']);self.assertEqual(len(results),2)
        self.assertTrue(flags['interrupted'])



class MemoryPressureGuardTests(unittest.TestCase):
    """Only the largest job is killed; peers continue; the victim re-runs alone once."""

    def harness(self, sizes, *, fail_again=False, low_until_killed=True):
        registry=scheduler.JobRegistry();lock=threading.Lock()
        kills=[];events=[];recorded={};live={};solo_overlap=[]
        stop={c:threading.Event() for c in sizes}
        def execute(case,cancel):
            cid=case['case_id'];attempt=case.get('attempt',1)
            with lock:
                live[cid]=attempt
                if attempt>1:solo_overlap.append(len(live))
            registry.register(cid,'job-'+cid,sizes[cid])
            try:
                killed=stop[cid].wait(timeout=.05 if attempt>1 else .3)
            finally:
                registry.unregister(cid)
                with lock:live.pop(cid)
            if killed and registry.pop_pressure_killed(cid):
                return {'status':'error','error_kind':'host_memory_pressure_killed','exit_code':137}
            if attempt>1 and fail_again:
                return {'status':'error','error_kind':'cgroup_oom'}
            return {'status':'ok','attempt':attempt}
        def kill(name):
            cid=name[4:];kills.append(cid)
            stop[cid].set()
            if not low_until_killed:return
        def memory():
            with lock:running=set(live)
            pressured=low_until_killed and not kills and len(running)>=2
            return {'used_bytes':2,'available_bytes':1 if pressured else 40}
        flags=scheduler.run_cases([{'case_id':c} for c in sizes],execute,
            lambda c,r:recorded.__setitem__(c['case_id'],(c.get('attempt',1),r)),
            jobs=len(sizes),threshold_bytes=20,reserve_bytes=6,poll_seconds=.001,launch_interval_seconds=0,
            memory_reader=memory,notify=lambda e,**kw:events.append((e,kw)),
            kill_below_available_bytes=4,registry=registry,kill_job=kill,
            job_memory_reader=lambda job:job['host_pid'],
            retry_solo=lambda c,r:r.get('error_kind') in ('host_memory_pressure_killed','cgroup_oom'))
        return flags,kills,events,recorded,solo_overlap

    def test_largest_job_killed_only_and_retried_alone(self):
        flags,kills,events,recorded,solo_overlap=self.harness({'small':1,'big':30,'mid':5})
        self.assertEqual(kills,['big'])
        self.assertEqual(recorded['small'],(1,{'status':'ok','attempt':1}))
        self.assertEqual(recorded['mid'],(1,{'status':'ok','attempt':1}))
        self.assertEqual(recorded['big'],(2,{'status':'ok','attempt':2}))
        self.assertEqual(solo_overlap,[1])
        self.assertEqual([e for e,_ in events].count('memory_pressure_kill'),1)
        self.assertEqual([kw['case_id'] for e,kw in events if e=='requeued_solo'],['big'])
        self.assertFalse(flags['interrupted'] or flags['validation_failed'])

    def test_no_kill_without_pressure(self):
        _,kills,events,recorded,_=self.harness({'a':1,'b':30},low_until_killed=False)
        self.assertEqual(kills,[]);self.assertEqual({k:v[0] for k,v in recorded.items()},{'a':1,'b':1})

    def test_retry_happens_at_most_once(self):
        _,kills,_,recorded,_=self.harness({'a':1,'b':30},fail_again=True)
        self.assertEqual(kills,['b'])
        self.assertEqual(recorded['b'],(2,{'status':'error','error_kind':'cgroup_oom'}))

    def test_disabled_guard_never_kills(self):
        registry=scheduler.JobRegistry();kills=[]
        def execute(case,cancel):
            registry.register(case['case_id'],'job',99);time.sleep(.02);registry.unregister(case['case_id'])
            return {'status':'ok'}
        scheduler.run_cases([{'case_id':'a'},{'case_id':'b'}],execute,lambda c,r:None,jobs=2,threshold_bytes=20,
            reserve_bytes=6,poll_seconds=.001,launch_interval_seconds=0,
            memory_reader=lambda:{'used_bytes':2,'available_bytes':0 if registry.snapshot() else 40},
            kill_below_available_bytes=0,registry=registry,kill_job=kills.append,job_memory_reader=lambda j:j['host_pid'])
        self.assertEqual(kills,[])

    def test_victim_is_largest_measurable(self):
        jobs=[{'case_id':'a','name':'a','host_pid':1},{'case_id':'b','name':'b','host_pid':2},
              {'case_id':'c','name':'c','host_pid':3}]
        victim=scheduler.choose_victim(jobs,lambda j:{1:10,2:None,3:7}[j['host_pid']])
        self.assertEqual((victim['case_id'],victim['memory_bytes']),('a',10))
        self.assertIsNone(scheduler.choose_victim(jobs,lambda j:None))

    def test_job_ram_prefers_anon_over_page_cache(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            cg=Path(directory)
            (cg/'memory.stat').write_text('anon 7000\nfile 900000\n')
            (cg/'memory.current').write_text('907000\n')
            with patch.object(scheduler,'_cgroup_dirs',return_value=[('v2',cg)]):
                self.assertEqual(scheduler.job_ram_bytes(123),7000)
            (cg/'memory.stat').write_text('file 1\n')
            with patch.object(scheduler,'_cgroup_dirs',return_value=[('v2',cg)]):
                self.assertEqual(scheduler.job_ram_bytes(123),907000)
        self.assertIsNone(scheduler.job_ram_bytes(None))

    def test_cgroup_path_parsing(self):
        with patch.object(scheduler.Path,'read_text',return_value='0::/system.slice/docker-abc.scope\n'):
            self.assertEqual(scheduler._cgroup_dirs(5),[('v2',Path('/sys/fs/cgroup/system.slice/docker-abc.scope'))])
        with patch.object(scheduler.Path,'read_text',return_value='4:cpu,cpuacct:/docker/x\n9:memory:/docker/x\n'):
            self.assertEqual(scheduler._cgroup_dirs(5),[('v1',Path('/sys/fs/cgroup/memory/docker/x'))])


class RetryPolicyTests(unittest.TestCase):
    def setUp(self):
        import run_prediction_recovery as recovery
        self.reason=recovery.retry_reason
        self.recovery=recovery

    def test_policy(self):
        g=1024**3
        self.assertEqual(self.reason({'status':'error','error_kind':'host_memory_pressure_killed'},1,32*g,32*g),
                         'killed_as_largest_job_under_host_memory_pressure')
        self.assertIsNone(self.reason({'status':'error','error_kind':'cgroup_oom'},1,32*g,32*g))
        self.assertEqual(self.reason({'status':'error','error_kind':'cgroup_oom'},1,40*g,32*g),'job_limit_raised_for_solo_retry')
        self.assertEqual(self.reason({'status':'error','error_kind':'process_exit_without_json','exit_code':137},1,32*g,32*g),
                         'sigkill_while_sharing_host')
        for kind in ('timeout','cancelled'):
            self.assertIsNone(self.reason({'status':'error','error_kind':kind,'exit_code':137},1,32*g,32*g))
        self.assertIsNone(self.reason({'status':'error','error_kind':'host_memory_pressure_killed'},2,32*g,32*g))
        self.assertIsNone(self.reason({'status':'error','error_kind':'predictor_error','exit_code':1},1,32*g,32*g))
        self.assertIsNone(self.reason({'status':'ok'},1,32*g,32*g))

    def test_job_containers_prefer_oom_kill_over_host_processes(self):
        command=self.recovery.job_command('image','/w','/c',8*1024**3,4)
        self.assertEqual(command[command.index('--oom-score-adj')+1],'1000')


if __name__=='__main__':unittest.main()
