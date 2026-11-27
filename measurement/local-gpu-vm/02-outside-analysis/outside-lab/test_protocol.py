"""CPU-only tests of measurement accounting, observer guard and run isolation."""
import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
import probe
from pilot import schedule
from analyse_probe import clean,analyse

class NV:
    def __init__(self,compute=(),graphics=()):self.compute=compute;self.graphics=graphics
    def nvmlDeviceGetComputeRunningProcesses(self,h):return self.compute
    def nvmlDeviceGetGraphicsRunningProcesses(self,h):return self.graphics

class ProtocolTests(unittest.TestCase):
    def test_pid_isolation_and_no_device_memory_substitution(self):
        n=NV([NS(pid=10,usedGpuMemory=400),NS(pid=20,usedGpuMemory=900)])
        r=probe.nvml_snapshot(n,None,10,2000)
        self.assertEqual(r['nvml_process_bytes'],400);self.assertEqual(r['other_gpu_pids'],[20])
        self.assertIsNone(probe.nvml_snapshot(n,None,30,2000)['nvml_process_bytes'])

    def test_invalid_and_duplicate_values_preserved(self):
        n=NV([NS(pid=10,usedGpuMemory=2**64-1)])
        r=probe.nvml_snapshot(n,None,10,2000)
        self.assertIsNone(r['nvml_process_bytes']);self.assertEqual(r['nvml_own_raw'],[2**64-1])
        n=NV([NS(pid=10,usedGpuMemory=400)],[NS(pid=10,usedGpuMemory=400)])
        self.assertEqual(probe.nvml_snapshot(n,None,10,2000)['nvml_process_bytes'],400)
        n.graphics=[NS(pid=10,usedGpuMemory=401)]
        self.assertIsNone(probe.nvml_snapshot(n,None,10,2000)['nvml_process_bytes'])

    def test_preinit_guard_does_not_call_memory_apis(self):
        class CUDA:
            def is_initialized(self):return False
            def memory_allocated(self,*a):raise AssertionError('must not call')
            def memory_reserved(self,*a):raise AssertionError('must not call')
        a=NS(run_id='x',repeat=0,query_order='TNT')
        r=probe.Recorder(NV(),None,1000,a);r.torch=NS(cuda=CUDA())
        row=r.sample('stable')
        self.assertIsNone(row['reserved_bytes']);self.assertIsNone(row['outside_bytes'])

    def test_negative_not_clamped_and_order_timestamps(self):
        import os
        a=NS(run_id='x',repeat=0,query_order='TNT')
        r=probe.Recorder(NV([NS(pid=os.getpid(),usedGpuMemory=200)]),None,1000,a)
        r.torch=NS(cuda=NS(is_initialized=lambda:True,memory_allocated=lambda d:100,memory_reserved=lambda d:300))
        row=r.sample('stable');self.assertEqual(row['outside_bytes'],-100)
        self.assertIn('negative_outside',clean(row))
        self.assertLessEqual(row['torch_end_ns'],row['nvml_begin_ns'])
        self.assertLessEqual(row['nvml_end_ns'],row['torch_after_begin_ns'])

    def test_randomised_blocks_and_fresh_run_ids(self):
        jobs=schedule('core',3,[])
        self.assertEqual(len(jobs),30);self.assertEqual(jobs,schedule('core',3,[]))
        for rep in range(3):
            self.assertEqual(len({x['name'] for x in jobs if x['repeat']==rep}),10)

    def test_group_analysis_counts_processes_not_samples(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'input';out=Path(td)/'output'
            for rep in range(3):
                d=root/str(rep);d.mkdir(parents=True)
                args=dict(scenario='gemm',case_id=None,history='direct',sampling='continuous',api_probe=False,query_order='TNT',op_order='gemm,conv,attention')
                (d/'manifest.json').write_text(json.dumps(dict(run_id=str(rep),independent_repeat=rep,pid=rep,status='ok',args=args,gpu=dict(uuid='GPU-a',name='GPU'),environment={})))
                rows=[dict(stage='final_idle',kind='stable',outside_bytes=200*2**20,query_begin_ns=i*50_000_000,query_end_ns=i*50_000_000+1000) for i in range(21)]
                (d/'samples.jsonl').write_text('\n'.join(map(json.dumps,rows)))
            analyse(root,out)
            import csv
            with (out/'repeat_groups.csv').open() as f:groups=list(csv.DictReader(f))
            self.assertEqual(len(groups),1);self.assertEqual(groups[0]['n_processes'],'3')

if __name__=='__main__':unittest.main()
