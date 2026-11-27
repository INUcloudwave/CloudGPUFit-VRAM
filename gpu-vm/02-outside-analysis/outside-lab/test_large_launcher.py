"""CPU-only tests that the custom plan reaches cache checking and GPU workers."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import container_run_large as runner
from pilot import schedule

class LargeLauncherTests(unittest.TestCase):
    def args(self,phase,**kw):
        d=dict(sudo=False,image='cloudgpufit-bench:cu124-v1',expected_image_id=runner.BASELINE,
            instance_label='local-rtx3090',cache=None,env=[],phase=phase,tier='models',repeats=3,
            gpu_uuid='GPU-17eb7299-8b39-914d-7c5b-26908502904f',output=None,
            plan='large-plan.json',timeout=600)
        d.update(kw);return argparse.Namespace(**d)

    def invoke(self,a):
        metadata=json.dumps([{'Id':runner.BASELINE,'Config':{},'RepoDigests':[]}])
        with patch.object(runner.subprocess,'check_output',return_value=metadata),patch.object(runner.subprocess,'call',return_value=0) as call:
            self.assertEqual(runner.main(a),0)
            return call.call_args.args[0]

    def test_plan_phase_has_no_gpu_and_uses_large_plan(self):
        cmd=self.invoke(self.args('plan'))
        self.assertNotIn('--gpus',cmd)
        self.assertEqual(cmd[cmd.index('--plan')+1],'/lab/large-plan.json')

    def test_cache_check_uses_large_plan_without_gpu(self):
        with tempfile.TemporaryDirectory() as cache:
            cmd=self.invoke(self.args('cache-check',cache=cache))
        self.assertIn('--check-only',cmd);self.assertNotIn('--gpus',cmd)
        self.assertIn('/lab/large-plan.json',cmd)

    def test_pilot_forwards_large_plan_and_timeout(self):
        with tempfile.TemporaryDirectory() as td:
            cmd=self.invoke(self.args('pilot',cache=td,output=str(Path(td)/'result')))
        self.assertIn('--gpus',cmd);self.assertIn('--pid=host',cmd)
        self.assertIn('/lab/large-plan.json',cmd);self.assertEqual(cmd[cmd.index('--timeout')+1],'600')

    def test_large_schedule_repeats_all_models_and_histories(self):
        p=Path(__file__).with_name('large-plan.json');cases=json.loads(p.read_text())['cases']
        ids=[c['case_id'] for c in cases];jobs=schedule('models',3,ids)
        self.assertEqual(len(jobs),48)
        for cid in ids:
            for hist in ('direct','performance_first'):
                self.assertEqual({j['repeat'] for j in jobs if j['case_id']==cid and j['history']==hist},{0,1,2})

if __name__=='__main__':unittest.main()
