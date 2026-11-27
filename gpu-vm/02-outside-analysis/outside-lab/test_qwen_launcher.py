import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import container_run_qwen as runner
from pilot import schedule

class QwenTests(unittest.TestCase):
    def args(self,phase,**kw):
        d=dict(sudo=False,image='cloudgpufit-bench:cu124-v1',expected_image_id=runner.BASELINE,
          instance_label='local-rtx3090',cache=None,env=[],phase=phase,tier='models',repeats=3,
          gpu_uuid='GPU-17eb7299-8b39-914d-7c5b-26908502904f',output=None,
          plan='qwen-plan.json',revisions='qwen-revisions.json',timeout=900)
        d.update(kw); return argparse.Namespace(**d)
    def invoke(self,a):
        info=json.dumps([{'Id':runner.BASELINE,'Config':{},'RepoDigests':[]}])
        with patch.object(runner.subprocess,'check_output',return_value=info),patch.object(runner.subprocess,'call',return_value=0) as call:
            runner.main(a); return call.call_args.args[0]
    def test_download_is_cpu_only_with_separate_revisions(self):
        with tempfile.TemporaryDirectory() as td,patch.object(runner.shutil,'disk_usage',return_value=argparse.Namespace(free=32*2**30)):
            cmd=self.invoke(self.args('cache-download',cache=td))
        self.assertNotIn('--gpus',cmd)
        self.assertIn('NVIDIA_VISIBLE_DEVICES=void',cmd)
        self.assertEqual(cmd[cmd.index('--revisions')+1],'/lab/qwen-revisions.json')
    def test_low_disk_stops_before_download(self):
        with tempfile.TemporaryDirectory() as td,patch.object(runner.shutil,'disk_usage',return_value=argparse.Namespace(free=14*2**30)),patch.object(runner.subprocess,'call') as call:
            with self.assertRaises(SystemExit): self.invoke(self.args('cache-download',cache=td))
            call.assert_not_called()
    def test_model_smoke_and_pilot_use_same_pins(self):
        for phase in ('smoke','pilot'):
            with tempfile.TemporaryDirectory() as td:
                cmd=self.invoke(self.args(phase,cache=td,output=str(Path(td)/'out')))
            self.assertIn('--pid=host',cmd)
            self.assertIn('HF_HUB_OFFLINE=1',cmd)
            self.assertEqual(cmd[cmd.index('--revisions')+1],'/lab/qwen-revisions.json')
            if phase=='smoke':
                self.assertEqual(cmd[cmd.index('--scenario')+1],'model')
                self.assertEqual(cmd[cmd.index('--case-id')+1],'core__decoder_llm__qwen7b__fp16__01')
    def test_schedule(self):
        plan=json.loads(Path(__file__).with_name('qwen-plan.json').read_text())
        jobs=schedule('models',3,[c['case_id'] for c in plan['cases']])
        self.assertEqual(len(jobs),12)
        self.assertEqual({j['repeat'] for j in jobs},{0,1,2})
if __name__=='__main__': unittest.main()
