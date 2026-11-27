import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import run_prediction_recovery as recovery


class UnlimitedRecoveryTests(unittest.TestCase):
    def test_unlimited_job_omits_memory_flags_but_keeps_oom_preference(self):
        command=recovery.job_command('image','/work-source','/cache-source',0,4)
        self.assertNotIn('--memory',command)
        self.assertNotIn('--memory-swap',command)
        self.assertEqual(command[command.index('--oom-score-adj')+1],'1000')

    def test_unlimited_oom_is_not_called_job_limit_exceeded(self):
        with tempfile.TemporaryDirectory() as directory:
            log=Path(directory)/'case.log';log.write_text('')
            result=recovery.classify(137,{'ExitCode':137,'OOMKilled':True},log,memory_limit_bytes=0)
        self.assertEqual(result['error_kind'],'oom_killed_without_job_limit')
        self.assertTrue(result['oom_killed'])

    def test_unlimited_rejects_parallel_or_pressure_kill_settings(self):
        for extra in (['--jobs','2','--kill-below-available-gib','0'],['--jobs','1']):
            with tempfile.TemporaryDirectory() as directory:
                target=Path(directory)/'not-created'
                argv=['run_prediction_recovery.py','--case-id','unused','--output-root',str(target),'--memory-gib','0',*extra]
                with patch.object(sys,'argv',argv),contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:recovery.main()
                self.assertEqual(raised.exception.code,2)
                self.assertFalse(target.exists())


if __name__=='__main__':unittest.main()
